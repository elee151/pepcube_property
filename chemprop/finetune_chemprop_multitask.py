"""
finetune_chemprop_multitask.py
Fine-tune a multihead Chemprop model on three regression tasks.

Environment variables read from config.py:
    PIPELINE_MODE: "chemeleon", "pretrained", or "scratch"
    ACCELERATOR: "cpu" or "gpu"
"""

import argparse
import json
import logging
import os
from datetime import datetime
from functools import partial
from pathlib import Path

os.environ.setdefault("MODEL_ARCH", "multi_head")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn_torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from chemprop import featurizers, nn as cpnn
from chemprop.data.collate import BatchMolGraph
from rdkit import Chem

import sys as _sys
from pathlib import Path as _Path
_p = _Path(__file__).resolve().parent
while _p.name != "pepcube_property" and _p != _p.parent:
    _p = _p.parent
if _p.name == "pepcube_property" and str(_p.parent) not in _sys.path:
    _sys.path.insert(0, str(_p.parent))

import pepcube_property.config as config
from pepcube_property.utils import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHEMELEON_SUBSET_LABEL = "chemeleon"
CHEMPROP_SUBSET_LABEL = "chemprop"

_RUN_DATE = datetime.now().strftime("%Y-%m-%d")
_JOB_ID   = os.environ.get("PBS_JOBID", "local").split(".")[0]

CLUSTER_COL = config.CLUSTER_COL

HEAD_KEYS = ["synthesizability", "camsol", "hemolysis"]


#  Multi-head module 

class MultiHeadMPNN(pl.LightningModule):

    def __init__(self, mp, agg, heads: dict, scalers: dict,
                 lr: float = 1e-4, warmup_epochs: int = 2, max_epochs: int = 50):
        super().__init__()
        self.message_passing = mp
        self.agg             = agg
        self.heads           = nn_torch.ModuleDict(heads)
        self.scalers         = scalers
        self.lr              = lr
        self.warmup_epochs   = warmup_epochs
        self.max_epochs      = max_epochs
        self.mse_loss        = nn_torch.MSELoss(reduction="none")

    def forward(self, bmg):
        """Returns z-scored predictions."""
        h = self.agg(self.message_passing(bmg), bmg.batch)
        return {name: head(h) for name, head in self.heads.items()}

    def predict(self, bmg):
        """Returns predictions with corresponding scaling."""
        z_preds = self(bmg)
        orig = {}
        for name, z in z_preds.items():
            sc    = self.scalers[name]
            mean  = torch.tensor(sc.mean_,  dtype=torch.float32, device=z.device)
            scale = torch.tensor(sc.scale_, dtype=torch.float32, device=z.device)
            orig[name] = z * scale + mean
        return orig

    def _masked_mse(self, pred, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return self.mse_loss(pred[mask], target[mask]).mean()

    def _shared_step(self, batch, stage: str):
        bmg, targets_dict = batch
        preds      = self(bmg)
        batch_size = int(bmg.batch.max().item()) + 1
        losses     = {}
        total      = torch.tensor(0.0, device=self.device)
        for name in HEAD_KEYS:
            l = self._masked_mse(preds[name].squeeze(-1), targets_dict[name])
            losses[f"{stage}_{name}_loss"] = l
            total = total + l
        losses[f"{stage}_loss"] = total / len(HEAD_KEYS)
        self.log_dict(
            losses,
            prog_bar=(stage == "val"),
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
        )
        return losses[f"{stage}_loss"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


#  Dataloader helpers

class MultiHeadDataset(torch.utils.data.Dataset):
    def __init__(self, smiles, targets_dict, featurizer):
        self.smiles       = smiles
        self.targets_dict = targets_dict
        self.featurizer   = featurizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi     = self.smiles[idx]
        targets = {name: float(self.targets_dict[name][idx]) for name in HEAD_KEYS}
        return smi, targets


def collate_multihead(batch, featurizer):
    smiles_list  = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]

    molgraphs = [featurizer(Chem.MolFromSmiles(smi)) for smi in smiles_list]
    bmg = BatchMolGraph(molgraphs)

    targets_dict = {}
    for name in HEAD_KEYS:
        vals = [t[name] for t in targets_list]
        def _to_float(v):
            try:
                f = float(v)
                return float("nan") if np.isnan(f) else f
            except (TypeError, ValueError):
                return float("nan")
        targets_dict[name] = torch.tensor(
            [_to_float(v) for v in vals], dtype=torch.float32,
        )
    return bmg, targets_dict


def build_loader(dataset, featurizer, batch_size, shuffle):
    collate_fn = partial(collate_multihead, featurizer=featurizer)
    generator = torch.Generator().manual_seed(config.SEED) if shuffle else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        persistent_workers=(config.NUM_WORKERS > 0),
        generator=generator,
    )


#  Encoder loading 

def load_encoder(encoder_ckpt):
    encoder_state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)

    if "hyper_parameters" in encoder_state and "state_dict" in encoder_state:
        raw_sd = encoder_state["state_dict"]
        hp     = encoder_state["hyper_parameters"]
        if any(k.startswith("message_passing.") for k in raw_sd):
            # MultiHeadMPNN checkpoint from pretrained model (encoder only)
            logger.info(f"  Loading full-model checkpoint (stripping message_passing. prefix) from {encoder_ckpt}")
            mp_state    = {k.replace("message_passing.", ""): v
                           for k, v in raw_sd.items()
                           if k.startswith("message_passing.")}
            hidden_size = mp_state["W_i.weight"].shape[0]
            depth = hp.get("depth")
            if depth is None:
                for v in hp.values():
                    if isinstance(v, dict) and "depth" in v:
                        depth = v["depth"]
                        break
            if depth is not None:
                logger.info(f"  Using depth={depth} from checkpoint hyper_parameters")
            else:
                depth = sum(1 for k in mp_state if k.startswith("W_h"))
                logger.warning(
                    f"  No depth found in hyper_parameters W_h keys (={depth})"
                )
            mp = cpnn.BondMessagePassing(d_h=hidden_size, depth=depth)
            missing, unexpected = mp.load_state_dict(mp_state, strict=False)
            logger.info(f"  Missing: {missing} | Unexpected: {unexpected}")
        else:
            # BondMessagePassing checkpoint if Chemeleon is used
            logger.info(f"  Loading Chemeleon / structured checkpoint from {encoder_ckpt}")
            mp = cpnn.BondMessagePassing(**hp)
            mp.load_state_dict(raw_sd)
            logger.info("  Weights loaded. All keys matched")
    else:
        logger.info(f"  Loading flat state-dict checkpoint from {encoder_ckpt}")
        mp_state    = {k.replace("message_passing.", ""): v
                       for k, v in encoder_state.items()
                       if k.startswith("message_passing.")}
        hidden_size = mp_state["W_i.weight"].shape[0]
        depth       = sum(1 for k in mp_state if k.startswith("W_h"))
        mp = cpnn.BondMessagePassing(d_h=hidden_size, depth=depth)
        missing, unexpected = mp.load_state_dict(mp_state, strict=False)
        logger.info(f"  Missing: {missing} | Unexpected: {unexpected}")

    return mp


#  Model builder 

def build_multihead_model(mp, scalers: dict, freeze_encoder: bool):
    agg   = cpnn.MeanAggregation()
    heads = {
        name: cpnn.RegressionFFN(n_tasks=1, input_dim=mp.output_dim)
        for name in HEAD_KEYS
    }
    model = MultiHeadMPNN(
        mp=mp,
        agg=agg,
        heads=heads,
        scalers=scalers,
        lr=config.FINETUNE_LR,
        warmup_epochs=2,
        max_epochs=config.FINETUNE_EPOCHS,
    )
    if freeze_encoder:
        logger.info("  Freezing message-passing encoder")
        for param in model.message_passing.parameters():
            param.requires_grad = False
    return model


#  Data merging + scaler fitting 

def load_and_merge_datasets(run_cfg):
    dfs = {}
    label_map = {}

    for head_key in HEAD_KEYS:
        ds_name   = run_cfg[head_key]
        ds_cfg    = config.DATASET_CONFIG[ds_name]
        label_col = ds_cfg["label_col"]
        label_map[head_key] = label_col

        df = pd.read_csv(ds_cfg["data_file"])
        required = [config.SMILES_COL, label_col, CLUSTER_COL]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[{ds_name}] Missing columns: {missing}")

        df = df[[config.SMILES_COL, CLUSTER_COL, label_col]].dropna(
            subset=[config.SMILES_COL, label_col]
        )
        df = df.rename(columns={label_col: f"label_{head_key}"})
        dfs[head_key] = df
        logger.info(f"  [{head_key}] {ds_name}: {len(df)} rows")

    merged = dfs[HEAD_KEYS[0]]
    for key in HEAD_KEYS[1:]:
        merged = merged.merge(
            dfs[key][[config.SMILES_COL, CLUSTER_COL, f"label_{key}"]],
            on=config.SMILES_COL,
            how="outer",
            suffixes=("", f"_{key}"),
        )

    cluster_cols = [CLUSTER_COL] + [f"{CLUSTER_COL}_{key}" for key in HEAD_KEYS[1:]]
    cluster_cols = [c for c in cluster_cols if c in merged.columns]
    merged[CLUSTER_COL] = merged[cluster_cols].bfill(axis=1).iloc[:, 0]
    merged = merged.drop(columns=[c for c in cluster_cols if c != CLUSTER_COL])
    merged = merged.dropna(subset=[config.SMILES_COL, CLUSTER_COL]).reset_index(drop=True)

    logger.info(
        f"  Merged: {len(merged)} unique SMILES | "
        f"synth={merged['label_synthesizability'].notna().sum()} "
        f"camsol={merged['label_camsol'].notna().sum()} "
        f"hemo={merged['label_hemolysis'].notna().sum()}"
    )
    return merged, label_map, dfs


def fit_scalers(merged_df, train_idx):
    """Fit one StandardScaler per head on the training subset."""
    scalers = {}
    for name in HEAD_KEYS:
        col        = f"label_{name}"
        train_vals = merged_df[col].iloc[train_idx].dropna().values.reshape(-1, 1)
        if len(train_vals) == 0:
            raise ValueError(f"Head '{name}' has no non-NaN training values — check data.")
        sc = StandardScaler().fit(train_vals)
        scalers[name] = sc
        logger.info(f"    Scaler [{name}]: mean={sc.mean_}, scale={sc.scale_}")
    return scalers


def scale_targets(merged_df, train_idx, val_idx, scalers, test_idx=None):
    """Scale targets for train/val using training set scalers."""
    train_targets, val_targets = {}, {}
    test_targets = {} if test_idx is not None else None

    for name in HEAD_KEYS:
        col = f"label_{name}"
        sc  = scalers[name]

        def _scale(arr):
            out = np.full(len(arr), np.nan)
            mask = ~np.isnan(arr)
            if mask.any():
                out[mask] = sc.transform(arr[mask].reshape(-1, 1)).flatten()
            return out

        train_targets[name] = _scale(merged_df[col].iloc[train_idx].values)
        val_targets[name]   = _scale(merged_df[col].iloc[val_idx].values)
        if test_idx is not None:
            test_targets[name] = _scale(merged_df[col].iloc[test_idx].values)

    if test_idx is not None:
        return train_targets, val_targets, test_targets
    return train_targets, val_targets


#  Argument parsing

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-head fine-tune of Chemprop")
    parser.add_argument("--run_id", type=str, required=True,
                        choices=list(config.MULTIHEAD_RUN_CONFIG.keys()))
    parser.add_argument("--subset_name", type=str, default=None)
    parser.add_argument("--pretrain_pipeline_mode", type=str, default=None,
                        choices=["chemeleon", "chemprop", "pretrained"])
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    return parser.parse_args()


def resolve_pretrain_source(args):
    if args.pretrain_ckpt:
        ckpt = Path(args.pretrain_ckpt)
        logger.info(f"Pipeline mode: PRETRAINED (--pretrain_ckpt override) — {ckpt}")
        return ckpt, args.subset_name or CHEMPROP_SUBSET_LABEL
    elif config.PIPELINE_MODE == "chemprop":
        logger.info("Pipeline mode: CHEMPROP — fresh encoder")
        return None, CHEMPROP_SUBSET_LABEL
    elif config.PIPELINE_MODE == "chemeleon":
        logger.info(f"Pipeline mode: CHEMELEON — Chemeleon from {config.CHEMELEON_PATH}")
        return config.CHEMELEON_PATH, CHEMELEON_SUBSET_LABEL
    else:
        if args.subset_name is None:
            raise ValueError("--subset_name required when PIPELINE_MODE=pretrained")
        pretrain_pm = args.pretrain_pipeline_mode or "scratch"
        ckpt = config.pretrain_checkpoint(args.subset_name,
                                          pipeline_mode=pretrain_pm,
                                          model_arch="multi_head")
        logger.info(f"Pipeline mode: PRETRAINED — {ckpt}")
        return ckpt, args.subset_name


#  Main

def main():
    args = parse_args()

    pl.seed_everything(config.SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_cfg      = config.MULTIHEAD_RUN_CONFIG[args.run_id]
    run_name     = run_cfg["run_name"]
    encoder_ckpt, subset_label = resolve_pretrain_source(args)
    output_dir   = config.finetune_dir(subset_label, run_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    _log_name = f"{_RUN_DATE}_job{_JOB_ID}_multihead_{run_name}"

    logger.info(f"=== FINETUNE (multi-head) ===")
    logger.info(f"Date/Job:        {_RUN_DATE} / {_JOB_ID}")
    logger.info(f"Run ID:          {args.run_id}  ({run_name})")
    logger.info(f"Pipeline mode:   {config.PIPELINE_MODE}")
    logger.info(f"Subset label:    {subset_label}")
    logger.info(f"Split strategy:  {config.SPLIT_STRATEGY}  |  N_FOLDS={config.N_FOLDS}")
    logger.info(f"Accelerator:     {config.ACCELERATOR}")
    logger.info(f"Epochs/Batch/LR: {config.FINETUNE_EPOCHS} / {config.FINETUNE_BATCH} / {config.FINETUNE_LR}")

    logger.info("Loading and merging datasets")
    merged_df, label_map, head_dfs = load_and_merge_datasets(run_cfg)

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    groups     = merged_df[CLUSTER_COL].values
    smiles_arr = merged_df[config.SMILES_COL].values

    per_dataset, fold_train_smiles, fold_val_smiles, test_smiles = make_multihead_splits(
        head_dfs, config.SMILES_COL, CLUSTER_COL, config.N_FOLDS,
        config.SPLIT_STRATEGY, config.SEED, test_frac=0.10)
    train_folds, val_folds, test_idx = resolve_multihead_indices(
        smiles_arr, fold_train_smiles, fold_val_smiles, test_smiles, config.N_FOLDS)

    logger.info(
        f"Split sizes — test: {len(test_idx)} | "
        f"fold 1 train: {len(train_folds[0])} | fold 1 val: {len(val_folds[0])}"
    )

    # Sanity check to confirm no dataset has an empty val fold
    for fold_i in range(config.N_FOLDS):
        for head_key, d in per_dataset.items():
            if len(d["val_folds"][fold_i]) == 0:
                logger.warning(f"  Fold {fold_i + 1}: '{head_key}' has 0 val samples!")

    # Save split indices so ChemBERTa can verify alignment
    split_meta = {
        "test_idx":       test_idx.tolist(),
        "train_folds":    [f.tolist() for f in train_folds],
        "val_folds":      [f.tolist() for f in val_folds],
        "n_total":        len(merged_df),
        "strategy":       config.SPLIT_STRATEGY,
        "seed":           config.SEED,
        "n_cv_folds":     config.N_FOLDS,
        "test_frac":      0.10,
        "split_method":   "per_dataset_independent_then_unioned",
    }
    with open(output_dir / "split_indices.json", "w") as f:
        json.dump(split_meta, f, indent=2)
    logger.info(f"Split indices saved: {output_dir / 'split_indices.json'}")

    #  CV folds
    fold_metrics = []

    for fold in range(config.N_FOLDS):
        logger.info(f"=== Fold {fold + 1}/{config.N_FOLDS} ===")
        fold_dir = output_dir / f"fold_{fold + 1}"
        fold_dir.mkdir(exist_ok=True)

        train_idx = train_folds[fold]
        val_idx   = val_folds[fold]

        scalers = fit_scalers(merged_df, train_idx)
        train_targets, val_targets = scale_targets(merged_df, train_idx, val_idx, scalers)

        train_ds = MultiHeadDataset(smiles_arr[train_idx], train_targets, featurizer)
        val_ds   = MultiHeadDataset(smiles_arr[val_idx],   val_targets,   featurizer)
        logger.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

        train_loader = build_loader(train_ds, featurizer, config.FINETUNE_BATCH, shuffle=True)
        val_loader   = build_loader(val_ds,   featurizer, config.FINETUNE_BATCH, shuffle=False)

        if encoder_ckpt:
            mp = load_encoder(encoder_ckpt)
        else:
            mp = cpnn.BondMessagePassing(depth=config.MPNN_DEPTH, d_h=config.MPNN_D_H)
            logger.info(f"  Default encoder: depth={config.MPNN_DEPTH}, d_h={config.MPNN_D_H}")
        model = build_multihead_model(mp, scalers, config.FREEZE_ENCODER)

        trainer = pl.Trainer(
            max_epochs=config.FINETUNE_EPOCHS,
            accelerator=config.ACCELERATOR,
            devices=1,
            default_root_dir=str(fold_dir),
            logger=pl.loggers.CSVLogger(
                str(fold_dir), name=f"{_log_name}_fold{fold+1}", version=0
            ),
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_loader, val_loader)

        val_loss = trainer.callback_metrics.get("val_loss", float("nan"))
        val_loss = float(val_loss.item() if hasattr(val_loss, "item") else val_loss)
        fold_metrics.append({"fold": fold + 1, "val_loss": val_loss})
        logger.info(f"  Fold {fold + 1} val_loss (z-score, mean across heads): {val_loss:.4f}")

        torch.save(model.state_dict(), fold_dir / "finetuned_model.pt")
        scaler_data = {
            name: {"mean": sc.mean_, "scale": sc.scale_}
            for name, sc in scalers.items()
        }
        torch.save(scaler_data, fold_dir / "scalers.pt")

    losses    = [m["val_loss"] for m in fold_metrics if not np.isnan(m["val_loss"])]
    mean_loss = float(np.mean(losses)) if losses else None
    std_loss  = float(np.std(losses))  if losses else None
    if losses:
        logger.info(f"CV val_loss: {mean_loss:.4f} ± {std_loss:.4f}")

    cv_summary = {
        "run_date":        _RUN_DATE,
        "job_id":          _JOB_ID,
        "model_arch":      "multi_head",
        "run_id":          args.run_id,
        "run_name":        run_name,
        "pipeline_mode":   config.PIPELINE_MODE,
        "subset_label":    subset_label,
        "cluster_col":     CLUSTER_COL,
        "split_strategy":  config.SPLIT_STRATEGY,
        "head_datasets":   {k: run_cfg[k] for k in HEAD_KEYS},
        "accelerator":     config.ACCELERATOR,
        "freeze_encoder":  config.FREEZE_ENCODER,
        "finetune_epochs": config.FINETUNE_EPOCHS,
        "finetune_lr":     config.FINETUNE_LR,
        "fold_metrics":    fold_metrics,
        "mean_val_loss":   mean_loss,
        "std_val_loss":    std_loss,
    }
    with open(output_dir / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    #  Final model (train on full trainval)
    logger.info("Training final model on full 90% train+val data")
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)

    # Reuse the already test-excluded CV fold indices (from resolve_multihead_indices)
    trainval_idx = np.sort(np.concatenate([train_folds[0], val_folds[0]]))

    final_scalers = fit_scalers(merged_df, trainval_idx)
    final_train_targets, _ = scale_targets(
        merged_df, trainval_idx, trainval_idx, final_scalers
    )

    final_train_ds = MultiHeadDataset(smiles_arr[trainval_idx], final_train_targets, featurizer)
    final_train_loader = build_loader(final_train_ds, featurizer, config.FINETUNE_BATCH, shuffle=True)
    final_val_loader = None

    if encoder_ckpt:
        final_mp = load_encoder(encoder_ckpt)
    else:
        final_mp = cpnn.BondMessagePassing(depth=config.MPNN_DEPTH, d_h=config.MPNN_D_H)
    final_model = build_multihead_model(final_mp, final_scalers, config.FREEZE_ENCODER)

    final_trainer = pl.Trainer(
        max_epochs=config.FINETUNE_EPOCHS,
        accelerator=config.ACCELERATOR,
        devices=1,
        default_root_dir=str(final_dir),
        logger=pl.loggers.CSVLogger(
            str(final_dir), name=f"{_log_name}_final", version=0
        ),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    final_trainer.fit(final_model, final_train_loader, final_val_loader)

    torch.save(final_model.state_dict(), final_dir / "finetuned_model.pt")
    torch.save({name: {"mean": sc.mean_, "scale": sc.scale_}
                for name, sc in final_scalers.items()}, final_dir / "scalers.pt")
    logger.info(f"Final model saved: {final_dir / 'finetuned_model.pt'}")

if __name__ == "__main__":
    main()
