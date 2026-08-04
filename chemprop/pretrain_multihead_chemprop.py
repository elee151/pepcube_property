"""
pretrain_multihead_chemprop.py
Pretrain a multi-head Chemprop MPNN on a peptide dataset.
Trains on the full dataset(subset name) and logs a train loss per epoch.
Encoder is saved down into specified dir.

Architecture:
    Shared encoder: BondMessagePassing (Chemeleon or ChemProp weights)
    5 RegressionFFN heads synthesizability_difficulty, instability, charge_at_pH_7.4, gravy, camsol_score

Args: --subset_name --cluster_col --pipeline_mode

Example usage:
    python pretrain_multihead_chemprop.py --subset_name all_generated_scored --cluster_col cluster_umap_hdb --pipeline_mode chemprop
"""

import argparse
import json
import logging
import os
from datetime import datetime
from functools import partial
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MODEL_ARCH", "multi_head")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn_torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from chemprop import featurizers
from chemprop import nn as cpnn
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

_RUN_DATE = datetime.now().strftime("%Y-%m-%d")
_JOB_ID   = os.environ.get("PBS_JOBID", "local").split(".")[0]

PRETRAIN_HEAD_KEYS = config.PRETRAIN_LABEL_COLS

# Sanitize column name for PyTorch ModuleDict (charge_at_pH_7.4 not accepted)
def _safe_key(name: str) -> str:
    return name.replace(".", "_")

SAFE_TO_COL = {_safe_key(k): k for k in PRETRAIN_HEAD_KEYS}
SAFE_KEYS    = list(SAFE_TO_COL.keys())


# Multi-head DMPNN (pretraining with 5 heads)
class PretrainMultiHeadMPNN(pl.LightningModule):
    """
    Shared BondMessagePassing encoder with one independent RegressionFFN head
    per scored property. NaN targets are masked from each head's loss.
    Forward returns z-scored predictions; scalers applied at inference via predict().
    """
    def __init__(self, mp, agg, heads: dict, scalers: dict,
                 lr: float = 6e-4, warmup_epochs: int = 2, max_epochs: int = 50):
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
        h = self.agg(self.message_passing(bmg), bmg.batch)
        return {SAFE_TO_COL[k]: head(h) for k, head in self.heads.items()}

    def predict(self, bmg):
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
        for name in PRETRAIN_HEAD_KEYS:
            l = self._masked_mse(preds[name].squeeze(-1), targets_dict[name])
            losses[f"{stage}_{name}_loss"] = l
            total = total + l
        losses[f"{stage}_loss"] = total / len(PRETRAIN_HEAD_KEYS)
        self.log_dict(
            losses,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
        )
        return losses[f"{stage}_loss"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

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


# DataLoader
class PretrainMultiHeadDataset(torch.utils.data.Dataset):
    def __init__(self, smiles, targets_dict):
        self.smiles       = smiles
        self.targets_dict = targets_dict

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        targets = {name: float(self.targets_dict[name][idx]) for name in PRETRAIN_HEAD_KEYS}
        return self.smiles[idx], targets


def collate_pretrain_multihead(batch, featurizer):
    smiles_list  = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]

    molgraphs = [featurizer(Chem.MolFromSmiles(smi)) for smi in smiles_list]
    bmg = BatchMolGraph(molgraphs)

    targets_dict = {}
    for name in PRETRAIN_HEAD_KEYS:
        vals = [t[name] for t in targets_list]
        def _to_float(v):
            try:
                f = float(v)
                return float("nan") if np.isnan(f) else f
            except (TypeError, ValueError):
                return float("nan")
        targets_dict[name] = torch.tensor(
            [_to_float(v) for v in vals], dtype=torch.float32
        )
    return bmg, targets_dict


def build_loader(dataset, featurizer, batch_size, shuffle):
    collate_fn = partial(collate_pretrain_multihead, featurizer=featurizer)
    generator = torch.Generator().manual_seed(config.SEED) if shuffle else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        persistent_workers=(config.NUM_WORKERS > 0),
        generator=generator,
    )


def load_encoder(pipeline_mode: str):
    """
    Initialise the message-passing encoder.
    "chemprop" : fresh BondMessagePassing, random weights
    """
    if pipeline_mode == "chemprop":
        logger.info(
            f"Encoder init: CHEMPROP"
            f"(depth={config.MPNN_DEPTH}, d_h={config.MPNN_D_H})"
        )
        return cpnn.BondMessagePassing(
            depth=config.MPNN_DEPTH,
            d_h=config.MPNN_D_H,
        )

    logger.info(f"Encoder init: CHEMELEON. Loading weights from {config.CHEMELEON_PATH}")
    ckpt = torch.load(config.CHEMELEON_PATH, map_location="cpu", weights_only=True)
    mp   = cpnn.BondMessagePassing(**ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    logger.info("Chemeleon weights loaded. All keys matched successfully")
    return mp


def build_model(mp, scalers):
    agg   = cpnn.MeanAggregation()
    #creates an FFN per task name, and uses mean aggregated loss
    heads = {
        _safe_key(name): cpnn.RegressionFFN(n_tasks=1, input_dim=mp.output_dim)
        for name in PRETRAIN_HEAD_KEYS
    }
    return PretrainMultiHeadMPNN(
        mp=mp,
        agg=agg,
        heads=heads,
        scalers=scalers,
        lr=config.PRETRAIN_LR,
        warmup_epochs=2,
        max_epochs=config.PRETRAIN_EPOCHS,
    )


# Scaler fitting on full dataset. done per head, ensures values are standard-scaled for training
def fit_scalers(df):
    scalers = {}
    for name in PRETRAIN_HEAD_KEYS:
        vals = df[name].dropna().values.reshape(-1, 1)
        if len(vals) == 0:
            raise ValueError(f"Head '{name}' has no non-NaN values.")
        sc = StandardScaler().fit(vals)
        scalers[name] = sc
        logger.info(f"  Scaler [{name}]: mean={sc.mean_}, scale={sc.scale_}")
    return scalers


def scale_targets(df, scalers):
    targets = {}
    for name in PRETRAIN_HEAD_KEYS:
        sc = scalers[name]
        arr = df[name].values
        out = np.full(len(arr), np.nan)
        mask = ~np.isnan(arr)
        if mask.any():
            out[mask] = sc.transform(arr[mask].reshape(-1, 1)).flatten()
        targets[name] = out
    return targets


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-head pretrain of Chemprop on scored subset")
    parser.add_argument("--subset_name", type=str, required=True)
    parser.add_argument("--cluster_col", type=str, required=True,
                        choices=["cluster_umap_hdb"])
    parser.add_argument("--pipeline_mode", type=str, default=None,
                        choices=["chemeleon", "pretrained", "chemprop"])
    return parser.parse_args()


def resolve_pipeline_mode(args) -> str:
    """calls pipeline mode to overwrite default in config"""
    if args.pipeline_mode is not None:
        os.environ["PIPELINE_MODE"] = args.pipeline_mode
        config.PIPELINE_MODE = args.pipeline_mode
        return args.pipeline_mode
    return config.PIPELINE_MODE


def main():
    args = parse_args()
    pipeline_mode = resolve_pipeline_mode(args)

    pl.seed_everything(config.SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Paths computed after pipeline_mode is resolved so dir name is correct
    output_dir = config.pretrain_dir(args.subset_name, pipeline_mode=pipeline_mode,
                                     model_arch="multi_head")
    final_dir  = output_dir / "final"
    log_dir    = config.get_checkpoint_root(pipeline_mode, model_arch="multi_head") / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    data_path    = config.DATA_DIR / f"{args.subset_name}.csv"
    subset_short = args.subset_name.replace("_cluster_umap_hdb", "").replace("_", "-")
    _log_name    = f"{_RUN_DATE}_job{_JOB_ID}_pretrain_mh_{subset_short}"

    logger.info(f"=== PRETRAIN multi-head (full dataset) ===")
    logger.info(f"Date/Job:        {_RUN_DATE} / {_JOB_ID}")
    logger.info(f"Pipeline mode:   {pipeline_mode}")
    logger.info(f"Output dir:      {output_dir}")
    logger.info(f"Subset:          {args.subset_name}")
    logger.info(f"Cluster col:     {args.cluster_col}")
    logger.info(f"Split strategy:  {config.SPLIT_STRATEGY}")
    logger.info(f"Accelerator:     {config.ACCELERATOR}")
    logger.info(f"Data:            {data_path}")
    logger.info(f"Heads:           {PRETRAIN_HEAD_KEYS}")
    logger.info(f"Epochs/Batch/LR: {config.PRETRAIN_EPOCHS} / {config.PRETRAIN_BATCH} / {config.PRETRAIN_LR}")

    df = pd.read_csv(data_path)
    required = [config.SMILES_COL, args.cluster_col] + PRETRAIN_HEAD_KEYS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].dropna(subset=[config.SMILES_COL, args.cluster_col]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows | {df[args.cluster_col].nunique()} unique groups")
    logger.info(f"Training on full dataset: {len(df)} molecules")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    smiles_arr = df[config.SMILES_COL].values

    scalers = fit_scalers(df)
    targets = scale_targets(df, scalers)

    full_ds     = PretrainMultiHeadDataset(smiles_arr, targets)
    full_loader = build_loader(full_ds, featurizer, config.PRETRAIN_BATCH, shuffle=True)

    mp    = load_encoder(pipeline_mode)
    model = build_model(mp, scalers)

    trainer = pl.Trainer(
        max_epochs=config.PRETRAIN_EPOCHS,
        accelerator=config.ACCELERATOR,
        devices=1,
        default_root_dir=str(final_dir),
        logger=pl.loggers.CSVLogger(str(final_dir), name=_log_name, version=0),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, full_loader)

    train_loss = trainer.callback_metrics.get("train_loss", float("nan"))
    train_loss = float(train_loss.item() if hasattr(train_loss, "item") else train_loss)
    logger.info(f"Final train loss (mean across heads): {train_loss:.4f}")

    ckpt_path = config.pretrain_checkpoint(args.subset_name, pipeline_mode=pipeline_mode,
                                           model_arch="multi_head")
    torch.save({
        "hyper_parameters": {
            "depth": model.message_passing.depth,
            "d_h":   model.message_passing.W_h.weight.shape[0],
        },
        "state_dict": model.state_dict(),
    }, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")

    scaler_data = {
        name: {"mean": sc.mean_, "scale": sc.scale_}
        for name, sc in scalers.items()
    }
    torch.save(scaler_data, final_dir / "scalers.pt")

    summary = {
        "run_date":        _RUN_DATE,
        "job_id":          _JOB_ID,
        "pipeline_mode":   pipeline_mode,
        "encoder":         "chemeleon" if pipeline_mode != "chemprop" else "chemprop",
        "model_arch":      "multi_head_pretrain",
        "subset_name":     args.subset_name,
        "cluster_col":     args.cluster_col,
        "split_strategy":  config.SPLIT_STRATEGY,
        "accelerator":     config.ACCELERATOR,
        "pretrain_epochs": config.PRETRAIN_EPOCHS,
        "pretrain_lr":     config.PRETRAIN_LR,
        "heads":           PRETRAIN_HEAD_KEYS,
        "n_total":         len(df),
        "train_loss":      train_loss,
        "output_dir":      str(output_dir),
    }
    with open(output_dir / "pretrain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
