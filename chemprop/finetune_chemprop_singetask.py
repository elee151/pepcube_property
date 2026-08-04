"""
finetune_chemprop_singetask.py
Fine-tune a pretrained (or Chemeleon-direct) Chemprop model with a single task per model.
One MPNN + one RegressionFFN head trained per dataset.

Usage examples:
    python finetune_chemprop_singetask.py --dataset_name hemolysis --fold 1
    python finetune_chemprop_singetask.py --dataset_name hemolysis
"""

import argparse
import copy
import json
import logging
import os
from datetime import datetime

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from chemprop import data, featurizers, nn
from chemprop.data import split_data_by_indices
from chemprop.models import MPNN

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

def parse_args():
    parser = argparse.ArgumentParser(description="Single-task fine-tune of Chemprop")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument("--subset_name", type=str, default=None,
                        help="Pretraining subset (required for PIPELINE_MODE=pretrained)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run a single fold only (1-indexed). Saves split_indices.json on fold 1.")
    parser.add_argument("--final_only", action="store_true",
                        help="Skip CV folds; train final model only. Requires split_indices.json to exist.")
    return parser.parse_args()


def resolve_pretrain_source(args):
    if config.PIPELINE_MODE == "chemprop":
        logger.info("Pipeline mode: CHEMPROP — initialising fresh Chemprop encoder")
        return None, CHEMPROP_SUBSET_LABEL
    elif config.PIPELINE_MODE == "chemeleon":
        logger.info(f"Pipeline mode: CHEMELEON — loading Chemeleon weights from {config.CHEMELEON_PATH}")
        return config.CHEMELEON_PATH, CHEMELEON_SUBSET_LABEL
    else:
        if args.subset_name is None:
            raise ValueError("--subset_name is required when PIPELINE_MODE=pretrained")
        ckpt = config.pretrain_checkpoint(args.subset_name)
        logger.info(f"Pipeline mode: PRETRAINED — loading checkpoint from {ckpt}")
        return ckpt, args.subset_name


def build_model(train_dataset, val_dataset, encoder_ckpt):
    if encoder_ckpt is None:
        logger.info("  Initialising fresh BondMessagePassing encoder (Chemprop default)")
        mp = nn.BondMessagePassing()
    else:
        encoder_state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
        if "hyper_parameters" in encoder_state and "state_dict" in encoder_state:
            logger.info("  Loading Chemeleon checkpoint")
            mp = nn.BondMessagePassing(**encoder_state["hyper_parameters"])
            mp.load_state_dict(encoder_state["state_dict"])
            logger.info("  Chemeleon weights loaded. All keys match")
        else:
            logger.info("  Loading pretrained model checkpoint")
            mp_state = {
                k.replace("message_passing.", ""): v
                for k, v in encoder_state.items()
                if k.startswith("message_passing.")
            }
            hidden_size = mp_state["W_i.weight"].shape[0]
            logger.info(f"  Detected hidden_size={hidden_size} from checkpoint")
            mp = nn.BondMessagePassing(d_h=hidden_size)
            missing, unexpected = mp.load_state_dict(mp_state, strict=False)
            logger.info(f"  Encoder loaded. Missing: {missing}. Unexpected: {unexpected}")

    agg = nn.MeanAggregation()

    scaler = train_dataset.normalize_targets()
    logger.info(f"  Scaler fitted: mean={scaler.mean_}, scale={scaler.scale_}")
    if val_dataset is not None:
        val_dataset.normalize_targets(scaler)

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(
        n_tasks=1,
        output_transform=output_transform,
        input_dim=mp.output_dim,
    )
    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        warmup_epochs=2,
        init_lr=config.FINETUNE_LR / 10,
        max_lr=config.FINETUNE_LR,
        final_lr=config.FINETUNE_LR / 100,
    )

    if config.FREEZE_ENCODER:
        logger.info("  Freezing message-passing encoder")
        for param in model.message_passing.parameters():
            param.requires_grad = False

    return model, scaler


def run_trainer(model, train_data, val_data, log_dir, log_name: str):
    train_loader = data.build_dataloader(
        train_data, batch_size=config.FINETUNE_BATCH, shuffle=True,
        num_workers=config.NUM_WORKERS, seed=config.SEED,
    )
    val_loader = None
    if val_data is not None:
        val_loader = data.build_dataloader(
            val_data, batch_size=config.FINETUNE_BATCH, shuffle=False, drop_last=False,
            num_workers=config.NUM_WORKERS,
        )
    trainer = pl.Trainer(
        max_epochs=config.FINETUNE_EPOCHS,
        accelerator=config.ACCELERATOR,
        devices=1,
        default_root_dir=str(log_dir),
        logger=pl.loggers.CSVLogger(str(log_dir), name=log_name, version=0),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)
    return trainer


def train_final_model(args, df, datapoints, groups, featurizer, encoder_ckpt, subset_label, output_dir):
    """Train final model on 90% of data (all train+val)."""
    logger.info("Training final model on 90% train+val data")
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)

    _log_name = f"{_RUN_DATE}_job{_JOB_ID}_singletask_{args.dataset_name}"

    # Load split indices to get the test holdout
    split_path = output_dir / "split_indices.json"
    if split_path.exists():
        with open(split_path) as f:
            split_meta = json.load(f)
        test_idx       = np.array(split_meta["test_idx"], dtype=int)
        train_val_mask = np.ones(len(df), dtype=bool)
        train_val_mask[test_idx] = False
        train_val_idx  = np.where(train_val_mask)[0]
    else:
        raise FileNotFoundError(
            f"{split_path} not found."
        )

    datapoints_final = copy.deepcopy(datapoints)
    full_train_dps, _, _ = split_data_by_indices(
        datapoints_final,
        train_indices=[train_val_idx],
        val_indices=None,
        test_indices=None,
    )
    full_train_data = data.MoleculeDataset(full_train_dps[0], featurizer=featurizer)
    logger.info(f"  Final train: {len(full_train_data)}")

    final_model, final_scaler = build_model(full_train_data, None, encoder_ckpt)
    run_trainer(final_model, full_train_data, None, final_dir,
                log_name=f"{_log_name}_final")

    ckpt_path = config.finetune_checkpoint(subset_label, args.dataset_name)
    torch.save(final_model.state_dict(), ckpt_path)
    torch.save({"mean": final_scaler.mean_, "scale": final_scaler.scale_}, final_dir / "scaler.pt")
    logger.info(f"Final model saved: {ckpt_path}")


def main():
    args = parse_args()

    pl.seed_everything(config.SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ds_cfg       = config.DATASET_CONFIG[args.dataset_name]
    label_col    = ds_cfg["label_col"]
    data_path    = ds_cfg["data_file"]
    encoder_ckpt, subset_label = resolve_pretrain_source(args)
    output_dir   = config.finetune_dir(subset_label, args.dataset_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    _log_name = f"{_RUN_DATE}_job{_JOB_ID}_singletask_{args.dataset_name}"

    if args.final_only:
        mode_str = "final model only"
    elif args.fold:
        mode_str = f"fold {args.fold} only"
    else:
        mode_str = "all folds + final"

    logger.info(f"=== FINETUNE (single-task) | {mode_str} ===")
    logger.info(f"Date/Job:        {_RUN_DATE} / {_JOB_ID}")
    logger.info(f"Pipeline mode:   {config.PIPELINE_MODE}")
    logger.info(f"Subset label:    {subset_label}")
    logger.info(f"Dataset:         {args.dataset_name}")
    logger.info(f"Cluster col:     {CLUSTER_COL}")
    logger.info(f"Split strategy:  {config.SPLIT_STRATEGY}")
    logger.info(f"Accelerator:     {config.ACCELERATOR}")
    logger.info(f"Freeze encoder:  {config.FREEZE_ENCODER}")
    logger.info(f"Epochs/Batch/LR: {config.FINETUNE_EPOCHS} / {config.FINETUNE_BATCH} / {config.FINETUNE_LR}")

    df = pd.read_csv(data_path)
    for col in [config.SMILES_COL, label_col, CLUSTER_COL]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
    df = df[[config.SMILES_COL, label_col, CLUSTER_COL]].dropna().reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows. {df[CLUSTER_COL].nunique()} unique clusters")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    datapoints = [
        data.MoleculeDatapoint.from_smi(smi, [y])
        for smi, y in zip(df[config.SMILES_COL], df[label_col])
    ]
    groups = df[CLUSTER_COL].values

    # Final-only mode
    if args.final_only:
        train_final_model(args, df, datapoints, groups, featurizer,
                          encoder_ckpt, subset_label, output_dir)
        return

    # Compute / load splits
    split_path = output_dir / "split_indices.json"

    if split_path.exists():
        with open(split_path) as f:
            split_meta = json.load(f)
        train_idx_folds = [np.array(f, dtype=int) for f in split_meta["train_folds"]]
        val_idx_folds   = [np.array(f, dtype=int) for f in split_meta["val_folds"]]
        test_idx        = np.array(split_meta["test_idx"], dtype=int)

    else:
        train_idx_folds, val_idx_folds, test_idx = make_train_val_test_indices(
            groups, config.N_FOLDS, config.SPLIT_STRATEGY, config.SEED, test_frac=0.10)

        if args.fold is None or args.fold == 1:
            split_meta = {
                "split_strategy": config.SPLIT_STRATEGY,
                "seed":           config.SEED,
                "train_folds":    [f.tolist() for f in train_idx_folds],
                "val_folds":      [f.tolist() for f in val_idx_folds],
                "test_idx":       test_idx.tolist(),
            }
            with open(split_path, "w") as f:
                json.dump(split_meta, f)
            logger.info(f"Split indices saved: {split_path}")

    # CV folds
    fold_range    = [args.fold - 1] if args.fold is not None else range(config.N_FOLDS)
    fold_metrics  = []

    for fold in fold_range:
        logger.info(f"=== Fold {fold + 1}/{config.N_FOLDS} ===")
        fold_dir = output_dir / f"fold_{fold + 1}"
        fold_dir.mkdir(exist_ok=True)

        datapoints_copy = copy.deepcopy(datapoints)
        train_dps, val_dps, _ = split_data_by_indices(
            datapoints_copy,
            train_indices=[train_idx_folds[fold]],
            val_indices=[val_idx_folds[fold]],
            test_indices=None,
        )
        train_data = data.MoleculeDataset(train_dps[0], featurizer=featurizer)
        val_data   = data.MoleculeDataset(val_dps[0],   featurizer=featurizer)
        logger.info(f"  Train: {len(train_data)} | Val: {len(val_data)}")

        model, scaler = build_model(train_data, val_data, encoder_ckpt)
        trainer = run_trainer(model, train_data, val_data, fold_dir,
                              log_name=f"{_log_name}_fold{fold+1}")

        val_loss = trainer.callback_metrics.get("val_loss", float("nan"))
        val_loss = float(val_loss.item() if hasattr(val_loss, "item") else val_loss)
        fold_metrics.append({"fold": fold + 1, "val_loss": val_loss})
        logger.info(f"  Fold {fold + 1} val_loss (z-score): {val_loss:.4f}")

        torch.save(model.state_dict(), fold_dir / "finetuned_model.pt")
        torch.save({"mean": scaler.mean_, "scale": scaler.scale_}, fold_dir / "scaler.pt")

    losses    = [m["val_loss"] for m in fold_metrics if not np.isnan(m["val_loss"])]
    mean_loss = float(np.mean(losses)) if losses else None
    std_loss  = float(np.std(losses))  if losses else None
    if losses:
        logger.info(f"CV val_loss: {mean_loss:.4f} ± {std_loss:.4f}")

    # Single-fold mode: save fold summary and exit
    if args.fold is not None:
        logger.info(f"Single-fold mode complete — fold {args.fold} done.")
        with open(output_dir / f"fold_{args.fold}_summary.json", "w") as f:
            json.dump({
                "run_date": _RUN_DATE, "job_id": _JOB_ID,
                "model_arch": "single_task", "pipeline_mode": config.PIPELINE_MODE,
                "subset_label": subset_label, "dataset_name": args.dataset_name,
                "fold": args.fold, "single_fold_mode": True,
                "fold_metrics": fold_metrics,
            }, f, indent=2)
        return

    # All-folds mode: save CV summary then train final model
    with open(output_dir / "cv_summary.json", "w") as f:
        json.dump({
            "run_date": _RUN_DATE, "job_id": _JOB_ID,
            "model_arch": "single_task", "pipeline_mode": config.PIPELINE_MODE,
            "subset_label": subset_label, "dataset_name": args.dataset_name,
            "cluster_col": CLUSTER_COL, "split_strategy": config.SPLIT_STRATEGY,
            "accelerator": config.ACCELERATOR, "freeze_encoder": config.FREEZE_ENCODER,
            "finetune_epochs": config.FINETUNE_EPOCHS, "finetune_lr": config.FINETUNE_LR,
            "fold_metrics": fold_metrics,
            "mean_val_loss": mean_loss, "std_val_loss": std_loss,
        }, f, indent=2)

    train_final_model(args, df, datapoints, groups, featurizer,
                      encoder_ckpt, subset_label, output_dir)


if __name__ == "__main__":
    main()
