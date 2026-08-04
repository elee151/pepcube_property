"""
predict_chemprop_multitask.py
Regenerates the 90% train+val split (seeded, same split functions and
config.SEED as finetune_chemprop_multitask.py's final-model block), trains a
multi-head Chemprop model on it, and predicts on an input CSV of peptide SMILES.

Example:
    python predict_chemprop_multitask.py \
        --run_id full_run \
        --pretrain_ckpt ~/pepcube_property/pretrained_chemprop_all_data/pretrained_model.pt \
        --model_dir pretrained_chemprop_90_trainval \
        --input_csv data/experimental_sequences.csv \
        --output_csv preds_90_trainval.csv
"""

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MODEL_ARCH", "multi_head")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
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

from finetune_chemprop_multitask import (
    build_multihead_model,
    load_and_merge_datasets,
    fit_scalers,
    scale_targets,
    MultiHeadDataset,
    build_loader,
    load_encoder,
    CHEMPROP_SUBSET_LABEL,
    CHEMELEON_SUBSET_LABEL,
    HEAD_KEYS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_FILENAME = "trainval_model.pt"
SCALER_FILENAME = "trainval_scalers.pt"
ARCH_FILENAME = "architecture.json"


def bmg_to_device(bmg, device):
    for attr in ("V", "E", "edge_index", "batch"):
        val = getattr(bmg, attr, None)
        if isinstance(val, torch.Tensor):
            setattr(bmg, attr, val.to(device))
    return bmg


def get_device():
    if config.ACCELERATOR == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if config.ACCELERATOR == "gpu" and not torch.cuda.is_available():
        logger.warning("ACCELERATOR=gpu but no CUDA device is available. Run on cpu")
    return torch.device("cpu")


def load_scalers(scaler_path):
    saved_scalers = torch.load(scaler_path, map_location="cpu", weights_only=False)
    scalers = {}
    for name, sc_data in saved_scalers.items():
        sc = StandardScaler()
        sc.mean_ = sc_data["mean"]
        sc.scale_ = sc_data["scale"]
        scalers[name] = sc
    return scalers


def load_model(model_dir, device):
    """Reload a previously-saved trainval artifact (used only by --skip_train)."""
    model_dir = Path(model_dir)
    model_path = model_dir / MODEL_FILENAME
    scaler_path = model_dir / SCALER_FILENAME
    arch_path = model_dir / ARCH_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(f"No model weights found at {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"No scalers found at {scaler_path}")

    if arch_path.exists():
        arch = json.loads(arch_path.read_text())
        depth, d_h = arch["mpnn_depth"], arch["mpnn_d_h"]
        logger.info(f"Loaded architecture.json: depth={depth}  d_h={d_h}")
    else:
        depth, d_h = config.MPNN_DEPTH, config.MPNN_D_H


    scalers = load_scalers(scaler_path)
    mp = cpnn.BondMessagePassing(depth=depth, d_h=d_h)
    model = build_multihead_model(mp, scalers, config.FREEZE_ENCODER)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.to(device)
    model.eval()
    logger.info(f"Loaded model weights from {model_path}")
    return model


def resolve_encoder(args):
    """For PIPELINE_MODE values other than 'chemprop' (fresh encoder) or
    'chemeleon', pulls a real pretrained checkpoint via
    config.pretrain_checkpoint(subset_name, pipeline_mode=config.PIPELINE_MODE)."""
    if args.pretrain_ckpt:
        ckpt = Path(args.pretrain_ckpt)
        return ckpt, args.subset_name or CHEMPROP_SUBSET_LABEL
    elif config.PIPELINE_MODE == "chemprop":
        return None, CHEMPROP_SUBSET_LABEL
    elif config.PIPELINE_MODE == "chemeleon":
        return config.CHEMELEON_PATH, CHEMELEON_SUBSET_LABEL
    else:
        if args.subset_name is None:
            raise ValueError("--subset_name required when PIPELINE_MODE=pretrained")
        ckpt = config.pretrain_checkpoint(args.subset_name,
                                          pipeline_mode=config.PIPELINE_MODE,
                                          model_arch="multi_head")
        return ckpt, args.subset_name


def resolve_trainval_dir(args, run_cfg, subset_label):
    if args.model_dir:
        return Path(args.model_dir)
    return config.finetune_dir(subset_label, run_cfg["run_name"]) / "trainval"


def _build_encoder(encoder_ckpt):
    if encoder_ckpt:
        mp = load_encoder(encoder_ckpt)
        d_h = mp.state_dict()["W_i.weight"].shape[0]
        depth = getattr(mp, "depth", config.MPNN_DEPTH)
        return mp, depth, d_h
    depth, d_h = config.MPNN_DEPTH, config.MPNN_D_H
    return cpnn.BondMessagePassing(depth=depth, d_h=d_h), depth, d_h


def train_trainval_model(args, device, out_dir):
    """Regenerate the 90% train+val split with split functions, config.SEED"""
    pl.seed_everything(config.SEED)

    run_cfg = config.MULTIHEAD_RUN_CONFIG[args.run_id]
    encoder_ckpt, subset_label = resolve_encoder(args)
    logger.info(f"Training trainval model — run_id={args.run_id}  subset={subset_label}")
    logger.info(f"Output dir: {out_dir}")

    merged_df, _, head_dfs = load_and_merge_datasets(run_cfg)
    smiles_arr = merged_df[config.SMILES_COL].values

    _, fold_train_smiles, fold_val_smiles, test_smiles = make_multihead_splits(
        head_dfs, config.SMILES_COL, config.CLUSTER_COL, config.N_FOLDS,
        config.SPLIT_STRATEGY, config.SEED, test_frac=0.10)
    train_folds, val_folds, test_idx = resolve_multihead_indices(
        smiles_arr, fold_train_smiles, fold_val_smiles, test_smiles, config.N_FOLDS)
    trainval_idx = np.sort(np.concatenate([train_folds[0], val_folds[0]]))
    logger.info(f"  Regenerated split — trainval: {len(trainval_idx)} | "
                f"test (held out, unused here): {len(test_idx)}")

    scalers = fit_scalers(merged_df, trainval_idx)
    train_targets, _ = scale_targets(merged_df, trainval_idx, trainval_idx, scalers)

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_ds = MultiHeadDataset(smiles_arr[trainval_idx], train_targets, featurizer)
    train_loader = build_loader(train_ds, featurizer, config.FINETUNE_BATCH, shuffle=True)

    mp, depth, d_h = _build_encoder(encoder_ckpt)
    model = build_multihead_model(mp, scalers, config.FREEZE_ENCODER)

    trainer = pl.Trainer(
        max_epochs=config.FINETUNE_EPOCHS,
        accelerator=config.ACCELERATOR,
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader)

    model.to(device)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / MODEL_FILENAME)
    torch.save(
        {name: {"mean": sc.mean_, "scale": sc.scale_} for name, sc in scalers.items()},
        out_dir / SCALER_FILENAME,
    )
    (out_dir / ARCH_FILENAME).write_text(json.dumps({"mpnn_depth": depth, "mpnn_d_h": d_h}, indent=2))
    logger.info(f"trainval model saved: {out_dir / MODEL_FILENAME}  (depth={depth}, d_h={d_h})")

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True,
                        choices=list(config.MULTIHEAD_RUN_CONFIG.keys()))
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Where the trainval artifact is cached/loaded (see --skip_train); "
                             "if omitted, defaults to config.finetune_dir(subset_label, run_name)/trainval.")
    parser.add_argument("--subset_name", type=str, default=None)
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--skip_train", action="store_true",
                        help="Reuse a cached model_dir artifact instead of retraining")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="CSV with a linear_SMILES column to predict on")
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    logger.info(f"Device: {device}")

    run_cfg = config.MULTIHEAD_RUN_CONFIG[args.run_id]
    _, subset_label = resolve_encoder(args)
    out_dir = resolve_trainval_dir(args, run_cfg, subset_label)

    cached = (
        args.skip_train
        and (out_dir / MODEL_FILENAME).exists()
        and (out_dir / SCALER_FILENAME).exists()
    )
    if cached:
        logger.info(f"--skip_train: loading cached trainval model from {out_dir}")
        model = load_model(out_dir, device)
    else:
        model = train_trainval_model(args, device, out_dir)

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    input_df = pd.read_csv(args.input_csv)
    smiles_col = config.SMILES_COL
    if smiles_col not in input_df.columns:
        raise ValueError(f"Input CSV must have a '{smiles_col}' column")

    pred_smiles = input_df[smiles_col].values

    valid_mask = [Chem.MolFromSmiles(s) is not None for s in pred_smiles]
    n_invalid = sum(not v for v in valid_mask)
    if n_invalid:
        logger.warning(f"{n_invalid} SMILES could not be parsed and will be skipped")

    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    valid_smiles = [pred_smiles[i] for i in valid_idx]

    PRED_CHUNK = 256
    all_preds = {name: [] for name in HEAD_KEYS}

    with torch.no_grad():
        for i in range(0, len(valid_smiles), PRED_CHUNK):
            chunk = valid_smiles[i : i + PRED_CHUNK]
            molgraphs = [featurizer(Chem.MolFromSmiles(s)) for s in chunk]
            bmg = BatchMolGraph(molgraphs)
            bmg = bmg_to_device(bmg, device)
            preds = model.predict(bmg)
            for name in HEAD_KEYS:
                all_preds[name].append(preds[name].squeeze(-1).cpu().numpy())
            if i % 10000 == 0:
                logger.info(f"  Predicted {i}/{len(valid_smiles)}")

    results = input_df.iloc[valid_idx].copy()
    for name in HEAD_KEYS:
        results[f"pred_{name}"] = np.concatenate(all_preds[name])

    results.to_csv(args.output_csv, index=False)
    logger.info(f"Predictions saved: {args.output_csv} ({len(results)} rows)")


if __name__ == "__main__":
    main()
