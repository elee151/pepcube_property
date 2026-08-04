"""
pretrain_chemberta.py
Pretrain ChemBERTa-77M-MTR on a peptide dataset.
Five regression heads (one per scored property), NaN-masked MSE and logs train loss per epoch
Saves a hugging-face checkpoint to be loaded with AutoModel.from_pretrained(path) in finetuning

Outputs:
    HF checkpoint (config.json + model weights)
    head_weights.pt
    scalers.pt (per head)
    pretrain_summary.json

Example usage:
    ACCELERATOR=gpu python pretrain_chemberta.py --subset_name all_generated_scored --cluster_col cluster_umap_hdb
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import sys as _sys
_p = Path(__file__).resolve().parent
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

CHEMBERTA_MODEL = os.environ.get("CHEMBERTA_MODEL", "DeepChem/ChemBERTa-77M-MTR")
ACCELERATOR     = os.environ.get("ACCELERATOR",     "cpu")
DEVICE          = "cuda" if ACCELERATOR == "gpu" and torch.cuda.is_available() else "cpu"

#parameters taken from HPO
PRETRAIN_EPOCHS  = int(os.environ.get("PRETRAIN_EPOCHS",  20))
PRETRAIN_BATCH   = int(os.environ.get("PRETRAIN_BATCH",   128))
PRETRAIN_LR      = float(os.environ.get("PRETRAIN_LR",    0.001))
WARMUP_FRAC      = float(os.environ.get("WARMUP_FRAC",    0.1))
WEIGHT_DECAY     = float(os.environ.get("WEIGHT_DECAY",   0.01))
MAX_GRAD_NORM    = float(os.environ.get("MAX_GRAD_NORM",   1.0))
MAX_LENGTH       = int(os.environ.get("MAX_LENGTH",        512))
FREEZE_LAYERS    = int(os.environ.get("FREEZE_LAYERS",     0))
DROPOUT          = float(os.environ.get("DROPOUT",         0.0))

PRETRAIN_HEAD_KEYS = config.PRETRAIN_LABEL_COLS


#  Output path helper
def chemberta_pretrain_dir(subset_name: str) -> Path:
    tag = f"chemberta_{config.SPLIT_STRATEGY}_raw"
    return config.BASE_DIR / "runs" / tag / "pretrain" / subset_name

def chemberta_pretrain_backbone_dir(subset_name: str) -> Path:
    return chemberta_pretrain_dir(subset_name) / "final" / "backbone"


#  Dataset
class PretrainDataset(Dataset):
    """SMILES + multi-label z-scored regression targets."""

    def __init__(self, smiles, targets_dict, tokenizer, max_length=512):
        self.smiles     = smiles
        self.targets    = targets_dict
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = torch.tensor(
            [float(self.targets[col][idx]) for col in PRETRAIN_HEAD_KEYS],
            dtype=torch.float32,
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "targets":        target,
        }


#  Model 

class ChemBERTaPretrainer(torch.nn.Module):
    """ChemBERTa backbone + one linear head per scored property. Uses mean of all non-padding token representations as default"""

    def __init__(self, model_name_or_path: str, n_tasks: int,
                 dropout: float = 0.0, freeze_layers: int = 0,
                 pooling: str = "mean"):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        self.pooling  = pooling
        hidden        = self.backbone.config.hidden_size

        self.heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, 1),
            )
            for _ in range(n_tasks)
        ])

        if freeze_layers > 0:
            layers = self.backbone.encoder.layer
            for layer in layers[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info(f"  Frozen {freeze_layers}/{len(layers)} transformer layers")

    def _pool(self, backbone_out, attention_mask):
        if self.pooling == "cls":
            return backbone_out.last_hidden_state[:, 0, :]
        else:  # mean
            token_emb = backbone_out.last_hidden_state           # (B, L, H)
            mask_exp  = attention_mask.unsqueeze(-1).float()     # (B, L, 1)
            summed    = (token_emb * mask_exp).sum(dim=1)        # (B, H)
            counts    = mask_exp.sum(dim=1).clamp(min=1e-9)      # (B, 1)
            return summed / counts

    def forward(self, input_ids, attention_mask):
        out  = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out, attention_mask)
        return torch.cat([head(pooled) for head in self.heads], dim=-1)  # (B, n_tasks)


#  Loss 
def masked_mse(preds, targets):
    """Mean MSE over tasks"""
    total    = torch.tensor(0.0, device=preds.device, requires_grad=True)
    n_active = 0
    for i in range(targets.shape[1]):
        mask = ~torch.isnan(targets[:, i])
        if mask.sum() == 0:
            continue
        loss  = torch.nn.functional.mse_loss(preds[mask, i], targets[mask, i])
        total = total + loss
        n_active += 1
    return total / max(n_active, 1)


#  Token-length filter
def filter_by_token_length(df: pd.DataFrame, tokenizer, max_tokens: int,
                           smiles_col: str) -> pd.DataFrame:
    """Drop rows whose SMILES tokenize to more than max_tokens"""
    logger.info(f"Filtering by token length (max={max_tokens}) ...")
    n_before = len(df)
    lengths  = df[smiles_col].apply(
        lambda s: tokenizer(s, truncation=False, return_tensors="pt")[
            "input_ids"
        ].shape[1]
    )
    mask       = lengths <= max_tokens
    n_dropped  = int((~mask).sum())
    df_out     = df[mask].reset_index(drop=True)
    logger.info(
        f"  Dropped {n_dropped}/{n_before} molecules "
        f"({n_dropped / n_before * 100:.1f}%) exceeding {max_tokens} tokens"
    )
    logger.info(f"  Retained: {len(df_out)} molecules for pretraining")
    return df_out


#  Scaler helpers
def fit_scalers(df: pd.DataFrame) -> dict:
    scalers = {}
    for col in PRETRAIN_HEAD_KEYS:
        vals = df[col].dropna().values.reshape(-1, 1)
        if len(vals) == 0:
            raise ValueError(f"Column '{col}' has no non-NaN values.")
        sc = StandardScaler().fit(vals)
        scalers[col] = sc
        logger.info(f"  Scaler [{col}]: mean={sc.mean_[0]:.4f}  scale={sc.scale_[0]:.4f}")
    return scalers


def scale_targets(df: pd.DataFrame, scalers: dict) -> dict:
    targets = {}
    for col in PRETRAIN_HEAD_KEYS:
        sc  = scalers[col]
        arr = df[col].values.astype(float)
        out = np.full(len(arr), np.nan)
        mask = ~np.isnan(arr)
        if mask.any():
            out[mask] = sc.transform(arr[mask].reshape(-1, 1)).flatten()
        targets[col] = out
    return targets


#  Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain ChemBERTa on full scored peptide dataset"
    )
    parser.add_argument("--subset_name",  type=str, required=True,
                        help="CSV filename stem in DATA_DIR (without .csv)")
    parser.add_argument("--cluster_col",  type=str, default="cluster_umap_hdb",
                        choices=["cluster_umap_hdb"])
    parser.add_argument("--pooling",      type=str, default="mean",
                        choices=["cls", "mean"],
                        help="Sequence pooling strategy (default: mean)")
    parser.add_argument("--model_name",   type=str, default=CHEMBERTA_MODEL,
                        help="HuggingFace model ID or local path")
    return parser.parse_args()


#  Main
def main():
    args = parse_args()
    global CHEMBERTA_MODEL
    CHEMBERTA_MODEL = args.model_name

    import random
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError:
        raise ImportError("Transformers required. Please ensure you're in the correct conda env and have it installed")

    output_dir   = chemberta_pretrain_dir(args.subset_name)
    final_dir    = output_dir / "final"
    backbone_dir = final_dir / "backbone"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    backbone_dir.mkdir(parents=True, exist_ok=True)

    data_path = config.DATA_DIR / f"{args.subset_name}.csv"

    logger.info("=== PRETRAIN ChemBERTa (full dataset, no val split) ===")
    logger.info(f"Date/Job:       {_RUN_DATE} / {_JOB_ID}")
    logger.info(f"Model:          {CHEMBERTA_MODEL}")
    logger.info(f"Subset:         {args.subset_name}")
    logger.info(f"Device:         {DEVICE}")
    logger.info(f"Pooling:        {args.pooling}")
    logger.info(f"Freeze layers:  {FREEZE_LAYERS}")
    logger.info(f"Epochs/Batch/LR: {PRETRAIN_EPOCHS} / {PRETRAIN_BATCH} / {PRETRAIN_LR}")
    logger.info(f"Weight decay:   {WEIGHT_DECAY}  |  Warmup frac: {WARMUP_FRAC}")
    logger.info(f"Max grad norm:  {MAX_GRAD_NORM}  |  Max length: {MAX_LENGTH}")
    logger.info(f"Heads:          {PRETRAIN_HEAD_KEYS}")
    logger.info(f"Output:         {output_dir}")

    #  Load data 
    df = pd.read_csv(data_path)
    required = [config.SMILES_COL, args.cluster_col] + PRETRAIN_HEAD_KEYS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[required].dropna(subset=[config.SMILES_COL, args.cluster_col]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows | {df[args.cluster_col].nunique()} unique groups")

    # Load tokenizer first, then filter by token length before fitting scalers
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
    df = filter_by_token_length(df, tokenizer, MAX_LENGTH, config.SMILES_COL)
    if len(df) == 0:
        raise ValueError(
            f"No molecules remain after token-length filtering at max_length={MAX_LENGTH}. "
            f"Try increasing MAX_LENGTH."
        )

    #  Scalers + targets 
    scalers = fit_scalers(df)
    targets = scale_targets(df, scalers)

    dataset   = PretrainDataset(
        df[config.SMILES_COL].values, targets, tokenizer, MAX_LENGTH
    )
    loader = DataLoader(
        dataset,
        batch_size=PRETRAIN_BATCH,
        shuffle=True,
        num_workers=min(config.NUM_WORKERS, 4),
        generator=torch.Generator().manual_seed(config.SEED),
    )

    #  Model 
    model = ChemBERTaPretrainer(
        CHEMBERTA_MODEL,
        n_tasks=len(PRETRAIN_HEAD_KEYS),
        dropout=DROPOUT,
        freeze_layers=FREEZE_LAYERS,
        pooling=args.pooling,
    ).to(DEVICE)

    n_params_total     = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Parameters: {n_params_total:,} total | {n_params_trainable:,} trainable"
    )

    optimizer    = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY,
    )
    total_steps  = len(loader) * PRETRAIN_EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    #  Training loop 
    epoch_losses = []

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        batch_losses = []

        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            targets_tensor = batch["targets"].to(DEVICE)

            preds = model(input_ids, attention_mask)
            loss  = masked_mse(preds, targets_tensor)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            batch_losses.append(loss.item())

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        epoch_losses.append(epoch_loss)
        logger.info(f"  Epoch {epoch+1:3d}/{PRETRAIN_EPOCHS}  train_loss={epoch_loss:.6f}")

    final_train_loss = epoch_losses[-1] if epoch_losses else float("nan")
    logger.info(f"Final train loss: {final_train_loss:.6f}")

    # Saves tokenizer backbone: config.json + pytorch_model.bin or safetensors
    model.backbone.save_pretrained(str(backbone_dir))
    tokenizer.save_pretrained(str(backbone_dir))
    logger.info(f"HF backbone checkpoint saved: {backbone_dir}")

    # Save regression head weights separately (for reference)
    head_state = {
        f"head_{i}_{col}": model.heads[i].state_dict()
        for i, col in enumerate(PRETRAIN_HEAD_KEYS)
    }
    torch.save(head_state, final_dir / "head_weights.pt")

    # Saves data scalers
    scaler_data = {
        col: {"mean": sc.mean_, "scale": sc.scale_}
        for col, sc in scalers.items()
    }
    torch.save(scaler_data, final_dir / "scalers.pt")
    logger.info(f"Head weights + scalers saved: {final_dir}")

    # Summary
    summary = {
        "run_date":        _RUN_DATE,
        "job_id":          _JOB_ID,
        "model":           CHEMBERTA_MODEL,
        "subset_name":     args.subset_name,
        "cluster_col":     args.cluster_col,
        "split_strategy":  config.SPLIT_STRATEGY,
        "accelerator":     DEVICE,
        "pooling":         args.pooling,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "pretrain_lr":     PRETRAIN_LR,
        "pretrain_batch":  PRETRAIN_BATCH,
        "warmup_frac":     WARMUP_FRAC,
        "weight_decay":    WEIGHT_DECAY,
        "max_grad_norm":   MAX_GRAD_NORM,
        "max_length":      MAX_LENGTH,
        "freeze_layers":   FREEZE_LAYERS,
        "dropout":         DROPOUT,
        "n_total":         len(df),
        "n_heads":         len(PRETRAIN_HEAD_KEYS),
        "heads":           PRETRAIN_HEAD_KEYS,
        "final_train_loss":final_train_loss,
        "epoch_losses":    epoch_losses,
        "backbone_dir":    str(backbone_dir),
    }
    with open(output_dir / "pretrain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
