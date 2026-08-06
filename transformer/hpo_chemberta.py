"""
hpo_chemberta.py
Pretraining survey for ChemBERTa-77M-MTR on the scored peptide dataset.

Usage:
    ACCELERATOR=gpu python hpo_chemberta.py --subset_name all_generated_scored_hpo100k --cluster_col cluster_umap_hdb
"""

import argparse
import copy
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import sys as _sys
from pathlib import Path as _Path
_p = _Path(__file__).resolve().parent
while _p.name != "pepcube_property" and _p != _p.parent:
    _p = _p.parent
if _p.name == "pepcube_property" and str(_p.parent) not in _sys.path:
    _sys.path.insert(0, str(_p.parent))

import pepcube_property.config as config
from pepcube_property.utils import make_train_val_test_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_RUN_DATE = datetime.now().strftime("%Y-%m-%d")
_JOB_ID   = os.environ.get("PBS_JOBID", "local").split(".")[0]

HPO_MAX_EPOCHS = int(os.environ.get("HPO_MAX_EPOCHS", 20))
HPO_SEED       = int(os.environ.get("HPO_SEED",       42))
ACCELERATOR    = os.environ.get("ACCELERATOR",         "cpu")
DEVICE         = "cuda" if ACCELERATOR == "gpu" and torch.cuda.is_available() else "cpu"

OUTPUT_DIR = config.BASE_DIR / "hpo_chemberta_results"
OUTPUT_DIR.mkdir(exist_ok=True)

CHEMBERTA_MODEL  = os.environ.get("CHEMBERTA_MODEL", "DeepChem/ChemBERTa-77M-MTR")
PRETRAIN_HEAD_KEYS = config.PRETRAIN_LABEL_COLS

SURVEY_GRID = [
    {"lr": 2e-5, "batch_size": 16, "epochs": 20, "dropout": 0.1,
     "pooling": "mean", "scheduler_type": "linear",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.1,
     "_label": "baseline_defaults"},

    {"lr": 1e-4, "batch_size": 64, "epochs": 10, "dropout": 0.0,
     "pooling": "mean", "scheduler_type": "linear",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.1,
     "_label": "lr_1e4_batch64_short"},

    {"lr": 5e-4, "batch_size": 128, "epochs": 20, "dropout": 0.0,
     "pooling": "mean", "scheduler_type": "linear",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.1,
     "_label": "lr_5e4_batch128"},

    {"lr": 2e-4, "batch_size": 64, "epochs": 30, "dropout": 0.0,
     "pooling": "mean", "scheduler_type": "linear",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.1,
     "_label": "lr_2e4_batch64_long"},

    {"lr": 1e-4, "batch_size": 32, "epochs": 20, "dropout": 0.0,
     "pooling": "mean", "scheduler_type": "linear",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.1,
     "_label": "lr_1e4_batch32"},

    {"lr": 1e-4, "batch_size": 128, "epochs": 15, "dropout": 0.0,
     "pooling": "mean", "scheduler_type": "linear",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.1,
     "_label": "lr_1e4_batch128_short"},

    {"lr": 1e-4, "batch_size": 64, "epochs": 20, "dropout": 0.0,
     "pooling": "mean", "scheduler_type": "linear",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 3, "warmup_frac": 0.1,
     "_label": "freeze_3layers"},

    {"lr": 5e-4, "batch_size": 64, "epochs": 30, "dropout": 0.0,
     "pooling": "mean", "scheduler_type": "linear",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 3, "warmup_frac": 0.1,
     "_label": "freeze_3layers_lr_5e4_long"},

    {"lr": 1e-4, "batch_size": 64, "epochs": 20, "dropout": 0.1,
     "pooling": "mean", "scheduler_type": "linear",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.1,
     "_label": "dropout_01"},

    {"lr": 1e-4, "batch_size": 64, "epochs": 10, "dropout": 0.0,
     "pooling": "mean", "scheduler_type": "linear",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.1,
     "_label": "dropout_0_short"},

    {"lr": 1e-4, "batch_size": 64, "epochs": 30, "dropout": 0.0,
     "pooling": "mean", "scheduler_type": "cosine",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.1,
     "_label": "cosine_long"},

    {"lr": 1e-4, "batch_size": 64, "epochs": 20, "dropout": 0.0,
     "pooling": "mean", "scheduler_type": "linear",
     "weight_decay": 0.001, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.1,
     "_label": "wd_low"},

    {"lr": 1e-4, "batch_size": 64, "epochs": 20, "dropout": 0.0,
     "pooling": "cls", "scheduler_type": "linear",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.1,
     "_label": "pooling_cls"},

    {"lr": 1e-4, "batch_size": 64, "epochs": 20, "dropout": 0.0,
     "pooling": "mean", "scheduler_type": "linear",
     "weight_decay": 0.01, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.2,
     "_label": "warmup_long"},

    
    {"lr": 2e-4, "batch_size": 64, "epochs": 40, "dropout": 0.0,
     "pooling": "mean", "scheduler_type": "cosine",
     "weight_decay": 0.001, "max_grad_norm": 1.0,
     "freeze_layers": 0, "warmup_frac": 0.1,
     "_label": "best_combo_candidate"},
]

# Deduplicate on all keys except _label
_seen = []
SURVEY_TRIALS = []
for cfg in SURVEY_GRID:
    key = {k: v for k, v in cfg.items() if k != "_label"}
    if key not in _seen:
        _seen.append(key)
        SURVEY_TRIALS.append(cfg)

logger.info(f"Survey: {len(SURVEY_TRIALS)} unique trials")


#  Dataset

class PretrainDataset(Dataset):
    def __init__(self, smiles, targets_dict, tokenizer, max_length=512):
        self.smiles    = smiles
        self.targets   = targets_dict
        self.tokenizer = tokenizer
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
    def __init__(self, model_name_or_path, n_tasks, dropout, freeze_layers, pooling):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        self.pooling  = pooling
        hidden        = self.backbone.config.hidden_size
        self.heads    = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))
            for _ in range(n_tasks)
        ])
        if freeze_layers > 0:
            layers = self.backbone.encoder.layer
            for layer in layers[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

    def _pool(self, out, attention_mask):
        if self.pooling == "cls":
            return out.last_hidden_state[:, 0, :]
        mask_exp = attention_mask.unsqueeze(-1).float()
        summed   = (out.last_hidden_state * mask_exp).sum(dim=1)
        counts   = mask_exp.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out, attention_mask)
        return torch.cat([h(pooled) for h in self.heads], dim=-1)


def masked_mse(preds, targets):
    total = torch.tensor(0.0, device=preds.device, requires_grad=True)
    n     = 0
    for i in range(targets.shape[1]):
        mask = ~torch.isnan(targets[:, i])
        if mask.sum() == 0:
            continue
        total = total + torch.nn.functional.mse_loss(preds[mask, i], targets[mask, i])
        n += 1
    return total / max(n, 1)


#  Token-length filter
def filter_by_token_length(df: pd.DataFrame, tokenizer, max_tokens: int,
                           smiles_col: str) -> pd.DataFrame:
    """Drop rows whose SMILES tokenise to more than max_tokens."""
    logger.info(f"Filtering by token length (max={max_tokens}) ...")
    n_before = len(df)
    lengths  = df[smiles_col].apply(
        lambda s: tokenizer(s, truncation=False, return_tensors="pt")[
            "input_ids"
        ].shape[1]
    )
    mask      = lengths <= max_tokens
    n_dropped = int((~mask).sum())
    df_out    = df[mask].reset_index(drop=True)
    logger.info(
        f"  Dropped {n_dropped}/{n_before} molecules "
        f"({n_dropped / n_before * 100:.1f}%) exceeding {max_tokens} tokens"
    )
    logger.info(f"  Retained: {len(df_out)} molecules for HPO")
    return df_out


#  Scaler helpers
def fit_scalers(df, train_idx):
    scalers = {}
    for col in PRETRAIN_HEAD_KEYS:
        vals = df[col].iloc[train_idx].dropna().values.reshape(-1, 1)
        if len(vals) == 0:
            raise ValueError(f"Column '{col}' has no non-NaN training values.")
        scalers[col] = StandardScaler().fit(vals)
    return scalers

def scale_targets(df, idx, scalers):
    out = {}
    for col in PRETRAIN_HEAD_KEYS:
        sc  = scalers[col]
        arr = df[col].iloc[idx].values.astype(float)
        res = np.full(len(arr), np.nan)
        mask = ~np.isnan(arr)
        if mask.any():
            res[mask] = sc.transform(arr[mask].reshape(-1, 1)).flatten()
        out[col] = res
    return out


#  Trial runner
def run_trial(cfg, train_smiles, val_smiles, train_targets, val_targets,
              tokenizer, encoder_path, max_length=512):
    """Train with one config; returns best val_loss."""
    from transformers import (
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
    )

    train_ds = PretrainDataset(train_smiles, train_targets, tokenizer, max_length)
    val_ds   = PretrainDataset(val_smiles,   val_targets,   tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=min(4, os.cpu_count() or 1),
                              generator=torch.Generator().manual_seed(HPO_SEED))
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=min(4, os.cpu_count() or 1))

    model = ChemBERTaPretrainer(
        encoder_path,
        n_tasks=len(PRETRAIN_HEAD_KEYS),
        dropout=cfg["dropout"],
        freeze_layers=cfg["freeze_layers"],
        pooling=cfg["pooling"],
    ).to(DEVICE)

    optimizer    = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    total_steps  = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_frac"])

    if cfg["scheduler_type"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

    best_val = np.inf
    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            inp  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            tgt  = batch["targets"].to(DEVICE)
            loss = masked_mse(model(inp, mask), tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inp  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                tgt  = batch["targets"].to(DEVICE)
                val_losses.append(masked_mse(model(inp, mask), tgt).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        if not np.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss
        logger.info(f"      epoch {epoch+1}/{cfg['epochs']}: val_loss={val_loss:.4f}")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return best_val


#  Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="ChemBERTa pretraining survey")
    parser.add_argument("--subset_name",  type=str, required=True)
    parser.add_argument("--cluster_col",  type=str, default="cluster_umap_hdb")
    parser.add_argument("--max_length",   type=int, default=512)
    parser.add_argument("--model_name",   type=str, default=CHEMBERTA_MODEL,
                        help="HF model ID or path to a local HF checkpoint dir")
    return parser.parse_args()


#  Main
def main():
    args = parse_args()
    global CHEMBERTA_MODEL
    CHEMBERTA_MODEL = args.model_name

    import random
    random.seed(HPO_SEED)
    np.random.seed(HPO_SEED)
    torch.manual_seed(HPO_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(HPO_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("transformers required. Install with: pip install transformers")

    data_path = config.DATA_DIR / f"{args.subset_name}.csv"
    logger.info(f"=== ChemBERTa Pretraining Survey ===")
    logger.info(f"Date/Job:     {_RUN_DATE} / {_JOB_ID}")
    logger.info(f"Model:        {CHEMBERTA_MODEL}")
    logger.info(f"Subset:       {args.subset_name}")
    logger.info(f"Device:       {DEVICE}")
    logger.info(f"Trials:       {len(SURVEY_TRIALS)}")
    logger.info(f"Heads:        {PRETRAIN_HEAD_KEYS}")

    df = pd.read_csv(data_path)
    required = [config.SMILES_COL, args.cluster_col] + PRETRAIN_HEAD_KEYS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[required].dropna(
        subset=[config.SMILES_COL, args.cluster_col]
    ).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows")

    # Load tokenizer first so we can filter by token length before splitting.
    # Filtering must happen before make_train_val_test_indices so indices are consistent.
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
    df = filter_by_token_length(df, tokenizer, args.max_length, config.SMILES_COL)
    if len(df) == 0:
        raise ValueError(
            f"No molecules remain after token-length filtering at max_length={args.max_length}."
        )

    groups = df[args.cluster_col].values
    train_folds, val_folds, test_idx = make_train_val_test_indices(
        groups, config.N_FOLDS, config.SPLIT_STRATEGY, config.SEED,
        test_frac=0.10,
    )
    train_idx = train_folds[0]
    val_idx   = val_folds[0]
    logger.info(f"train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    scalers       = fit_scalers(df, train_idx)
    train_targets = scale_targets(df, train_idx, scalers)
    val_targets   = scale_targets(df, val_idx,   scalers)
    train_smiles  = df[config.SMILES_COL].iloc[train_idx].values
    val_smiles    = df[config.SMILES_COL].iloc[val_idx].values

    results  = []
    best_val = np.inf
    best_cfg = None

    for i, trial_cfg in enumerate(SURVEY_TRIALS):
        label = trial_cfg.get("_label", "")
        cfg   = {k: v for k, v in trial_cfg.items() if k != "_label"}
        # Cap epochs at HPO_MAX_EPOCHS for the survey
        cfg["epochs"] = min(cfg["epochs"], HPO_MAX_EPOCHS)

        logger.info(f"\n  Trial {i+1}/{len(SURVEY_TRIALS)}"
                    f"{'  [' + label + ']' if label else ''}: {cfg}")
        t0 = time.time()
        try:
            val_loss = run_trial(
                cfg, train_smiles, val_smiles,
                train_targets, val_targets,
                tokenizer, CHEMBERTA_MODEL,
                max_length=args.max_length,
            )
        except Exception as e:
            logger.warning(f"  Trial failed: {e}")
            val_loss = np.inf

        elapsed = time.time() - t0
        logger.info(f"  => val_loss={val_loss:.4f}  ({elapsed:.1f}s)")

        results.append({
            **cfg,
            "_label":    label,
            "val_loss":  val_loss,
            "elapsed_s": elapsed,
        })

        if val_loss < best_val:
            best_val = val_loss
            best_cfg = copy.deepcopy(cfg)
            logger.info(f"  *** New best: {best_val:.4f}")

    results.sort(key=lambda x: x["val_loss"])

    out = {
        "run_date":      _RUN_DATE,
        "job_id":        _JOB_ID,
        "model":         CHEMBERTA_MODEL,
        "subset_name":   args.subset_name,
        "split_seed":    config.SEED,
        "n_train":       len(train_idx),
        "n_val":         len(val_idx),
        "n_test":        len(test_idx),
        "heads":         PRETRAIN_HEAD_KEYS,
        "hpo_max_epochs":HPO_MAX_EPOCHS,
        "n_trials":      len(SURVEY_TRIALS),
        "best_config":   best_cfg,
        "best_val_loss": best_val,
        "all_results":   results,
    }
    out_path = OUTPUT_DIR / "chemberta_pretrain_survey_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    best_path = OUTPUT_DIR / "best_chemberta_pretrain_config.json"
    with open(best_path, "w") as f:
        json.dump(best_cfg, f, indent=2)

    logger.info(f"Best config: {best_cfg}")
    logger.info(f"Best val_loss: {best_val:.4f}")
    logger.info(f"Results saved: {out_path}")
    logger.info(f"Best config saved: {best_path}")

if __name__ == "__main__":
    main()
