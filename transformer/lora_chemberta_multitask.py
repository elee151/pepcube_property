"""
lora_chemberta_multitask.py
LoRA fine-tuning of ChemBERTa-77M-MTR — JOINT MULTI-HEAD model.

One shared LoRA-wrapped backbone is trained simultaneously on all three
experimental datasets (synthesizability, camsol, hemolysis) using masked
MSE loss over sparse NaN labels.

Evaluation:
  - 5-fold CV on the 90% training pool  →  per-head CV metrics
  - Single 90% model evaluated on 10% held-out test set  →  test metrics
  - That same 90% model is saved to disk as the production checkpoint
    (test set is never seen during training)

Final model is written to disk (model.pt + scalers.pt + lora_adapter/).

"""

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

import sys as _sys
from pathlib import Path as _Path
_p = _Path(__file__).resolve().parent
while _p.name != "pepcube_property" and _p != _p.parent:
    _p = _p.parent
if _p.name == "pepcube_property" and str(_p.parent) not in _sys.path:
    _sys.path.insert(0, str(_p.parent))

import pepcube_property.config as config
from pepcube_property.utils import *
from pepcube_property.results import append_results

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_RUN_DATE = datetime.now().strftime("%Y-%m-%d")
_JOB_ID   = os.environ.get("PBS_JOBID", "local").split(".")[0]

CHEMBERTA_MODEL = os.environ.get("CHEMBERTA_MODEL", "DeepChem/ChemBERTa-77M-MTR")
ACCELERATOR     = os.environ.get("ACCELERATOR",     "cpu")
DEVICE          = "cuda" if ACCELERATOR == "gpu" and torch.cuda.is_available() else "cpu"

FINETUNE_EPOCHS = int(os.environ.get("FINETUNE_EPOCHS", config.FINETUNE_EPOCHS))
FINETUNE_BATCH  = int(os.environ.get("FINETUNE_BATCH",  config.FINETUNE_BATCH))
FINETUNE_LR     = float(os.environ.get("FINETUNE_LR",   config.FINETUNE_LR))
WEIGHT_DECAY    = float(os.environ.get("WEIGHT_DECAY",  0.01))
WARMUP_FRAC     = float(os.environ.get("WARMUP_FRAC",   0.1))
MAX_GRAD_NORM   = float(os.environ.get("MAX_GRAD_NORM", 1.0))
DROPOUT         = float(os.environ.get("DROPOUT",       config.DROPOUT))
POOLING         = os.environ.get("POOLING",             "mean")
SCHEDULER_TYPE  = os.environ.get("SCHEDULER_TYPE",      "linear")
MAX_LENGTH      = int(os.environ.get("MAX_LENGTH",       512))

LORA_R       = int(os.environ.get("LORA_R",       8))
LORA_ALPHA   = int(os.environ.get("LORA_ALPHA",   32))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.1))
LORA_BIAS    = os.environ.get("LORA_BIAS",        "none")
LORA_LAYERS  = int(os.environ.get("LORA_LAYERS",  3))
LORA_TARGETS = os.environ.get("LORA_TARGETS",     "query,value")

HEAD_KEYS = ["synthesizability", "camsol", "hemolysis"]


def build_head_configs(run_id: str) -> dict:
    """Derive HEAD_CONFIGS from config.MULTIHEAD_RUN_CONFIG[run_id]"""
    run_cfg = config.MULTIHEAD_RUN_CONFIG[run_id]
    head_configs = {}
    for head_key in HEAD_KEYS:
        ds_name = run_cfg[head_key]
        ds_cfg  = config.DATASET_CONFIG[ds_name]
        head_configs[head_key] = {
            "data_file":    ds_cfg["data_file"],
            "label_col":    ds_cfg["label_col"],
            "dataset_name": ds_name,
        }
    return head_configs

HEAD_CONFIGS: dict = {}


#  Output path
def finetune_output_dir(run_name: str) -> Path:
    tag = f"chemberta_lora_{config.SPLIT_STRATEGY}_raw"
    return config.BASE_DIR / "runs" / tag / "finetune" / run_name


#  Shared utilities
def pool(backbone_out, attention_mask, pooling: str):
    if pooling == "cls":
        return backbone_out.last_hidden_state[:, 0, :]
    token_emb = backbone_out.last_hidden_state
    mask_exp  = attention_mask.unsqueeze(-1).float()
    return (token_emb * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)


def make_scheduler(optimizer, total_steps, warmup_steps):
    from transformers import (
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )
    if SCHEDULER_TYPE == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


def compute_metrics(y_true, y_pred) -> dict:
    mask   = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return {}
    pr, _ = pearsonr(y_true, y_pred)
    sp, _ = spearmanr(y_true, y_pred)
    return {
        "rmse":       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":        float(mean_absolute_error(y_true, y_pred)),
        "r2":         float(r2_score(y_true, y_pred)),
        "pearson_r":  float(pr),
        "spearman_r": float(sp),
        "n_samples":  int(mask.sum()),
    }


def aggregate_metrics(fold_results_list: list) -> dict:
    agg = {}
    for key in ["rmse", "mae", "r2", "pearson_r", "spearman_r"]:
        vals = [r[key] for r in fold_results_list if key in r]
        if vals:
            agg[f"mean_{key}"] = float(np.mean(vals))
            agg[f"std_{key}"]  = float(np.std(vals))
    return agg


def filter_by_token_length(df, tokenizer, smiles_col):
    n_before = len(df)
    lengths  = df[smiles_col].apply(
        lambda s: tokenizer(s, truncation=False, return_tensors="pt")["input_ids"].shape[1]
    )
    mask   = lengths <= MAX_LENGTH
    n_drop = int((~mask).sum())
    logger.info(f"  Token filter: dropped {n_drop}/{n_before} "
                f"({n_drop/n_before*100:.1f}%) exceeding {MAX_LENGTH} tokens")
    return df[mask].reset_index(drop=True), {
        "n_before": n_before, "n_dropped": n_drop,
        "pct_dropped": float(n_drop / n_before * 100),
    }


#  Data loading 

def load_and_merge(tokenizer) -> tuple[pd.DataFrame, dict, dict]:
    """Load each head's CSV, filter by token length, outer-merge on SMILES+cluster."""
    dfs, logs = {}, {}
    for name, cfg in HEAD_CONFIGS.items():
        df = pd.read_csv(cfg["data_file"])
        df = df[[config.SMILES_COL, config.CLUSTER_COL, cfg["label_col"]]].dropna(
            subset=[config.SMILES_COL, config.CLUSTER_COL]
        ).reset_index(drop=True)
        df, log = filter_by_token_length(df, tokenizer, config.SMILES_COL)
        df = df.rename(columns={cfg["label_col"]: f"label_{name}"})
        dfs[name]  = df
        logs[name] = log
        logger.info(f"  [{name}]: {len(df)} molecules after filtering")

    merged = dfs["synthesizability"]
    for name in ["camsol", "hemolysis"]:
        merged = pd.merge(merged, dfs[name],
                          on=[config.SMILES_COL, config.CLUSTER_COL], how="outer")
    merged = merged.dropna(subset=[config.SMILES_COL, config.CLUSTER_COL]).reset_index(drop=True)
    logger.info(f"Merged dataset: {len(merged)} total molecules")
    return merged, logs, dfs


#  Scaler helpers 

def fit_scalers(merged_df, train_idx) -> dict:
    scalers = {}
    for name in HEAD_KEYS:
        vals = merged_df[f"label_{name}"].iloc[train_idx].dropna().values.reshape(-1, 1)
        if len(vals) == 0:
            raise ValueError(f"No non-NaN training values for head '{name}'")
        scalers[name] = StandardScaler().fit(vals)
    return scalers


def scale_targets(merged_df, indices, scalers) -> dict:
    targets = {}
    for name in HEAD_KEYS:
        arr  = merged_df[f"label_{name}"].values[indices].astype(float)
        out  = np.full(len(arr), np.nan)
        mask = ~np.isnan(arr)
        if mask.any():
            out[mask] = scalers[name].transform(arr[mask].reshape(-1, 1)).flatten()
        targets[name] = out
    return targets


def save_scalers(scalers: dict, path: Path):
    torch.save(
        {name: {"mean": sc.mean_, "scale": sc.scale_} for name, sc in scalers.items()},
        path,
    )


#  Model ─

def build_lora_config(backbone):
    from peft import LoraConfig, TaskType
    n_layers    = len(backbone.encoder.layer)
    layer_idxs  = list(range(n_layers - LORA_LAYERS, n_layers))
    target_keys = [t.strip() for t in LORA_TARGETS.split(",")]
    target_modules = [
        f"encoder.layer.{i}.attention.self.{proj}"
        for i in layer_idxs for proj in target_keys
    ]
    logger.info(f"  LoRA targets: layers {layer_idxs}, projections {target_keys}")
    logger.info(f"  LoRA: r={LORA_R}  alpha={LORA_ALPHA}  dropout={LORA_DROPOUT}  bias={LORA_BIAS}")
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS, target_modules=target_modules, inference_mode=False,
    )


class LoRAMultiHeadModel(nn.Module):
    """Shared LoRA backbone + one regression head per task."""
    def __init__(self, encoder_path: str, dropout: float, pooling: str):
        super().__init__()
        from peft import get_peft_model
        backbone_raw  = AutoModel.from_pretrained(encoder_path)
        self.backbone = get_peft_model(backbone_raw, build_lora_config(backbone_raw))
        self.backbone.print_trainable_parameters()
        self.pooling  = pooling
        hidden        = self.backbone.config.hidden_size
        self.heads    = nn.ModuleDict({
            name: nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
            for name in HEAD_KEYS
        })

    def _encode(self, input_ids, attention_mask):
        seq_len      = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long,
                                    device=input_ids.device).unsqueeze(0)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids)
        return pool(out, attention_mask, self.pooling)

    def forward(self, input_ids, attention_mask):
        h = self._encode(input_ids, attention_mask)
        return {name: head(h).squeeze(-1) for name, head in self.heads.items()}


#  Dataset ─

class FinetuneDataset(Dataset):
    def __init__(self, smiles, targets_dict, tokenizer, max_length=512):
        self.smiles    = smiles
        self.targets   = targets_dict
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        for name in HEAD_KEYS:
            val = self.targets[name][idx]
            item[f"label_{name}"] = torch.tensor(
                float("nan") if np.isnan(val) else val, dtype=torch.float32)
        return item


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}


def build_loaders(merged_df, smiles, train_idx, val_idx, scalers, tokenizer):
    n_workers     = min(int(os.environ.get("NUM_WORKERS", config.NUM_WORKERS)), 4)
    train_targets = scale_targets(merged_df, train_idx, scalers)
    val_targets   = scale_targets(merged_df, val_idx,   scalers)
    train_ds = FinetuneDataset(smiles[train_idx], train_targets, tokenizer, MAX_LENGTH)
    val_ds   = FinetuneDataset(smiles[val_idx],   val_targets,   tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BATCH, shuffle=True,
                              num_workers=n_workers, collate_fn=collate_fn,
                              pin_memory=(DEVICE == "cuda"),
                              generator=torch.Generator().manual_seed(config.SEED))
    val_loader   = DataLoader(val_ds,   batch_size=FINETUNE_BATCH, shuffle=False,
                              num_workers=n_workers, collate_fn=collate_fn,
                              pin_memory=(DEVICE == "cuda"))
    return train_loader, val_loader


#  Training loop

def predict_original_scale(model, loader, scalers) -> dict:
    model.eval()
    all_preds = {name: [] for name in HEAD_KEYS}
    with torch.no_grad():
        for batch in loader:
            inp  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            outs = model(inp, mask)
            for name in HEAD_KEYS:
                z  = outs[name].cpu().numpy()
                sc = scalers[name]
                all_preds[name].append(z * sc.scale_[0] + sc.mean_[0])
    return {name: np.concatenate(v) for name, v in all_preds.items()}


def train_model(encoder_path, train_loader, val_loader, tmp_ckpt: Path) -> tuple[float, object]:
    """
    Train LoRAMultiHeadModel. Saves best weights to tmp_ckpt.
    Returns (best_val_loss, model_with_best_weights_loaded).
    """
    model = LoRAMultiHeadModel(encoder_path, dropout=DROPOUT, pooling=POOLING).to(DEVICE)
    optimizer    = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY,
    )
    total_steps  = len(train_loader) * FINETUNE_EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler    = make_scheduler(optimizer, total_steps, warmup_steps)

    best_val_loss = np.inf
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        for batch in train_loader:
            inp  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            outs = model(inp, mask)
            loss    = torch.tensor(0.0, device=DEVICE)
            n_heads = 0
            for name in HEAD_KEYS:
                lbl   = batch[f"label_{name}"].to(DEVICE)
                valid = ~torch.isnan(lbl)
                if valid.sum() == 0:
                    continue
                loss    = loss + nn.functional.mse_loss(outs[name][valid], lbl[valid])
                n_heads += 1
            if n_heads:
                loss = loss / n_heads
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step(); scheduler.step()

        if val_loader is None:
            logger.info(f"    epoch {epoch+1}/{FINETUNE_EPOCHS}: (no validation)")
            continue

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inp  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                outs = model(inp, mask)
                bl, n = torch.tensor(0.0, device=DEVICE), 0
                for name in HEAD_KEYS:
                    lbl   = batch[f"label_{name}"].to(DEVICE)
                    valid = ~torch.isnan(lbl)
                    if valid.sum() == 0:
                        continue
                    bl = bl + nn.functional.mse_loss(outs[name][valid], lbl[valid])
                    n += 1
                if n:
                    val_losses.append((bl / n).item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        logger.info(f"    epoch {epoch+1}/{FINETUNE_EPOCHS}: val_loss={val_loss:.4f}")
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), tmp_ckpt)

    if val_loader is None:
        return float("nan"), model

    if tmp_ckpt.exists():
        model.load_state_dict(
            torch.load(tmp_ckpt, map_location=DEVICE, weights_only=False), strict=False)
    return best_val_loss, model


def save_model(model, scalers, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    model.backbone.save_pretrained(str(out_dir / "lora_adapter"))
    save_scalers(scalers, out_dir / "scalers.pt")


#  Main finetune routine

#  Argument parsing + entrypoint

def parse_args():
    parser = argparse.ArgumentParser(
        description="ChemBERTa LoRA joint multi-head finetune"
    )
    parser.add_argument("--run_id",        type=str, default="run3",
                        help="Run config ID from config.MULTIHEAD_RUN_CONFIG "
                             "(e.g. run3, smoke, or any key you add there)")
    parser.add_argument("--pretrain_ckpt", type=str, required=True,
                        help="Path to HF backbone dir (from 01_pretrain or base model)")
    parser.add_argument("--lora_r",        type=int,   default=None)
    parser.add_argument("--lora_alpha",    type=int,   default=None)
    parser.add_argument("--lora_dropout",  type=float, default=None)
    parser.add_argument("--lora_bias",     type=str,   default=None,
                        choices=["none", "all", "lora_only"])
    parser.add_argument("--lora_layers",   type=int,   default=None)
    parser.add_argument("--lora_targets",  type=str,   default=None)
    return parser.parse_args()


def run_finetune(encoder_path: str, run_id: str, run_name: str, tokenizer) -> dict:
    out_dir = finetune_output_dir(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = out_dir / "_tmp_best.pt"

    merged_df, filter_logs, head_dfs = load_and_merge(tokenizer)
    groups = merged_df[config.CLUSTER_COL].values
    smiles = merged_df[config.SMILES_COL].values

    # Split each dataset independently (own cluster-id space each), then
    # union each fold's SMILES across datasets — same call a single-task
    # script makes on that same dataset, avoids a merged-pool GroupKFold
    # leaving a smaller dataset with zero of its own groups in some fold.
    per_dataset, fold_train_smiles, fold_val_smiles, test_smiles = make_multihead_splits(
        head_dfs, config.SMILES_COL, config.CLUSTER_COL, config.N_FOLDS,
        config.SPLIT_STRATEGY, config.SEED, test_frac=0.10,
    )
    train_folds, val_folds, test_idx = resolve_multihead_indices(
        smiles, fold_train_smiles, fold_val_smiles, test_smiles, config.N_FOLDS)
    for fold_i in range(config.N_FOLDS):
        for name, d in per_dataset.items():
            if len(d["val_folds"][fold_i]) == 0:
                logger.warning(f"  Fold {fold_i + 1}: '{name}' has 0 val samples!")

    split_meta = {
        "test_idx":     test_idx.tolist(),
        "train_folds":  [f.tolist() for f in train_folds],
        "val_folds":    [f.tolist() for f in val_folds],
        "n_total":      len(merged_df),
        "strategy":     config.SPLIT_STRATEGY,
        "seed":         config.SEED,
        "n_cv_folds":   config.N_FOLDS,
        "test_frac":    0.10,
        "encoder_path": encoder_path,
        "max_length":   MAX_LENGTH,
        "split_method": "per_dataset_independent_then_unioned",
        "lora_r":       LORA_R, "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT, "lora_bias": LORA_BIAS,
        "lora_layers":  LORA_LAYERS, "lora_targets": LORA_TARGETS,
        "dropout":      DROPOUT, "pooling": POOLING,
    }
    with open(out_dir / "split_indices.json", "w") as f:
        json.dump(split_meta, f, indent=2)
    logger.info(f"Split: test={len(test_idx)} | "
                f"fold1 train={len(train_folds[0])} val={len(val_folds[0])}")

    #  5-fold CV on the 90% pool ─
    fold_results = {name: [] for name in HEAD_KEYS}

    for fold in range(config.N_FOLDS):
        logger.info(f"\n=== CV Fold {fold+1}/{config.N_FOLDS} ===")
        train_idx = train_folds[fold]
        val_idx   = val_folds[fold]
        logger.info(f"  Train: {len(train_idx)}  Val: {len(val_idx)}")

        scalers = fit_scalers(merged_df, train_idx)
        train_loader, val_loader = build_loaders(
            merged_df, smiles, train_idx, val_idx, scalers, tokenizer)

        val_loss, model = train_model(encoder_path, train_loader, val_loader, tmp_ckpt)
        logger.info(f"  Fold {fold+1} best val_loss={val_loss:.4f}")

        val_preds = predict_original_scale(model, val_loader, scalers)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        for name in HEAD_KEYS:
            y_true = merged_df[f"label_{name}"].values[val_idx]
            m      = compute_metrics(y_true, val_preds[name])
            m["fold"] = fold + 1
            fold_results[name].append(m)
            logger.info(f"  [{name}] r2={m.get('r2', float('nan')):.3f}  "
                        f"pearson={m.get('pearson_r', float('nan')):.3f}  "
                        f"rmse={m.get('rmse', float('nan')):.4f}")

    if tmp_ckpt.exists():
        tmp_ckpt.unlink()

    cv_agg = {name: aggregate_metrics(fold_results[name]) for name in HEAD_KEYS}
    logger.info("\n=== CV AGGREGATED ===")
    for name, agg in cv_agg.items():
        logger.info(f"  [{name}] mean_r2={agg.get('mean_r2', float('nan')):.3f} "
                    f"(±{agg.get('std_r2', float('nan')):.3f})  "
                    f"mean_pearson={agg.get('mean_pearson_r', float('nan')):.3f}  "
                    f"mean_rmse={agg.get('mean_rmse', float('nan')):.4f}")

    #  Test metrics: train on full 90%, evaluate on held-out 10% 
    test_metrics = {}
    if len(test_idx) > 0:
        logger.info("\n=== Test set evaluation (90% model) ===")
        trainval_idx = np.sort(np.concatenate([train_folds[0], val_folds[0]]))
        scalers_90 = fit_scalers(merged_df, trainval_idx)
        train_targets_90 = scale_targets(merged_df, trainval_idx, scalers_90)
        n_workers    = min(int(os.environ.get("NUM_WORKERS", config.NUM_WORKERS)), 4)
        train_ds_90  = FinetuneDataset(smiles[trainval_idx], train_targets_90, tokenizer, MAX_LENGTH)
        train_loader_90 = DataLoader(train_ds_90, batch_size=FINETUNE_BATCH, shuffle=True,
                                     num_workers=n_workers, collate_fn=collate_fn,
                                     pin_memory=(DEVICE == "cuda"),
                                     generator=torch.Generator().manual_seed(config.SEED))
        tmp_90 = out_dir / "_tmp_90.pt"
        _, model_90 = train_model(encoder_path, train_loader_90, None, tmp_90)
        if tmp_90.exists():
            tmp_90.unlink()

        # Build test loader
        test_targets = scale_targets(merged_df, test_idx, scalers_90)
        n_workers    = min(int(os.environ.get("NUM_WORKERS", config.NUM_WORKERS)), 4)
        test_ds      = FinetuneDataset(smiles[test_idx], test_targets, tokenizer, MAX_LENGTH)
        test_loader  = DataLoader(test_ds, batch_size=FINETUNE_BATCH, shuffle=False,
                                  num_workers=n_workers, collate_fn=collate_fn,
                                  pin_memory=(DEVICE == "cuda"))
        test_preds = predict_original_scale(model_90, test_loader, scalers_90)

        for name in HEAD_KEYS:
            y_true = merged_df[f"label_{name}"].values[test_idx]
            m      = compute_metrics(y_true, test_preds[name])
            test_metrics[name] = m
            logger.info(f"  [{name}] TEST r2={m.get('r2', float('nan')):.3f}  "
                        f"pearson={m.get('pearson_r', float('nan')):.3f}  "
                        f"n={m.get('n_samples', 0)}")

    #  Save cv_summary.json 
    cv_summary = {
        "run_timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_date":              _RUN_DATE,
        "job_id":                _JOB_ID,
        "model_arch":            "chemberta_lora_multi_head",
        "encoder_path":          encoder_path,
        "run_id":                run_id,
        "run_name":              run_name,
        "split_strategy":        config.SPLIT_STRATEGY,
        "n_folds":               config.N_FOLDS,
        "n_test":                int(len(test_idx)),
        "pooling":               POOLING,
        "scheduler_type":        SCHEDULER_TYPE,
        "finetune_epochs":       FINETUNE_EPOCHS,
        "finetune_lr":           FINETUNE_LR,
        "finetune_batch":        FINETUNE_BATCH,
        "weight_decay":          WEIGHT_DECAY,
        "warmup_frac":           WARMUP_FRAC,
        "max_grad_norm":         MAX_GRAD_NORM,
        "lora_r":                LORA_R,
        "lora_alpha":            LORA_ALPHA,
        "lora_dropout":          LORA_DROPOUT,
        "lora_bias":             LORA_BIAS,
        "lora_layers":           LORA_LAYERS,
        "lora_targets":          LORA_TARGETS,
        "filter_logs":           filter_logs,
        "per_head_fold_results": fold_results,
        "cv_agg":                cv_agg,
        "test_metrics":          test_metrics,
    }
    with open(out_dir / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)
    logger.info(f"CV+test summary saved: {out_dir}/cv_summary.json")

    #  Appends to master CSV 
    rows = []
    ts   = cv_summary["run_timestamp"]
    for name in HEAD_KEYS:
        base = {
            "run_timestamp":  ts,
            "run_date":       _RUN_DATE,
            "model_arch":     "chemberta_lora_multi_head",
            "model_name":     "chemberta-lora-multihead",
            "run_id":         run_id,
            "run_name":       run_name,
            "head":           name,
            "dataset_name":   HEAD_CONFIGS[name]["data_file"].stem,
            "pipeline_mode":  "lora_finetune",
            "split_strategy": config.SPLIT_STRATEGY,
            "n_folds":        config.N_FOLDS,
            "lora_r":         LORA_R,
            "lora_alpha":     LORA_ALPHA,
            "lora_dropout":   LORA_DROPOUT,
            "lora_layers":    LORA_LAYERS,
            "lora_targets":   LORA_TARGETS,
        }
        rows.append({**base, "eval_split": "cv_val",  **cv_agg[name]})
        if name in test_metrics and test_metrics[name]:
            rows.append({**base, "eval_split": "test", "n_folds": 1,
                         **test_metrics[name]})

    #  Save full results JSON (same layout/location as finetune_transformer_multitask.py)
    json_path = out_dir / f"lora_chemberta_{run_id}_{config.SPLIT_STRATEGY}_results.json"
    with open(json_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    logger.info(f"Results saved: {json_path}")

    #  Adds resultss into transformer_results.csv
    append_results(rows, config.TRANSFORMER_RESULTS_CSV)
    logger.info(f"Results upserted to: {config.TRANSFORMER_RESULTS_CSV}")

    #  Save the 90% model as the production checkpoint (test set never seen in training)
    if len(test_idx) > 0:
        final_dir = out_dir / "final_model"
        save_model(model_90, scalers_90, final_dir)
        del model_90
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"Final model (90% train+val, test held out) saved: {final_dir}")
    else:
        logger.warning("No test set — skipping final model save (nothing trained on 90% pool).")

    return cv_summary

def main():
    global HEAD_CONFIGS
    args = parse_args()

    import random
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.run_id not in config.MULTIHEAD_RUN_CONFIG:
        raise ValueError(
            f"--run_id '{args.run_id}' not found in config.MULTIHEAD_RUN_CONFIG. "
            f"Available: {list(config.MULTIHEAD_RUN_CONFIG.keys())}"
        )
    HEAD_CONFIGS = build_head_configs(args.run_id)
    logger.info(f"HEAD_CONFIGS resolved for run_id={args.run_id}:")
    for k, v in HEAD_CONFIGS.items():
        logger.info(f"  [{k}] {v['dataset_name']} -> {v['data_file'].name}")

    global LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_BIAS, LORA_LAYERS, LORA_TARGETS
    if args.lora_r       is not None: LORA_R       = args.lora_r
    if args.lora_alpha   is not None: LORA_ALPHA   = args.lora_alpha
    if args.lora_dropout is not None: LORA_DROPOUT = args.lora_dropout
    if args.lora_bias    is not None: LORA_BIAS    = args.lora_bias
    if args.lora_layers  is not None: LORA_LAYERS  = args.lora_layers
    if args.lora_targets is not None: LORA_TARGETS = args.lora_targets

    encoder_path = args.pretrain_ckpt
    if not Path(encoder_path).exists():
        raise FileNotFoundError(f"--pretrain_ckpt not found: {encoder_path}")

    run_cfg  = config.MULTIHEAD_RUN_CONFIG[args.run_id]
    run_name = run_cfg["run_name"]

    logger.info("=" * 60)
    logger.info(f"ChemBERTa LoRA — JOINT MULTI-HEAD FINETUNE")
    logger.info(f"Date/Job:        {_RUN_DATE} / {_JOB_ID}")
    logger.info(f"Run ID:          {args.run_id}  ({run_name})")
    logger.info(f"Encoder:         {encoder_path}")
    logger.info(f"Device:          {DEVICE}")
    logger.info(f"Pooling:         {POOLING}  |  Scheduler: {SCHEDULER_TYPE}")
    logger.info(f"Epochs/Batch/LR: {FINETUNE_EPOCHS} / {FINETUNE_BATCH} / {FINETUNE_LR}")
    logger.info(f"Weight decay:    {WEIGHT_DECAY}  |  Warmup frac: {WARMUP_FRAC}")
    logger.info(f"LoRA:            r={LORA_R}  alpha={LORA_ALPHA}  dropout={LORA_DROPOUT}  "
                f"bias={LORA_BIAS}  layers={LORA_LAYERS}  targets={LORA_TARGETS}")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    run_finetune(encoder_path, args.run_id, run_name, tokenizer)
    logger.info("Done.")

if __name__ == "__main__":
    main()
