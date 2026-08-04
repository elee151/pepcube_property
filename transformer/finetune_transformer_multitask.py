"""
finetune_chemberta_multitask.py
Joint multi-head finetuning of transformer models (ChemBERTa, PepDoRA)

Example Usage:
    python finetune_transformer_multitask.py --model_name pepdora --run_id run3
"""

import argparse
import json
import logging
import os
import sys
import copy
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MODEL_ARCH",                   "multi_head")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",  "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModel

import sys as _sys
_p = Path(__file__).resolve().parent
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

#  Local HF cache paths — layout populated by setup_hf_cache.py 
_HF_HUB_ROOT = config.HF_HUB_CACHE

CHEMBERTA_MTR_DIR = _HF_HUB_ROOT / (
    "models--DeepChem--ChemBERTa-77M-MTR/snapshots/"
    "fc007d31c2fb774ab7a8e5a8d318e25cb01d2da1"
)  # has config.json + model.safetensors + tokenizer files

CHEMBERTA_MLM_WEIGHTS_DIR = _HF_HUB_ROOT / (
    "models--DeepChem--ChemBERTa-77M-MLM/snapshots/"
    "d62f784b9a0a3aab09c788a7fb95a8e1ce89116f"
)  # has config.json + model.safetensors ONLY — no tokenizer files

CHEMBERTA_MLM_TOKENIZER_DIR = _HF_HUB_ROOT / (
    "models--DeepChem--ChemBERTa-77M-MLM/snapshots/"
    "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"
)  # has tokenizer files (vocab/merges/tokenizer.json) + its own config.json

PEPDORA_ADAPTER_DIR = _HF_HUB_ROOT / (
    "models--ChatterjeeLab--PepDoRA/snapshots/"
    "e034544e8f2ab1c34fffcfd4984f4183db7f12ed"
)  # has adapter_config.json + adapter_model.safetensors (LoRA/DoRA adapter)


def _check_local_dir(path: Path, required_files: list[str], label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{label}: directory not found: {path}\n"
            f"Check HF_HUB_CACHE / hf_cache layout"
        )
    missing = [f for f in required_files if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{label}: missing {missing} in {path}"
        )


#  Model registry 
MODEL_REGISTRY = {

    "chemberta-77m-mtr": {
        "hf_name":           str(CHEMBERTA_MTR_DIR),
        "tokenizer_path":    str(CHEMBERTA_MTR_DIR),
        "trust_remote_code": False,
        "is_pepdora":        False,
        "max_tokens":        512,
    },
    "pepdora": {
        "hf_name":           str(PEPDORA_ADAPTER_DIR),
        "base_model":        str(CHEMBERTA_MLM_WEIGHTS_DIR),
        "tokenizer_path":    str(CHEMBERTA_MLM_TOKENIZER_DIR),
        "trust_remote_code": False,
        "is_pepdora":        True,
        "max_tokens":        512,
    },
}

#  Head definitions from config.MULTIHEAD_RUN_CONFIG
def build_head_configs(run_id: str) -> dict:
    """Derive HEAD_CONFIGS from config.MULTIHEAD_RUN_CONFIG[run_id]."""
    run_cfg = config.MULTIHEAD_RUN_CONFIG[run_id]
    head_configs = {}
    for head_key in ["synthesizability", "camsol", "hemolysis"]:
        ds_name = run_cfg[head_key]
        ds_cfg  = config.DATASET_CONFIG[ds_name]
        head_configs[head_key] = {
            "data_file": ds_cfg["data_file"],
            "label_col": ds_cfg["label_col"],
            "dataset_name": ds_name,
        }
    return head_configs

HEAD_CONFIGS: dict = {}
HEAD_KEYS = ["synthesizability", "camsol", "hemolysis"]


#  Token length filtering
def filter_by_token_length(df, tokenizer, max_tokens, smiles_col):
    """Drop rows whose SMILES tokenize to more than max_tokens"""
    logger.info(f"Filtering by token length (max={max_tokens}) ...")
    lengths = df[smiles_col].apply(
        lambda s: tokenizer(
            s, truncation=False, return_tensors="pt"
        )["input_ids"].shape[1]
    )
    mask        = lengths <= max_tokens
    n_before    = len(df)
    n_dropped   = (~mask).sum()
    df_filtered = df[mask].reset_index(drop=True)
    logger.info(
        f"  Dropped {n_dropped}/{n_before} molecules "
        f"({n_dropped/n_before*100:.1f}%) exceeding {max_tokens} tokens"
    )
    return df_filtered, {
        "n_before": n_before,
        "n_dropped": int(n_dropped),
        "max_tokens": max_tokens,
        "pct_dropped": float(n_dropped / n_before * 100),
    }


#  Data loading and merging
def load_and_merge(tokenizer, max_tokens):
    """
    Load all three head datasets, filter by token length, outer-merge on SMILES.
    Returns merged_df with columns: linear_SMILES, cluster_umap_hdb,
    label_synthesizability, label_camsol, label_hemolysis.
    """
    dfs   = {}
    logs  = {}
    for name, cfg in HEAD_CONFIGS.items():
        df = pd.read_csv(cfg["data_file"])
        df = df[[config.SMILES_COL, config.CLUSTER_COL, cfg["label_col"]]].dropna(
            subset=[config.SMILES_COL, config.CLUSTER_COL]
        ).reset_index(drop=True)
        df, log = filter_by_token_length(df, tokenizer, max_tokens, config.SMILES_COL)
        df = df.rename(columns={cfg["label_col"]: f"label_{name}"})
        dfs[name]  = df
        logs[name] = log
        logger.info(f"  [{name}]: {len(df)} molecules after filtering")

    merged = dfs["synthesizability"]
    for name in ["camsol", "hemolysis"]:
        merged = pd.merge(
            merged, dfs[name],
            on=[config.SMILES_COL, config.CLUSTER_COL],
            how="outer"
        )

    merged = merged.dropna(subset=[config.SMILES_COL, config.CLUSTER_COL])
    merged = merged.reset_index(drop=True)
    logger.info(f"Merged dataset: {len(merged)} total molecules")
    return merged, logs, dfs


#  Dataset
class TransformerPeptideDataset(Dataset):
    def __init__(self, smiles, targets_dict, tokenizer, max_tokens):
        self.smiles      = smiles
        self.targets     = targets_dict   # {head_name: np.ndarray}
        self.tokenizer   = tokenizer
        self.max_tokens  = max_tokens

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            max_length=self.max_tokens,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        for name in HEAD_KEYS:
            val = self.targets[name][idx]
            item[f"label_{name}"] = torch.tensor(
                float("nan") if np.isnan(val) else val, dtype=torch.float32
            )
        return item


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


class TransformerMultiHead(pl.LightningModule):
    """
    Shared transformer encoder + one linear regression head per property.
    Uses token (index 0) as the molecular embedding.
    NaN targets are masked from each head's loss.
    """
    def __init__(self, encoder, hidden_size, scalers,
                 lr=1e-4, warmup_steps=100,
                 freeze_encoder_epochs=0, dropout=0.1):
        super().__init__()
        self.encoder               = encoder
        self.scalers               = scalers
        self.lr                    = lr
        self.warmup_steps          = warmup_steps
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.dropout_layer         = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, 1) for name in HEAD_KEYS
        })
        self.mse = nn.MSELoss(reduction="none")

        if freeze_encoder_epochs > 0:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _encode(self, batch):
        enc_inputs = {
            "input_ids":      batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if "token_type_ids" in batch:
            enc_inputs["token_type_ids"] = batch["token_type_ids"]
        # Provide position_ids explicitly
        seq_len = enc_inputs["input_ids"].shape[1]
        enc_inputs["position_ids"] = torch.arange(
            seq_len, dtype=torch.long, device=enc_inputs["input_ids"].device
        ).unsqueeze(0)
        out = self.encoder(**enc_inputs)
        # [CLS] token embedding
        cls = out.last_hidden_state[:, 0, :]
        return self.dropout_layer(cls)

    def forward(self, batch):
        h = self._encode(batch)
        return {name: head(h).squeeze(-1) for name, head in self.heads.items()}

    def _masked_mse(self, pred, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return self.mse(pred[mask], target[mask]).mean()

    def _shared_step(self, batch, stage):
        preds = self(batch)
        total = torch.tensor(0.0, device=self.device)
        logs  = {}
        for name in HEAD_KEYS:
            l = self._masked_mse(preds[name], batch[f"label_{name}"])
            logs[f"{stage}_{name}_loss"] = l
            total = total + l
        avg = total / len(HEAD_KEYS)
        logs[f"{stage}_loss"] = avg
        self.log_dict(logs, prog_bar=True, on_epoch=True, on_step=False,
                      batch_size=batch["input_ids"].shape[0])
        return avg

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

    def on_train_epoch_start(self):
        if (self.freeze_encoder_epochs > 0 and
                self.current_epoch == self.freeze_encoder_epochs):
            for p in self.encoder.parameters():
                p.requires_grad = True
            logger.info(f"Encoder unfrozen at epoch {self.current_epoch}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr, weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def predict_original_scale(self, batch):
        """Return predictions in original (unscaled) label space."""
        enc_batch = {k: v for k, v in batch.items() if not k.startswith("label_")}
        with torch.no_grad():
            z_preds = self(enc_batch)
        out = {}
        for name in HEAD_KEYS:
            z   = z_preds[name].cpu().numpy()
            sc  = self.scalers[name]
            out[name] = z * sc.scale_[0] + sc.mean_[0]
        return out


#  Scaler helpers
def fit_scalers(merged_df, train_idx):
    scalers = {}
    for name in HEAD_KEYS:
        col  = f"label_{name}"
        vals = merged_df[col].iloc[train_idx].dropna().values.reshape(-1, 1)
        if len(vals) == 0:
            raise ValueError(f"No non-NaN values for head '{name}' in training set")
        sc = StandardScaler().fit(vals)
        scalers[name] = sc
    return scalers


def scale_targets(merged_df, indices, scalers):
    targets = {}
    for name in HEAD_KEYS:
        col  = f"label_{name}"
        arr  = merged_df[col].values[indices]
        out  = np.full(len(arr), np.nan)
        mask = ~np.isnan(arr)
        if mask.any():
            out[mask] = scalers[name].transform(
                arr[mask].reshape(-1, 1)
            ).flatten()
        targets[name] = out
    return targets


#  Metrics
def compute_metrics(y_true, y_pred):
    mask   = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return {}
    r_p, _ = pearsonr(y_true, y_pred)
    r_s, _ = spearmanr(y_true, y_pred)
    return {
        "rmse":      float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":       float(mean_absolute_error(y_true, y_pred)),
        "r2":        float(r2_score(y_true, y_pred)),
        "pearson_r": float(r_p),
        "spearman_r":float(r_s),
        "n":         int(len(y_true)),
    }


#  Inference
def run_inference(model, loader, device):
    model.eval()
    model.to(device)
    all_preds = {name: [] for name in HEAD_KEYS}
    with torch.no_grad():
        for batch in loader:
            # move encoder inputs to device; drop label_* keys
            enc_batch = {
                k: v.to(device) for k, v in batch.items()
                if not k.startswith("label_")
            }
            preds = model.predict_original_scale(enc_batch)
            for name in HEAD_KEYS:
                all_preds[name].append(preds[name])
    return {name: np.concatenate(v) for name, v in all_preds.items()}


#  Main 

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer multi-head finetuning"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Which transformer to finetune")
    parser.add_argument("--run_id", type=str, default="run3",
                        help="Run config ID (default: run3)")
    parser.add_argument("--lr",                   type=float, default=2e-5)
    parser.add_argument("--batch_size",           type=int,   default=16)
    parser.add_argument("--epochs",               type=int,   default=20)
    parser.add_argument("--dropout",              type=float, default=0.1)
    parser.add_argument("--freeze_encoder_epochs",type=int,   default=0)
    parser.add_argument("--warmup_steps",         type=int,   default=100)
    parser.add_argument("--pretrain_ckpt",        type=str,   default=None,
                        help="Path to pretrained backbone dir (overrides HF load)")
    return parser.parse_args()


def main():
    global HEAD_CONFIGS
    args         = parse_args()

    pl.seed_everything(config.SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    HEAD_CONFIGS = build_head_configs(args.run_id)
    logger.info(f"HEAD_CONFIGS resolved for run_id={args.run_id}:")
    for k, v in HEAD_CONFIGS.items():
        logger.info(f"  [{k}] {v['dataset_name']} -> {v['data_file'].name}")

    model_cfg = MODEL_REGISTRY[args.model_name]
    hf_name   = model_cfg["hf_name"]
    max_tok   = model_cfg["max_tokens"]
    trust_rc  = model_cfg["trust_remote_code"]
    is_pep    = model_cfg["is_pepdora"]

    if not args.pretrain_ckpt:
        if args.model_name == "chemberta-77m-mtr":
            _check_local_dir(CHEMBERTA_MTR_DIR,
                ["config.json", "model.safetensors"], "ChemBERTa-77M-MTR")
        elif args.model_name == "pepdora":
            _check_local_dir(CHEMBERTA_MLM_WEIGHTS_DIR,
                ["config.json", "model.safetensors"], "ChemBERTa-77M-MLM weights")
            _check_local_dir(CHEMBERTA_MLM_TOKENIZER_DIR,
                ["tokenizer.json", "vocab.json", "merges.txt"], "ChemBERTa-77M-MLM tokenizer")
            _check_local_dir(PEPDORA_ADAPTER_DIR,
                ["adapter_config.json", "adapter_model.safetensors"], "PepDoRA adapter")

    out_dir = (config.BASE_DIR / "runs" / "finetune_transformer" /
               args.model_name / args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TRANSFORMER MULTI-HEAD FINETUNE + EVALUATE")
    logger.info(f"  Model:    {hf_name}")
    logger.info(f"  Max tok:  {max_tok}")
    logger.info(f"  Run:      {args.run_id}")
    logger.info(f"  LR:       {args.lr}  Batch: {args.batch_size}  Epochs: {args.epochs}")
    logger.info(f"  Output:   {out_dir}")
    logger.info("=" * 60)

    #  Tokenizer + data splitting
    tok_name  = model_cfg["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_rc)
    merged_df, filter_logs, head_dfs = load_and_merge(tokenizer, max_tok)
    groups = merged_df[config.CLUSTER_COL].values
    smiles = merged_df[config.SMILES_COL].values

    per_dataset, fold_train_smiles, fold_val_smiles, test_smiles = make_multihead_splits(
        head_dfs, config.SMILES_COL, config.CLUSTER_COL, config.N_FOLDS,
        config.SPLIT_STRATEGY, config.SEED, test_frac=0.10,
    )
    train_folds, val_folds, test_idx = resolve_multihead_indices(
        smiles, fold_train_smiles, fold_val_smiles, test_smiles, config.N_FOLDS)
    logger.info(f"Split — test: {len(test_idx)} | "
                f"fold-1 train: {len(train_folds[0])} | fold-1 val: {len(val_folds[0])}")
    for fold_i in range(config.N_FOLDS):
        for name, d in per_dataset.items():
            if len(d["val_folds"][fold_i]) == 0:
                logger.warning(f"  Fold {fold_i + 1}: '{name}' has 0 val samples!")

    split_meta = {
        "n_total":        len(groups),
        "split_strategy": config.SPLIT_STRATEGY,
        "seed":           config.SEED,
        "n_folds":        config.N_FOLDS,
        "test_idx":       test_idx.tolist(),
        "train_folds":    [f.tolist() for f in train_folds],
        "val_folds":      [f.tolist() for f in val_folds],
        "split_method":   "per_dataset_independent_then_unioned",
    }
    with open(out_dir / "split_indices.json", "w") as f:
        json.dump(split_meta, f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_encoder():
        if args.pretrain_ckpt:
            return AutoModel.from_pretrained(args.pretrain_ckpt, trust_remote_code=trust_rc)
        elif is_pep:
            from peft import PeftModel
            base = AutoModel.from_pretrained(
                model_cfg["base_model"], trust_remote_code=trust_rc,
            )
            enc  = PeftModel.from_pretrained(base, model_cfg["hf_name"])
            return enc.merge_and_unload()
        else:
            return AutoModel.from_pretrained(hf_name, trust_remote_code=trust_rc)

    #  5-fold CV on 90% of data
    fold_results = {name: [] for name in HEAD_KEYS}

    for fold in range(config.N_FOLDS):
        logger.info(f"\n=== CV Fold {fold+1}/{config.N_FOLDS} ===")
        train_idx = train_folds[fold]
        val_idx   = val_folds[fold]
        logger.info(f"  Train: {len(train_idx)}  Val: {len(val_idx)}")

        scalers       = fit_scalers(merged_df, train_idx)
        train_targets = scale_targets(merged_df, train_idx, scalers)
        val_targets   = scale_targets(merged_df, val_idx,   scalers)

        train_ds = TransformerPeptideDataset(smiles[train_idx], train_targets, tokenizer, max_tok)
        val_ds   = TransformerPeptideDataset(smiles[val_idx],   val_targets,   tokenizer, max_tok)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
                                  persistent_workers=(config.NUM_WORKERS > 0),
                                  generator=torch.Generator().manual_seed(config.SEED))
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
                                  persistent_workers=(config.NUM_WORKERS > 0))

        encoder = _load_encoder()
        model   = TransformerMultiHead(
            encoder=encoder, hidden_size=encoder.config.hidden_size,
            scalers=scalers, lr=args.lr, warmup_steps=args.warmup_steps,
            freeze_encoder_epochs=args.freeze_encoder_epochs, dropout=args.dropout,
        )
        trainer = pl.Trainer(
            max_epochs=args.epochs, accelerator=device, devices=1,
            logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=(fold == 0),
        )
        trainer.fit(model, train_loader, val_loader)

        preds = run_inference(model, val_loader, device)
        for name in HEAD_KEYS:
            y_true = merged_df[f"label_{name}"].values[val_idx]
            m      = compute_metrics(y_true, preds[name])
            m["fold"] = fold + 1
            fold_results[name].append(m)
            logger.info(f"  [{name}] r2={m.get('r2', float('nan')):.3f}  "
                        f"pearson={m.get('pearson_r', float('nan')):.3f}  "
                        f"rmse={m.get('rmse', float('nan')):.4f}")

    cv_agg = {}
    for name in HEAD_KEYS:
        cv_agg[name] = {}
        for metric in ["rmse", "mae", "r2", "pearson_r", "spearman_r"]:
            vals = [r[metric] for r in fold_results[name] if metric in r]
            if vals:
                cv_agg[name][f"mean_{metric}"] = float(np.mean(vals))
                cv_agg[name][f"std_{metric}"]  = float(np.std(vals))
        logger.info(f"  [{name}] CV agg: {cv_agg[name]}")

    #  Train on full 90%, evaluate on 10% test set
    logger.info("\n=== Training 90% model for test-set evaluation ===")
    trainval_idx = np.sort(np.concatenate([train_folds[0], val_folds[0]]))

    ninety_scalers       = fit_scalers(merged_df, trainval_idx)
    ninety_train_targets = scale_targets(merged_df, trainval_idx, ninety_scalers)

    ninety_train_ds = TransformerPeptideDataset(smiles[trainval_idx], ninety_train_targets, tokenizer, max_tok)
    ninety_train_loader = DataLoader(ninety_train_ds, batch_size=args.batch_size, shuffle=True,
                                     num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
                                     persistent_workers=(config.NUM_WORKERS > 0),
                                     generator=torch.Generator().manual_seed(config.SEED))
    ninety_val_loader   = None

    ninety_encoder = _load_encoder()
    ninety_model   = TransformerMultiHead(
        encoder=ninety_encoder, hidden_size=ninety_encoder.config.hidden_size,
        scalers=ninety_scalers, lr=args.lr, warmup_steps=args.warmup_steps,
        freeze_encoder_epochs=args.freeze_encoder_epochs, dropout=args.dropout,
    )
    ninety_trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator=device, devices=1,
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    ninety_trainer.fit(ninety_model, ninety_train_loader, ninety_val_loader)

    test_metrics = {}
    logger.info("=== TEST SET EVALUATION ===")
    test_targets_dummy = {name: np.full(len(test_idx), np.nan) for name in HEAD_KEYS}
    test_ds     = TransformerPeptideDataset(smiles[test_idx], test_targets_dummy, tokenizer, max_tok)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
                             persistent_workers=(config.NUM_WORKERS > 0))
    test_preds  = run_inference(ninety_model, test_loader, device)
    for name in HEAD_KEYS:
        y_true = merged_df[f"label_{name}"].values[test_idx]
        m      = compute_metrics(y_true, test_preds[name])
        test_metrics[name] = m
        logger.info(f"  [{name}] TEST r2={m.get('r2', float('nan')):.3f}  "
                    f"pearson={m.get('pearson_r', float('nan')):.3f}  ")

    #  Save the 90% model as the production checkpoint (test set never seen in training)
    final_dir = out_dir / "final_model"
    final_dir.mkdir(exist_ok=True)
    torch.save(ninety_model.state_dict(), final_dir / "model.pt")
    torch.save(
        {name: {"mean": sc.mean_, "scale": sc.scale_} for name, sc in ninety_scalers.items()},
        final_dir / "scalers.pt",
    )
    logger.info(f"Final model (90% train+val, test held out) saved: {final_dir / 'model.pt'}")

    #  Save full results JSON 
    result_record = {
        "run_timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_date":              _RUN_DATE,
        "job_id":                _JOB_ID,
        "model_arch":            "transformer_multi_head",
        "model_name":            args.model_name,
        "hf_name":               hf_name,
        "run_id":                args.run_id,
        "max_tokens":            max_tok,
        "token_filter_logs":     filter_logs,
        "lr":                    args.lr,
        "batch_size":            args.batch_size,
        "epochs":                args.epochs,
        "dropout":               args.dropout,
        "freeze_encoder_epochs": args.freeze_encoder_epochs,
        "n_folds":               config.N_FOLDS,
        "split_strategy":        config.SPLIT_STRATEGY,
        "seed":                  config.SEED,
        "per_head_fold_results": fold_results,
        "cv_agg":                cv_agg,
        "test_metrics":          test_metrics,
    }
    json_path = (config.RESULTS_DIR /
                 f"transformer_{args.model_name}_{args.run_id}_{config.SPLIT_STRATEGY}_results.json")
    with open(json_path, "w") as f:
        json.dump(result_record, f, indent=2)
    logger.info(f"Results saved: {json_path}")

    #  Upsert into transformer_results.csv (trimmed schema) 
    rows = []
    run_name = f"{args.run_id}_{args.model_name}"
    base_row = {
        "run_timestamp":  result_record["run_timestamp"],
        "run_date":       _RUN_DATE,
        "job_id":         _JOB_ID,
        "model_arch":     f"transformer_multi_head_{args.model_name}",
        "run_name":       run_name,
        "pipeline_mode":  "transformer",
        "subset_label":   args.model_name,
        "split_strategy": config.SPLIT_STRATEGY,
    }
    for name in HEAD_KEYS:
        dataset_name = HEAD_CONFIGS[name]["data_file"].stem
        rows.append({
            **base_row,
            "head":         name,
            "dataset_name": dataset_name,
            **cv_agg[name],
        })
        tm = test_metrics.get(name, {})
        if tm:
            rows.append({
                **base_row,
                "head":         name,
                "dataset_name": dataset_name,
                "rmse":         tm.get("rmse"),
                "mae":          tm.get("mae"),
                "r2":           tm.get("r2"),
                "pearson_r":    tm.get("pearson_r"),
                "spearman_r":   tm.get("spearman_r"),
            })

    append_results(rows, config.TRANSFORMER_RESULTS_CSV)
    logger.info(f"Results upserted to: {config.TRANSFORMER_RESULTS_CSV}")


if __name__ == "__main__":
    main()
