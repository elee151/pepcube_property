"""
finetune_transformer_singletask.py
Single-task finetuning of transformer models (ChemBERTa, PepDoRA).
One shared encoder + one linear regression head, trained per dataset.

Example Usage:
    python finetune_transformer_singletask.py --model_name pepdora --dataset_name hemolysis --fold 1
    python finetune_transformer_singletask.py --model_name chemberta-77m-mtr --dataset_name synthesizability
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MODEL_ARCH",                   "single_task")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",  "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
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
)  # has config.json + model.safetensors

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


#  Dataset
class TransformerPeptideDataset(Dataset):
    def __init__(self, smiles, targets, tokenizer, max_tokens):
        self.smiles      = smiles
        self.targets     = targets   # np.ndarray of scaled labels
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
        item["label"] = torch.tensor(float(self.targets[idx]), dtype=torch.float32)
        return item


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


class TransformerSingleTask(pl.LightningModule):
    """
    Shared transformer encoder + single linear regression head.
    Uses token (index 0) as the molecular embedding.
    """
    def __init__(self, encoder, hidden_size, scaler,
                 lr=1e-4, warmup_steps=100,
                 freeze_encoder_epochs=0, dropout=0.1):
        super().__init__()
        self.encoder               = encoder
        self.scaler                = scaler
        self.lr                    = lr
        self.warmup_steps          = warmup_steps
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.dropout_layer         = nn.Dropout(dropout)
        self.head                  = nn.Linear(hidden_size, 1)
        self.mse                   = nn.MSELoss()

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
        return self.head(h).squeeze(-1)

    def _shared_step(self, batch, stage):
        pred = self(batch)
        loss = self.mse(pred, batch["label"])
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=batch["input_ids"].shape[0])
        return loss

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
        enc_batch = {k: v for k, v in batch.items() if k != "label"}
        with torch.no_grad():
            z = self(enc_batch).cpu().numpy()
        return z * self.scaler.scale_[0] + self.scaler.mean_[0]


#  Scaler helpers
def fit_scaler(df, label_col, train_idx):
    vals = df[label_col].values[train_idx].reshape(-1, 1)
    return StandardScaler().fit(vals)


def scale_targets(df, label_col, indices, scaler):
    arr = df[label_col].values[indices].reshape(-1, 1)
    return scaler.transform(arr).flatten()


#  Metrics
def compute_metrics(y_true, y_pred) -> dict:
    mask   = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return {}
    r_p, _ = pearsonr(y_true, y_pred)
    r_s, _ = spearmanr(y_true, y_pred)
    return {
        "rmse":       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":        float(mean_absolute_error(y_true, y_pred)),
        "r2":         float(r2_score(y_true, y_pred)),
        "pearson_r":  float(r_p),
        "spearman_r": float(r_s),
    }


#  Inference
def run_inference(model, loader, device):
    model.eval()
    model.to(device)
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            enc_batch = {
                k: v.to(device) for k, v in batch.items() if k != "label"
            }
            all_preds.append(model.predict_original_scale(enc_batch))
    return np.concatenate(all_preds)


#  Final model: train on 90% train+val, evaluate on held-out 10% test set
def train_final_model(args, df, smiles, label_col, tokenizer, max_tok,
                      load_encoder, out_dir, device):
    split_path = out_dir / "split_indices.json"
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} not found.")
    with open(split_path) as f:
        split_meta = json.load(f)
    test_idx    = np.array(split_meta["test_idx"], dtype=int)
    train_folds = [np.array(f, dtype=int) for f in split_meta["train_folds"]]
    val_folds   = [np.array(f, dtype=int) for f in split_meta["val_folds"]]

    trainval_idx = np.sort(np.concatenate([train_folds[0], val_folds[0]]))
    logger.info(f"Training final model on 90% train+val data: {len(trainval_idx)} rows")

    scaler        = fit_scaler(df, label_col, trainval_idx)
    train_targets = scale_targets(df, label_col, trainval_idx, scaler)

    train_ds     = TransformerPeptideDataset(smiles[trainval_idx], train_targets, tokenizer, max_tok)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
                              persistent_workers=(config.NUM_WORKERS > 0),
                              generator=torch.Generator().manual_seed(config.SEED))

    encoder = load_encoder()
    model   = TransformerSingleTask(
        encoder=encoder, hidden_size=encoder.config.hidden_size,
        scaler=scaler, lr=args.lr, warmup_steps=args.warmup_steps,
        freeze_encoder_epochs=args.freeze_encoder_epochs, dropout=args.dropout,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator=device, devices=1,
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, train_loader, None)

    test_targets_dummy = np.full(len(test_idx), np.nan)
    test_ds     = TransformerPeptideDataset(smiles[test_idx], test_targets_dummy, tokenizer, max_tok)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
                             persistent_workers=(config.NUM_WORKERS > 0))
    test_preds  = run_inference(model, test_loader, device)
    y_true      = df[label_col].values[test_idx]
    test_metrics = compute_metrics(y_true, test_preds)
    logger.info(f"  TEST r2={test_metrics.get('r2', float('nan')):.3f}  "
                f"pearson={test_metrics.get('pearson_r', float('nan')):.3f}")

    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_dir / "finetuned_model.pt")
    torch.save({"mean": scaler.mean_, "scale": scaler.scale_}, final_dir / "scaler.pt")
    logger.info(f"Final model (90% train+val, test held out) saved: {final_dir / 'finetuned_model.pt'}")

    return test_metrics


#  Results writing
def save_results(args, ds_cfg, filter_logs, hf_name, cv_agg, test_metrics):
    result_record = {
        "run_timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_date":              _RUN_DATE,
        "job_id":                _JOB_ID,
        "model_arch":            "transformer_single_task",
        "model_name":            args.model_name,
        "hf_name":               hf_name,
        "dataset_name":          args.dataset_name,
        "max_tokens":            filter_logs["max_tokens"],
        "token_filter_logs":     filter_logs,
        "lr":                    args.lr,
        "batch_size":            args.batch_size,
        "epochs":                args.epochs,
        "dropout":               args.dropout,
        "freeze_encoder_epochs": args.freeze_encoder_epochs,
        "n_folds":               config.N_FOLDS,
        "split_strategy":        config.SPLIT_STRATEGY,
        "seed":                  config.SEED,
        "cv_agg":                cv_agg,
        "test_metrics":          test_metrics,
    }
    json_path = (config.RESULTS_DIR /
                f"transformer_{args.model_name}_{args.dataset_name}_{config.SPLIT_STRATEGY}_results.json")
    with open(json_path, "w") as f:
        json.dump(result_record, f, indent=2)
    logger.info(f"Results saved: {json_path}")

    base_row = {
        "run_timestamp":  result_record["run_timestamp"],
        "run_date":       _RUN_DATE,
        "job_id":         _JOB_ID,
        "model_arch":     f"transformer_single_task_{args.model_name}",
        "run_name":       f"{args.model_name}_{args.dataset_name}",
        "dataset_name":   ds_cfg["data_file"].stem,
        "pipeline_mode":  "transformer",
        "subset_label":   args.model_name,
        "split_strategy": config.SPLIT_STRATEGY,
    }
    rows = []
    if cv_agg:
        rows.append({**base_row, "head": args.dataset_name, **cv_agg})
    if test_metrics:
        rows.append({
            **base_row,
            "head":       args.dataset_name,
            "rmse":       test_metrics.get("rmse"),
            "mae":        test_metrics.get("mae"),
            "r2":         test_metrics.get("r2"),
            "pearson_r":  test_metrics.get("pearson_r"),
            "spearman_r": test_metrics.get("spearman_r"),
        })

    append_results(rows, config.TRANSFORMER_RESULTS_CSV)
    logger.info(f"Results appended to: {config.TRANSFORMER_RESULTS_CSV}")


#  Main 

def parse_args():
    parser = argparse.ArgumentParser(description="Single-task transformer finetuning")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Which transformer to finetune")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument("--lr",                    type=float, default=2e-5)
    parser.add_argument("--batch_size",            type=int,   default=16)
    parser.add_argument("--epochs",                type=int,   default=20)
    parser.add_argument("--dropout",                type=float, default=0.1)
    parser.add_argument("--freeze_encoder_epochs",  type=int,   default=0)
    parser.add_argument("--warmup_steps",           type=int,   default=100)
    parser.add_argument("--pretrain_ckpt",          type=str,   default=None,
                        help="Path to pretrained backbone dir (overrides HF load)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run a single fold only (1-indexed). Saves split_indices.json on fold 1.")
    parser.add_argument("--final_only", action="store_true",
                        help="Skip CV folds; train final model only. Requires split_indices.json to exist.")
    return parser.parse_args()


def main():
    args = parse_args()

    pl.seed_everything(config.SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    ds_cfg    = config.DATASET_CONFIG[args.dataset_name]
    label_col = ds_cfg["label_col"]

    out_dir = config.finetune_dir(args.model_name, args.dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.final_only:
        mode_str = "final model only"
    elif args.fold:
        mode_str = f"fold {args.fold} only"
    else:
        mode_str = "all folds + final"

    logger.info("=" * 60)
    logger.info(f"TRANSFORMER SINGLE-TASK FINETUNE + EVALUATE | {mode_str}")
    logger.info(f"  Model:    {hf_name}")
    logger.info(f"  Max tok:  {max_tok}")
    logger.info(f"  Dataset:  {args.dataset_name}")
    logger.info(f"  LR:       {args.lr}  Batch: {args.batch_size}  Epochs: {args.epochs}")
    logger.info(f"  Output:   {out_dir}")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_path"], trust_remote_code=trust_rc)

    df = pd.read_csv(ds_cfg["data_file"])
    for col in [config.SMILES_COL, label_col, config.CLUSTER_COL]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
    df = df[[config.SMILES_COL, config.CLUSTER_COL, label_col]].dropna(
        subset=[config.SMILES_COL, config.CLUSTER_COL]
    ).reset_index(drop=True)
    df, filter_logs = filter_by_token_length(df, tokenizer, max_tok, config.SMILES_COL)
    logger.info(f"Loaded {len(df)} rows. {df[config.CLUSTER_COL].nunique()} unique clusters")

    smiles = df[config.SMILES_COL].values
    groups = df[config.CLUSTER_COL].values

    def load_encoder():
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

    # Final-only mode
    if args.final_only:
        test_metrics = train_final_model(args, df, smiles, label_col, tokenizer, max_tok,
                                         load_encoder, out_dir, device)
        save_results(args, ds_cfg, filter_logs, hf_name, {}, test_metrics)
        return

    # Compute / load splits
    split_path = out_dir / "split_indices.json"

    if split_path.exists():
        with open(split_path) as f:
            split_meta = json.load(f)
        train_folds = [np.array(f, dtype=int) for f in split_meta["train_folds"]]
        val_folds   = [np.array(f, dtype=int) for f in split_meta["val_folds"]]
        test_idx    = np.array(split_meta["test_idx"], dtype=int)
    else:
        train_folds, val_folds, test_idx = make_train_val_test_indices(
            groups, config.N_FOLDS, config.SPLIT_STRATEGY, config.SEED, test_frac=0.10)

        if args.fold is None or args.fold == 1:
            split_meta = {
                "n_total":        len(groups),
                "split_strategy": config.SPLIT_STRATEGY,
                "seed":           config.SEED,
                "n_folds":        config.N_FOLDS,
                "test_idx":       test_idx.tolist(),
                "train_folds":    [f.tolist() for f in train_folds],
                "val_folds":      [f.tolist() for f in val_folds],
            }
            with open(split_path, "w") as f:
                json.dump(split_meta, f)
            logger.info(f"Split indices saved: {split_path}")

    logger.info(f"Split — test: {len(test_idx)} | "
                f"fold-1 train: {len(train_folds[0])} | fold-1 val: {len(val_folds[0])}")

    # CV folds
    fold_range   = [args.fold - 1] if args.fold is not None else range(config.N_FOLDS)
    fold_results = []

    for fold in fold_range:
        logger.info(f"\n=== CV Fold {fold+1}/{config.N_FOLDS} ===")
        train_idx = train_folds[fold]
        val_idx   = val_folds[fold]
        logger.info(f"  Train: {len(train_idx)}  Val: {len(val_idx)}")

        scaler        = fit_scaler(df, label_col, train_idx)
        train_targets = scale_targets(df, label_col, train_idx, scaler)
        val_targets   = scale_targets(df, label_col, val_idx,   scaler)

        train_ds = TransformerPeptideDataset(smiles[train_idx], train_targets, tokenizer, max_tok)
        val_ds   = TransformerPeptideDataset(smiles[val_idx],   val_targets,   tokenizer, max_tok)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
                                  persistent_workers=(config.NUM_WORKERS > 0),
                                  generator=torch.Generator().manual_seed(config.SEED))
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
                                  persistent_workers=(config.NUM_WORKERS > 0))

        encoder = load_encoder()
        model   = TransformerSingleTask(
            encoder=encoder, hidden_size=encoder.config.hidden_size,
            scaler=scaler, lr=args.lr, warmup_steps=args.warmup_steps,
            freeze_encoder_epochs=args.freeze_encoder_epochs, dropout=args.dropout,
        )
        trainer = pl.Trainer(
            max_epochs=args.epochs, accelerator=device, devices=1,
            logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=(fold == 0),
        )
        trainer.fit(model, train_loader, val_loader)

        val_preds = run_inference(model, val_loader, device)
        y_true    = df[label_col].values[val_idx]
        m         = compute_metrics(y_true, val_preds)
        m["fold"] = fold + 1
        fold_results.append(m)
        logger.info(f"  fold {fold+1} r2={m.get('r2', float('nan')):.3f}  "
                    f"pearson={m.get('pearson_r', float('nan')):.3f}  "
                    f"rmse={m.get('rmse', float('nan')):.4f}")

        fold_dir = out_dir / f"fold_{fold+1}"
        fold_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), fold_dir / "finetuned_model.pt")
        torch.save({"mean": scaler.mean_, "scale": scaler.scale_}, fold_dir / "scaler.pt")

    cv_agg = {}
    for metric in ["rmse", "mae", "r2", "pearson_r", "spearman_r"]:
        vals = [r[metric] for r in fold_results if metric in r]
        if vals:
            cv_agg[f"mean_{metric}"] = float(np.mean(vals))
            cv_agg[f"std_{metric}"]  = float(np.std(vals))
    logger.info(f"CV agg: {cv_agg}")

    # Single-fold mode: save fold summary and exit
    if args.fold is not None:
        logger.info(f"Single-fold mode complete — fold {args.fold} done.")
        with open(out_dir / f"fold_{args.fold}_summary.json", "w") as f:
            json.dump({
                "run_date": _RUN_DATE, "job_id": _JOB_ID,
                "model_arch": "transformer_single_task", "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "fold": args.fold, "single_fold_mode": True,
                "fold_results": fold_results,
            }, f, indent=2)
        return

    # All-folds mode: train final model on 90%, evaluate on test set
    test_metrics = train_final_model(args, df, smiles, label_col, tokenizer, max_tok,
                                     load_encoder, out_dir, device)
    save_results(args, ds_cfg, filter_logs, hf_name, cv_agg, test_metrics)


if __name__ == "__main__":
    main()
