"""
hpo_pretrain_chemprop.py
Phase 2 hyperparameter optimization for pretraining training.
Searches lr, batch_size, epochs, dropout, freeze_encoder_epochs

Example usage:
    PIPELINE_MODE=chemprop ACCELERATOR=gpu python hpo_pretrain_chemprop.py --subset_name all_generated_scored_hpo100k
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from functools import partial as fn_partial

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_RUN_DATE = datetime.now().strftime("%Y-%m-%d")
_JOB_ID   = os.environ.get("PBS_JOBID", "local").split(".")[0]

#  HPO settings
HPO_EPOCHS    = int(os.environ.get("HPO_EPOCHS",    20))   # epochs per trial
HPO_P2_TRIALS = int(os.environ.get("HPO_P2_TRIALS", 30))   # number of random trials
HPO_SEED      = int(os.environ.get("HPO_SEED",      42))

OUTPUT_DIR = Path("hpo_pretrain_results")
OUTPUT_DIR.mkdir(exist_ok=True)

#  Optimal architecture from previous p1 pretraining HPO
OPTIMAL_ARCH = {
    "depth":          6,
    "d_h":            600,
    "ffn_num_layers": 2,
    "ffn_hidden_dim": 300,
    "aggregation":    "mean",
}

#  Phase 2 grid - learning parameters
PHASE2_RANDOM = {
    "lr":                   [1e-5, 5e-5, 1e-4, 5e-4],
    "batch_size":           [64, 128, 256, 512],
    "epochs":               [20, 30, 50, 75],
    "dropout":              [0.0, 0.1, 0.2, 0.3],
    "freeze_encoder_epochs":[0, 3, 5, 10],
}

PRETRAIN_HEAD_KEYS = config.PRETRAIN_LABEL_COLS

#  Adjusts ph column to make sure no errors in key
def _safe_key(name: str) -> str:
    return name.replace(".", "_")

SAFE_TO_COL = {_safe_key(k): k for k in PRETRAIN_HEAD_KEYS}

#  Dataset
class PretrainMultiHeadDataset(torch.utils.data.Dataset):
    def __init__(self, smiles, targets_dict):
        self.smiles       = smiles
        self.targets_dict = targets_dict

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return (
            self.smiles[idx],
            {name: float(self.targets_dict[name][idx]) for name in PRETRAIN_HEAD_KEYS},
        )


def collate_pretrain(batch, featurizer):
    smiles_list  = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]
    molgraphs = [featurizer(Chem.MolFromSmiles(s)) for s in smiles_list]
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
    collate_fn = fn_partial(collate_pretrain, featurizer=featurizer)
    generator = torch.Generator().manual_seed(config.SEED) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=(config.NUM_WORKERS > 0),
        generator=generator,
    )

#  Scaler helpers
def fit_scalers(df, indices=None):
    """Fit scalers on training indices only (or full df if indices=None)."""
    subset = df.iloc[indices] if indices is not None else df
    scalers = {}
    for name in PRETRAIN_HEAD_KEYS:
        vals = subset[name].dropna().values.reshape(-1, 1)
        if len(vals) == 0:
            raise ValueError(f"Head '{name}' has no non-NaN values in training set.")
        sc = StandardScaler().fit(vals)
        scalers[name] = sc
    return scalers

def scale_targets(df, indices, scalers):
    """Scale targets for a given set of indices using pre-fitted scalers."""
    targets = {}
    for name in PRETRAIN_HEAD_KEYS:
        sc  = scalers[name]
        arr = df[name].values[indices]
        out = np.full(len(arr), np.nan)
        mask = ~np.isnan(arr)
        if mask.any():
            out[mask] = sc.transform(arr[mask].reshape(-1, 1)).flatten()
        targets[name] = out
    return targets

#  Single split for pretraining HPO
def make_pretrain_split(df, cluster_col, seed):
    """Split cluster groups into 80% train / 10% val / 10% test"""
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].values
    unique_clusters = np.unique(clusters)
    rng.shuffle(unique_clusters)

    n = len(unique_clusters)
    n_val  = max(1, int(round(n * 0.10)))
    n_test = max(1, int(round(n * 0.10)))

    test_clusters  = set(unique_clusters[:n_test])
    val_clusters   = set(unique_clusters[n_test:n_test + n_val])
    train_clusters = set(unique_clusters[n_test + n_val:])

    all_idx    = np.arange(len(df))
    train_idx  = all_idx[np.isin(clusters, list(train_clusters))]
    val_idx    = all_idx[np.isin(clusters, list(val_clusters))]
    test_idx   = all_idx[np.isin(clusters, list(test_clusters))]

    logger.info(
        f"Split: train={len(train_idx)} ({len(train_clusters)} clusters) | "
        f"val={len(val_idx)} ({len(val_clusters)} clusters) | "
        f"test={len(test_idx)} ({len(test_clusters)} clusters)"
    )
    return train_idx, val_idx, test_idx

#  Lightning module
class PretrainMultiHeadMPNN(pl.LightningModule):
    def __init__(self, mp, agg, heads, scalers,
                 lr=6e-4, warmup_epochs=2, max_epochs=30, dropout=0.0,
                 freeze_encoder_epochs=0):
        super().__init__()
        self.message_passing      = mp
        self.agg                  = agg
        self.heads                = nn_torch.ModuleDict(heads)
        self.scalers              = scalers
        self.lr                   = lr
        self.warmup_epochs        = warmup_epochs
        self.max_epochs           = max_epochs
        self.freeze_encoder_epochs= freeze_encoder_epochs
        self.mse_loss             = nn_torch.MSELoss(reduction="none")

    def forward(self, bmg):
        h = self.agg(self.message_passing(bmg), bmg.batch)
        return {SAFE_TO_COL[k]: head(h) for k, head in self.heads.items()}

    def _masked_mse(self, pred, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return self.mse_loss(pred[mask], target[mask]).mean()

    def _shared_step(self, batch, stage):
        bmg, targets_dict = batch
        preds      = self(bmg)
        batch_size = int(bmg.batch.max().item()) + 1
        total      = torch.tensor(0.0, device=self.device)
        logs       = {}
        for name in PRETRAIN_HEAD_KEYS:
            l = self._masked_mse(preds[name].squeeze(-1), targets_dict[name])
            logs[f"{stage}_{name}_loss"] = l
            total = total + l
        avg = total / len(PRETRAIN_HEAD_KEYS)
        logs[f"{stage}_loss"] = avg
        self.log_dict(logs, prog_bar=(stage == "val"),
                      on_epoch=True, on_step=False, batch_size=batch_size)
        return avg

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_train_epoch_start(self):
        if self.current_epoch == self.freeze_encoder_epochs and self.freeze_encoder_epochs > 0:
            for p in self.message_passing.parameters():
                p.requires_grad = True
            logger.info(f"  Encoder unfrozen at epoch {self.current_epoch}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


#  Model builder
def build_model(scalers, encoder_ckpt, dynamics):
    """Build PretrainMultiHeadMPNN with fixed arch + learning parameters."""
    arch = OPTIMAL_ARCH

    # Encoder
    if encoder_ckpt is None:
        mp = cpnn.BondMessagePassing(depth=arch["depth"], d_h=arch["d_h"])
    else:
        ckpt_data = torch.load(encoder_ckpt, map_location="cpu", weights_only=True)
        if "hyper_parameters" in ckpt_data:
            mp = cpnn.BondMessagePassing(**ckpt_data["hyper_parameters"])
            mp.load_state_dict(ckpt_data["state_dict"])
        else:
            mp_state    = {k.replace("message_passing.", ""): v
                           for k, v in ckpt_data.items()
                           if k.startswith("message_passing.")}
            hidden_size = mp_state["W_i.weight"].shape[0]
            depth       = sum(1 for k in mp_state if k.startswith("W_h"))
            mp = cpnn.BondMessagePassing(d_h=hidden_size, depth=depth)
            mp.load_state_dict(mp_state, strict=False)

    # Aggregation
    agg_map = {
        "mean": cpnn.MeanAggregation(),
        "sum":  cpnn.SumAggregation(),
        "norm": cpnn.NormAggregation(),
    }
    agg = agg_map[arch["aggregation"]]

    # Per-head FFN with dropout
    dropout = dynamics.get("dropout", 0.0)
    heads   = {}
    for name in PRETRAIN_HEAD_KEYS:
        layers = []
        in_dim = mp.output_dim
        for _ in range(arch["ffn_num_layers"] - 1):
            layers += [
                nn_torch.Linear(in_dim, arch["ffn_hidden_dim"]),
                nn_torch.ReLU(),
                nn_torch.Dropout(dropout),
            ]
            in_dim = arch["ffn_hidden_dim"]
        layers.append(nn_torch.Linear(in_dim, 1))
        heads[_safe_key(name)] = nn_torch.Sequential(*layers)

    freeze_ep = dynamics.get("freeze_encoder_epochs", 0)
    if freeze_ep > 0:
        for p in mp.parameters():
            p.requires_grad = False

    model = PretrainMultiHeadMPNN(
        mp=mp, agg=agg, heads=heads, scalers=scalers,
        lr=dynamics["lr"],
        warmup_epochs=2,
        max_epochs=dynamics["epochs"],
        dropout=dropout,
        freeze_encoder_epochs=freeze_ep,
    )
    return model


#  Runs the specified single split
def run_trial(dynamics, train_loader, val_loader, encoder_ckpt, scalers):
    try:
        model = build_model(scalers, encoder_ckpt, dynamics)
        trainer = pl.Trainer(
            max_epochs=dynamics["epochs"],
            accelerator=config.ACCELERATOR,
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(model, train_loader, val_loader)
        val_loss   = float(trainer.callback_metrics.get(
            "val_loss", torch.tensor(float("nan"))).item())
        train_loss = float(trainer.callback_metrics.get(
            "train_loss", torch.tensor(float("nan"))).item())
        return val_loss, train_loss, model
    except Exception as e:
        logger.warning(f"  Trial failed: {e}")
        return np.inf, np.inf, None


#  Evaluate on test set
def evaluate_test(model, test_loader):
    """Run inference on test loader and return mean MSE across heads."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    mse_fn = nn_torch.MSELoss(reduction="none")
    with torch.no_grad():
        for bmg, targets_dict in test_loader:
            preds = model(bmg)
            batch_loss = 0.0
            for name in PRETRAIN_HEAD_KEYS:
                pred   = preds[name].squeeze(-1)
                target = targets_dict[name]
                mask   = ~torch.isnan(target)
                if mask.sum() > 0:
                    batch_loss += mse_fn(pred[mask], target[mask]).mean().item()
            total_loss += batch_loss / len(PRETRAIN_HEAD_KEYS)
            n_batches  += 1
    return total_loss / max(n_batches, 1)


#  Main HPO loop
def run_phase2(df, train_idx, val_idx, test_idx, encoder_ckpt, args):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    smiles_arr = df[config.SMILES_COL].values

    # Fit scalers on train only
    scalers = fit_scalers(df, train_idx)

    train_targets = scale_targets(df, train_idx, scalers)
    val_targets   = scale_targets(df, val_idx,   scalers)
    test_targets  = scale_targets(df, test_idx,  scalers)

    train_ds = PretrainMultiHeadDataset(smiles_arr[train_idx], train_targets)
    val_ds   = PretrainMultiHeadDataset(smiles_arr[val_idx],   val_targets)
    test_ds  = PretrainMultiHeadDataset(smiles_arr[test_idx],  test_targets)

    rng     = np.random.default_rng(HPO_SEED)
    results = []
    best_val_loss = np.inf
    best_dynamics = None
    best_model    = None

    logger.info("=" * 60)
    logger.info("PHASE 2: Pretraining Randomised search")
    logger.info(f"  Fixed arch:    {OPTIMAL_ARCH}")
    logger.info(f"  Trials:        {HPO_P2_TRIALS}")
    logger.info(f"  Epochs/trial:  {HPO_EPOCHS}")
    logger.info(f"  Encoder ckpt:  {encoder_ckpt}")
    logger.info("=" * 60)

    for i in range(HPO_P2_TRIALS):
        dynamics = {
            "lr":                   float(rng.choice(PHASE2_RANDOM["lr"])),
            "batch_size":           int(rng.choice(PHASE2_RANDOM["batch_size"])),
            "epochs":               int(rng.choice(PHASE2_RANDOM["epochs"])),
            "dropout":              float(rng.choice(PHASE2_RANDOM["dropout"])),
            "freeze_encoder_epochs":int(rng.choice(PHASE2_RANDOM["freeze_encoder_epochs"])),
        }

        t0 = time.time()
        logger.info(f"  Trial {i+1}/{HPO_P2_TRIALS}: {dynamics}")

        bs = dynamics["batch_size"]
        train_loader = build_loader(train_ds, featurizer, bs, shuffle=True)
        val_loader   = build_loader(val_ds,   featurizer, bs, shuffle=False)

        val_loss, train_loss, model = run_trial(
            dynamics, train_loader, val_loader, encoder_ckpt, scalers
        )
        elapsed = time.time() - t0
        logger.info(f"    train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({elapsed:.1f}s)")

        results.append({
            **dynamics,
            "val_loss":   val_loss,
            "train_loss": train_loss,
            "elapsed_s":  elapsed,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dynamics = copy.deepcopy(dynamics)
            best_model    = model
            logger.info(f"    *** New best val_loss={best_val_loss:.4f}")

    # Evaluate best config on held-out test set
    test_loss = np.inf
    if best_model is not None:
        test_loader = build_loader(test_ds, featurizer,
                                   best_dynamics["batch_size"], shuffle=False)
        test_loss = evaluate_test(best_model, test_loader)
        logger.info(f"\nBest config test_loss (holdout): {test_loss:.4f}")

    results.sort(key=lambda x: x["val_loss"])
    return results, best_dynamics, best_val_loss, test_loss


#  Argument parsing 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_name", type=str, required=True)
    parser.add_argument("--cluster_col", type=str, default="cluster_umap_hdb")
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    pl.seed_everything(HPO_SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_path = config.DATA_DIR / f"{args.subset_name}.csv"
    logger.info(f"=== PRETRAIN HPO ===")
    logger.info(f"Date/Job:      {_RUN_DATE} / {_JOB_ID}")
    logger.info(f"Pipeline mode: {config.PIPELINE_MODE}")
    logger.info(f"Accelerator:   {config.ACCELERATOR}")
    logger.info(f"Subset:        {data_path}  (HPO_EPOCHS={HPO_EPOCHS}, trials={HPO_P2_TRIALS})")
    logger.info(f"Fixed arch:    {OPTIMAL_ARCH}")

    df = pd.read_csv(data_path)
    df = df.dropna(subset=[config.SMILES_COL, args.cluster_col]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows, {df[args.cluster_col].nunique()} clusters")

    # Resolve encoder
    if args.pretrain_ckpt:
        encoder_ckpt = Path(args.pretrain_ckpt)
    elif config.PIPELINE_MODE == "chemeleon":
        encoder_ckpt = config.CHEMELEON_PATH
        logger.info(f"Using Chemeleon encoder: {encoder_ckpt}")
    else:
        encoder_ckpt = None
        logger.info("Using ChemProp default encoder")

    # 90/10/10 group-aware split
    train_idx, val_idx, test_idx = make_pretrain_split(
        df, args.cluster_col, args.split_seed
    )

    # Run Phase 2 randomised search
    results, best_dynamics, best_val_loss, test_loss = run_phase2(
        df, train_idx, val_idx, test_idx, encoder_ckpt, args
    )

    # Save all trial results
    phase2_out = {
        "run_date":       _RUN_DATE,
        "job_id":         _JOB_ID,
        "pipeline_mode":  config.PIPELINE_MODE,
        "subset_name":    args.subset_name,
        "cluster_col":    args.cluster_col,
        "split_seed":     args.split_seed,
        "n_train":        int(len(train_idx)),
        "n_val":          int(len(val_idx)),
        "n_test":         int(len(test_idx)),
        "fixed_arch":     OPTIMAL_ARCH,
        "hpo_epochs":     HPO_EPOCHS,
        "n_trials":       HPO_P2_TRIALS,
        "best_dynamics":  best_dynamics,
        "best_val_loss":  best_val_loss,
        "test_loss":      test_loss,
        "all_results":    results,
    }

    out_path = OUTPUT_DIR / "pretrain_random_results.json"
    with open(out_path, "w") as f:
        json.dump(phase2_out, f, indent=2)
    logger.info(f"Results saved: {out_path}")

    # Save best combined config
    best_config = {**OPTIMAL_ARCH, **best_dynamics, "test_loss": test_loss}
    cfg_path = OUTPUT_DIR / "best_pretrain_config.json"
    with open(cfg_path, "w") as f:
        json.dump(best_config, f, indent=2)

    logger.info("=" * 60)
    logger.info("BEST PRETRAIN CONFIG")
    logger.info("=" * 60)
    for k, v in best_config.items():
        logger.info(f"  {k:30s} = {v}")
    logger.info(f"  MPNN_DEPTH           = {OPTIMAL_ARCH['depth']}")
    logger.info(f"  MPNN_D_H             = {OPTIMAL_ARCH['d_h']}")
    logger.info(f"  FFN_NUM_LAYERS       = {OPTIMAL_ARCH['ffn_num_layers']}")
    logger.info(f"  FFN_HIDDEN_DIM       = {OPTIMAL_ARCH['ffn_hidden_dim']}")
    logger.info(f"  AGGREGATION          = '{OPTIMAL_ARCH['aggregation']}'")
    logger.info(f"  PRETRAIN_LR          = {best_dynamics['lr']}")
    logger.info(f"  PRETRAIN_BATCH       = {best_dynamics['batch_size']}")
    logger.info(f"  PRETRAIN_EPOCHS      = {best_dynamics['epochs']}")
    logger.info(f"  DROPOUT              = {best_dynamics['dropout']}")
    logger.info(f"  FREEZE_ENCODER_EPOCHS= {best_dynamics['freeze_encoder_epochs']}")
    logger.info(f"\nSaved: {cfg_path}")


if __name__ == "__main__":
    main()
