"""
hpo_finetune_chemprop.py
Two-phase hyperparameter optimisation for the multi-head Chemprop pipeline.
Evaluated with 5-fold CV on the 90% trainval portion (10% test is excluded).

Phase 1: Searches depth, message hidden dim, FFN layers, FFN hidden dim, aggregation.
Phase 2: Training dynamics (lr/batch_size/epochs/dropout/freeze_encoder_epochs)

Output into hpo_results as .jsons

Example usage:
    PIPELINE_MODE=scratch python hpo_finetune_chemprop.py --run_id run3 --phase 2
        --phase1_result hpo_results/phase1_grid_results_run3.json
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from itertools import product
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

from chemprop import featurizers, nn as cpnn

import sys as _sys
from pathlib import Path as _Path
_p = _Path(__file__).resolve().parent
while _p.name != "pepcube_property" and _p != _p.parent:
    _p = _p.parent
if _p.name == "pepcube_property" and str(_p.parent) not in _sys.path:
    _sys.path.insert(0, str(_p.parent))

import pepcube_property.config as config
from pepcube_property.utils import *

# Import shared helpers from finetune_chemprop_multitask.py
_ft_path = Path(__file__).parent / "finetune_chemprop_multitask.py"
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("finetune_multihead", _ft_path)
_ft   = _ilu.module_from_spec(_spec)
sys.modules["finetune_multihead"] = _ft
_spec.loader.exec_module(_ft)

MultiHeadMPNN           = _ft.MultiHeadMPNN
MultiHeadDataset        = _ft.MultiHeadDataset
collate_multihead       = _ft.collate_multihead
build_loader            = _ft.build_loader
load_and_merge_datasets = _ft.load_and_merge_datasets
fit_scalers             = _ft.fit_scalers
scale_targets           = _ft.scale_targets
HEAD_KEYS               = _ft.HEAD_KEYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_RUN_DATE = datetime.now().strftime("%Y-%m-%d")
_JOB_ID   = os.environ.get("PBS_JOBID", "local").split(".")[0]

HPO_EPOCHS    = int(os.environ.get("HPO_EPOCHS",    20))
HPO_FRAC      = float(os.environ.get("HPO_FRAC",   1.0))
HPO_SEED      = int(os.environ.get("HPO_SEED",      42))

OUTPUT_DIR = Path("hpo_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# Search spaces
# Phase 1: architecture
PHASE1_GRID = {
    "depth":          [3, 6],
    "d_h":            [300, 600],
    "ffn_num_layers": [1, 2],
    "ffn_hidden_dim": [300],
    "aggregation":    ["mean"],
}

# Phase 2: hyperparameters trial list
PHASE2_CURATED = [
    {"lr": 5e-4, "batch_size": 64,  "epochs": 20, "dropout": 0.3, "freeze_encoder_epochs": 20,
     "_label": "current_optimal"},
    {"lr": 5e-4, "batch_size": 64,  "epochs": 50, "dropout": 0.0, "freeze_encoder_epochs": 50},
    {"lr": 1e-4, "batch_size": 64,  "epochs": 20, "dropout": 0.0, "freeze_encoder_epochs": 20},
    {"lr": 5e-5, "batch_size": 64,  "epochs": 50, "dropout": 0.1, "freeze_encoder_epochs": 50},
    {"lr": 5e-4, "batch_size": 128, "epochs": 50, "dropout": 0.0, "freeze_encoder_epochs": 50},
    {"lr": 5e-4, "batch_size": 128, "epochs": 20, "dropout": 0.3, "freeze_encoder_epochs": 20},
    {"lr": 1e-4, "batch_size": 128, "epochs": 50, "dropout": 0.0, "freeze_encoder_epochs": 50,
     "_label": "config_defaults_frozen"},
    {"lr": 5e-4, "batch_size": 32,  "epochs": 30, "dropout": 0.0, "freeze_encoder_epochs": 30,
     "_label": "small_batch_frozen"},
    {"lr": 2e-4, "batch_size": 64,  "epochs": 30, "dropout": 0.0, "freeze_encoder_epochs": 30,
     "_label": "lr_midpoint_frozen"},
    {"lr": 1e-3, "batch_size": 64,  "epochs": 30, "dropout": 0.1, "freeze_encoder_epochs": 30,
     "_label": "higher_lr_frozen"},
]

# Deduplicate: strip _label for comparison, keep first occurrence
_seen = []
PHASE2_TRIALS = []
for cfg in PHASE2_CURATED:
    key = {k: v for k, v in cfg.items() if k != "_label"}
    if key not in _seen:
        _seen.append(key)
        PHASE2_TRIALS.append(cfg)

#  Model builder
def build_model_hpo(arch: dict, dynamics: dict, scalers: dict, encoder_ckpt):
    if encoder_ckpt is None:
        mp = cpnn.BondMessagePassing(depth=arch["depth"], d_h=arch["d_h"])
    else:
        encoder_state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
        raw_state = encoder_state.get("state_dict", encoder_state)

        if any(k.startswith("message_passing.") for k in raw_state):
            mp_state = {k[len("message_passing."):]: v
                        for k, v in raw_state.items()
                        if k.startswith("message_passing.")}
        else:
            mp_state = raw_state

        if "W_i.weight" not in mp_state:
            raise ValueError(
                f"Could not find encoder weights ('W_i.weight') in checkpoint "
                f"{encoder_ckpt}. Keys found: {list(raw_state.keys())[:10]}..."
            )

        hidden_size = mp_state["W_i.weight"].shape[0]
        depth = sum(1 for k in mp_state if k.startswith("W_h"))
        mp = cpnn.BondMessagePassing(d_h=hidden_size, depth=depth)
        missing, unexpected = mp.load_state_dict(mp_state, strict=False)
        if missing:
            logger.debug(f"  Encoder load — missing keys: {missing}")

    agg_map = {"mean": cpnn.MeanAggregation(), "sum": cpnn.SumAggregation(),
               "norm": cpnn.NormAggregation()}
    agg = agg_map[arch.get("aggregation", "mean")]

    dropout = dynamics.get("dropout", 0.0)
    heads = {}
    for name in HEAD_KEYS:
        ffn_layers = []
        in_dim = mp.output_dim
        for _ in range(arch.get("ffn_num_layers", 2) - 1):
            ffn_layers += [
                nn_torch.Linear(in_dim, arch.get("ffn_hidden_dim", 300)),
                nn_torch.ReLU(),
                nn_torch.Dropout(dropout),
            ]
            in_dim = arch.get("ffn_hidden_dim", 300)
        ffn_layers.append(nn_torch.Linear(in_dim, 1))
        heads[name] = nn_torch.Sequential(*ffn_layers)

    n_epochs = dynamics["epochs"]
    model = MultiHeadMPNN(
        mp=mp, agg=agg, heads=heads, scalers=scalers,
        lr=dynamics["lr"], warmup_epochs=2, max_epochs=n_epochs,
    )

    freeze_epochs = dynamics.get("freeze_encoder_epochs", 0)
    if freeze_epochs > 0:
        for param in model.message_passing.parameters():
            param.requires_grad = False

    return model, freeze_epochs


#  Single fold runner
def _run_fold(arch, dynamics, train_ds, val_ds, scalers, featurizer, encoder_ckpt):
    model, freeze_epochs = build_model_hpo(arch, dynamics, scalers, encoder_ckpt)

    callbacks = []
    if freeze_epochs > 0:
        class UnfreezeCallback(pl.Callback):
            def __init__(self, epoch): self.epoch = epoch
            def on_train_epoch_start(self, trainer, pl_module):
                if trainer.current_epoch == self.epoch:
                    for p in pl_module.message_passing.parameters():
                        p.requires_grad = True
        callbacks.append(UnfreezeCallback(freeze_epochs))

    train_loader = build_loader(train_ds, featurizer, dynamics["batch_size"], shuffle=True)
    val_loader   = build_loader(val_ds,   featurizer, dynamics["batch_size"], shuffle=False)

    trainer = pl.Trainer(
        max_epochs=dynamics["epochs"],
        accelerator=config.ACCELERATOR,
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)
    val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("nan")))
    return float(val_loss.item() if hasattr(val_loss, "item") else val_loss)


def run_trial_cv(arch, dynamics, folds, featurizer, encoder_ckpt):
    fold_losses = []
    for fold_idx, (train_ds, val_ds, scalers) in enumerate(folds):
        try:
            loss = _run_fold(arch, dynamics, train_ds, val_ds, scalers, featurizer, encoder_ckpt)
            fold_losses.append(loss if not np.isnan(loss) else np.inf)
            logger.info(f"      fold {fold_idx+1}/{len(folds)}: val_loss={loss:.4f}")
        except Exception as e:
            logger.warning(f"      fold {fold_idx+1} failed: {e}")
            fold_losses.append(np.inf)

    valid = [l for l in fold_losses if not np.isinf(l)]
    mean_loss = float(np.mean(valid)) if valid else np.inf
    std_loss  = float(np.std(valid))  if valid else np.inf
    return mean_loss, std_loss, fold_losses


#  Data preparation
def prepare_all_folds(merged_df, head_dfs, frac: float = 1.0):
    """Compute train/val folds (90/10)"""
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    smiles_arr = merged_df[config.SMILES_COL].values

    per_dataset, fold_train_smiles, fold_val_smiles, test_smiles = make_multihead_splits(
        head_dfs, config.SMILES_COL, config.CLUSTER_COL, config.N_FOLDS,
        config.SPLIT_STRATEGY, config.SEED, test_frac=0.10,
    )
    train_folds, val_folds, test_idx = resolve_multihead_indices(
        smiles_arr, fold_train_smiles, fold_val_smiles, test_smiles, config.N_FOLDS)

    logger.info(
        f"  Split: {len(test_idx)} test excluded | "
        f"fold 1 train={len(train_folds[0])}, val={len(val_folds[0])}"
    )
    for fold_i in range(config.N_FOLDS):
        for head_key, d in per_dataset.items():
            if len(d["val_folds"][fold_i]) == 0:
                logger.warning(f"  Fold {fold_i + 1}: '{head_key}' has 0 val samples!")

    folds = []
    for fold in range(config.N_FOLDS):
        train_idx = train_folds[fold]
        val_idx   = val_folds[fold]

        if frac < 1.0:
            rng   = np.random.default_rng(HPO_SEED + fold)
            n_sub = max(1, int(len(train_idx) * frac))
            train_idx = rng.choice(train_idx, size=n_sub, replace=False)

        scalers = fit_scalers(merged_df, train_idx)
        train_targets, val_targets = scale_targets(merged_df, train_idx, val_idx, scalers)

        train_ds = MultiHeadDataset(smiles_arr[train_idx], train_targets, featurizer)
        val_ds   = MultiHeadDataset(smiles_arr[val_idx],   val_targets,   featurizer)
        folds.append((train_ds, val_ds, scalers))

    return folds, featurizer


#  Phase 1: architecture grid
def run_phase1(encoder_ckpt, merged_df, head_dfs, run_id: str):
    logger.info("=" * 60)
    logger.info("PHASE 1: Architecture grid search (5-fold CV, 90% trainval)")
    logger.info(f"  Epochs/trial : {HPO_EPOCHS}  |  Data fraction : {HPO_FRAC}")
    logger.info("=" * 60)

    folds, featurizer = prepare_all_folds(merged_df, head_dfs, frac=HPO_FRAC)
    keys   = list(PHASE1_GRID.keys())
    values = list(PHASE1_GRID.values())
    grid   = [dict(zip(keys, combo)) for combo in product(*values)]
    logger.info(f"  {len(grid)} architecture combinations")

    results = []
    best_loss = np.inf
    best_arch = None

    dynamics_fixed = {
        "lr": config.FINETUNE_LR,
        "batch_size": config.FINETUNE_BATCH,
        "epochs": HPO_EPOCHS,
        "dropout": config.DROPOUT,
        "freeze_encoder_epochs": config.FREEZE_ENCODER_EPOCHS,
    }

    for i, arch in enumerate(grid):
        t0 = time.time()
        logger.info(f"  Trial {i+1}/{len(grid)}: {arch}")
        mean_loss, std_loss, fold_losses = run_trial_cv(
            arch, dynamics_fixed, folds, featurizer, encoder_ckpt
        )
        elapsed = time.time() - t0
        logger.info(f"    mean_val_loss={mean_loss:.4f} ± {std_loss:.4f}  ({elapsed:.1f}s)")
        results.append({
            **arch,
            "mean_val_loss": mean_loss,
            "std_val_loss":  std_loss,
            "fold_losses":   fold_losses,
            "elapsed_s":     elapsed,
        })
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_arch = copy.deepcopy(arch)
            logger.info(f"    *** New best: {best_loss:.4f}")

    results.sort(key=lambda x: x["mean_val_loss"])
    out = {
        "run_date":           _RUN_DATE,
        "run_id":             run_id,
        "n_folds":            config.N_FOLDS,
        "hpo_epochs":         HPO_EPOCHS,
        "hpo_frac":           HPO_FRAC,
        "encoder_ckpt":       str(encoder_ckpt),
        "dynamics_fixed":     dynamics_fixed,
        "best_arch":          best_arch,
        "best_mean_val_loss": best_loss,
        "all_results":        results,
    }
    out_path = OUTPUT_DIR / f"phase1_grid_results_{run_id}_{config.SPLIT_STRATEGY}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Phase 1 complete. Best: {best_arch}  loss={best_loss:.4f}")
    logger.info(f"Saved: {out_path}")
    return best_arch, best_loss


#  Phase 2: curated + extended trial list
def run_phase2(best_arch: dict, encoder_ckpt, merged_df, head_dfs, run_id: str):
    logger.info("=" * 60)
    logger.info(f"PHASE 2: Hyperparameter search ({len(PHASE2_TRIALS)} trials)")
    logger.info(f"  Best arch: {best_arch}")
    logger.info(f"  Split: 10% test excluded, 5-fold CV on 90%")
    logger.info("=" * 60)

    folds, featurizer = prepare_all_folds(merged_df, head_dfs, frac=HPO_FRAC)

    results = []
    best_loss     = np.inf
    best_dynamics = None

    for i, dynamics_cfg in enumerate(PHASE2_TRIALS):
        label    = dynamics_cfg.get("_label", "")
        dynamics = {k: v for k, v in dynamics_cfg.items() if k != "_label"}
        arch     = {**best_arch, "dropout": dynamics.get("dropout", 0.0)}

        t0 = time.time()
        logger.info(f"  Trial {i+1}/{len(PHASE2_TRIALS)}"
                    f"{'  [' + label + ']' if label else ''}: {dynamics}")

        mean_loss, std_loss, fold_losses = run_trial_cv(
            arch, dynamics, folds, featurizer, encoder_ckpt
        )
        elapsed = time.time() - t0
        logger.info(f"    mean_val_loss={mean_loss:.4f} ± {std_loss:.4f}  ({elapsed:.1f}s)")

        results.append({
            **dynamics,
            "_label":        label,
            "mean_val_loss": mean_loss,
            "std_val_loss":  std_loss,
            "fold_losses":   fold_losses,
            "elapsed_s":     elapsed,
        })

        if mean_loss < best_loss:
            best_loss     = mean_loss
            best_dynamics = copy.deepcopy(dynamics)
            logger.info(f"    *** New best: {best_loss:.4f}")

    results.sort(key=lambda x: x["mean_val_loss"])

    out = {
        "run_date":           _RUN_DATE,
        "run_id":             run_id,
        "n_folds":            config.N_FOLDS,
        "best_arch":          best_arch,
        "n_trials":           len(PHASE2_TRIALS),
        "encoder_ckpt":       str(encoder_ckpt),
        "best_dynamics":      best_dynamics,
        "best_mean_val_loss": best_loss,
        "all_results":        results,
    }
    out_path = OUTPUT_DIR / f"phase2_curated_results_{run_id}_{config.SPLIT_STRATEGY}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Phase 2 complete. Best dynamics: {best_dynamics}  loss={best_loss:.4f}")
    logger.info(f"Saved: {out_path}")
    return best_dynamics, best_loss


#  Write best config
def write_best_config(best_arch: dict, best_dynamics: dict, run_id: str):
    best = {**best_arch, **best_dynamics}
    out_path = OUTPUT_DIR / f"best_config_{run_id}_{config.SPLIT_STRATEGY}.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    logger.info("=" * 60)
    logger.info("BEST COMBINED CONFIG")
    logger.info("=" * 60)
    for k, v in best.items():
        logger.info(f"  {k:30s} = {v}")

    logger.info(f"  MPNN_DEPTH            = {best_arch['depth']}")
    logger.info(f"  MPNN_D_H              = {best_arch['d_h']}")
    logger.info(f"  FFN_NUM_LAYERS        = {best_arch['ffn_num_layers']}")
    logger.info(f"  FFN_HIDDEN_DIM        = {best_arch['ffn_hidden_dim']}")
    logger.info(f"  AGGREGATION           = '{best_arch['aggregation']}'")
    logger.info(f"  FINETUNE_LR           = {best_dynamics['lr']}")
    logger.info(f"  FINETUNE_BATCH        = {best_dynamics['batch_size']}")
    logger.info(f"  FINETUNE_EPOCHS       = {best_dynamics['epochs']}")
    logger.info(f"  DROPOUT               = {best_dynamics['dropout']}")
    logger.info(f"  FREEZE_ENCODER_EPOCHS = {best_dynamics['freeze_encoder_epochs']}")
    logger.info(f"\nSaved: {out_path}")
    return best

#  Argument parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True,
                        choices=list(config.MULTIHEAD_RUN_CONFIG.keys()))
    parser.add_argument("--phase", type=str, default="both",
                        choices=["1", "2", "both"])
    parser.add_argument("--phase1_result", type=str, default=None,
                        help="Path to existing Phase 1 JSON — skip Phase 1")
    parser.add_argument("--pretrain_ckpt", type=str, default=None,
                        help="Path to pretrained encoder checkpoint (.pt)")
    return parser.parse_args()


#  Main
def main():
    args = parse_args()

    pl.seed_everything(HPO_SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_cfg = config.MULTIHEAD_RUN_CONFIG[args.run_id]

    if args.pretrain_ckpt:
        encoder_ckpt = Path(args.pretrain_ckpt)
    elif config.PIPELINE_MODE == "chemeleon":
        encoder_ckpt = config.CHEMELEON_PATH
    else:
        encoder_ckpt = None

    logger.info(f"Run ID:        {args.run_id}  ({run_cfg['run_name']})")
    logger.info(f"Pipeline mode: {config.PIPELINE_MODE}")
    logger.info(f"Encoder:       {encoder_ckpt}")
    logger.info(f"Accelerator:   {config.ACCELERATOR}")
    logger.info(f"HPO output:    {OUTPUT_DIR.resolve()}")
    logger.info(f"Phase 2:       {len(PHASE2_TRIALS)} curated trials")

    logger.info("Loading datasets")
    merged_df, _, head_dfs = load_and_merge_datasets(run_cfg)
    logger.info(f"Merged dataset: {len(merged_df)} molecules")

    # Phase 1 (or load an existing/manual result)
    if args.phase1_result:
        with open(args.phase1_result) as f:
            p1 = json.load(f)
        best_arch = p1["best_arch"]
    elif args.phase in ("1", "both"):
        best_arch, _ = run_phase1(encoder_ckpt, merged_df, head_dfs, args.run_id)
    else:
        raise ValueError("--phase=2 requires --phase1_result")

    # Phase 2
    if args.phase in ("2", "both"):
        best_dynamics, _ = run_phase2(best_arch, encoder_ckpt, merged_df, head_dfs, args.run_id)
        write_best_config(best_arch, best_dynamics, args.run_id)
    else:
        logger.info(f"Phase 2 skipped. Best arch: {best_arch}")

if __name__ == "__main__":
    main()
