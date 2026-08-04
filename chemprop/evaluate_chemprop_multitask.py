"""
evaluate_chemprop_multitask.py
Cross-validation + test-set evaluation of the multihead Chemprop model.

"""
import argparse
import json
import logging
import os
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from chemprop import featurizers, nn as cpnn
from chemprop.data.collate import BatchMolGraph
from rdkit import Chem
from torch.utils.data import DataLoader

os.environ.setdefault("MODEL_ARCH", "multi_head")
import sys as _sys
_p = Path(__file__).resolve().parent
while _p.name != "pepcube_property" and _p != _p.parent:
    _p = _p.parent
if _p.name == "pepcube_property" and str(_p.parent) not in _sys.path:
    _sys.path.insert(0, str(_p.parent))

import pepcube_property.config as config
from pepcube_property.utils import *
from pepcube_property.results import append_results

# Import shared helpers from finetune_chemprop_multitask.py
import importlib, sys
_mh_path = Path(__file__).parent / "finetune_chemprop_multitask.py"
_spec     = importlib.util.spec_from_file_location("finetune_multihead", _mh_path)
_mh_mod   = importlib.util.module_from_spec(_spec)
sys.modules["finetune_multihead"] = _mh_mod
_spec.loader.exec_module(_mh_mod)

MultiHeadMPNN           = _mh_mod.MultiHeadMPNN
MultiHeadDataset        = _mh_mod.MultiHeadDataset
collate_multihead       = _mh_mod.collate_multihead
load_and_merge_datasets = _mh_mod.load_and_merge_datasets
HEAD_KEYS               = _mh_mod.HEAD_KEYS
build_loader            = _mh_mod.build_loader
load_encoder            = _mh_mod.load_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHEMELEON_SUBSET_LABEL = "chemeleon"
CHEMPROP_SUBSET_LABEL = "chemprop"
_RUN_DATE = datetime.now().strftime("%Y-%m-%d")
_JOB_ID   = os.environ.get("PBS_JOBID", "local").split(".")[0]

CLUSTER_COL = config.CLUSTER_COL


#  Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-head Chemprop model — CV + test metrics"
    )
    parser.add_argument("--run_id", type=str, required=True,
                        choices=list(config.MULTIHEAD_RUN_CONFIG.keys()))
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--subset_name", type=str, default=None)
    return parser.parse_args()


def resolve_subset_label(args):
    if config.PIPELINE_MODE == "chemprop":
        return CHEMPROP_SUBSET_LABEL
    elif config.PIPELINE_MODE == "chemeleon":
        return CHEMELEON_SUBSET_LABEL
    else:
        if args.subset_name is None:
            raise ValueError("--subset_name required when PIPELINE_MODE=pretrained")
        return args.subset_name


#  Model loading
def load_scalers(scalers_path) -> dict:
    saved = torch.load(scalers_path, map_location="cpu", weights_only=False)
    scalers = {}
    for name, d in saved.items():
        sc = StandardScaler()
        sc.mean_  = d["mean"]
        sc.scale_ = d["scale"]
        scalers[name] = sc
    return scalers


def load_multihead_model(checkpoint_path, scalers: dict, depth: int):
    state       = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hidden_size = state["message_passing.W_i.weight"].shape[0]

    mp  = cpnn.BondMessagePassing(d_h=hidden_size, depth=depth)
    agg = cpnn.MeanAggregation()
    heads = {
        name: cpnn.RegressionFFN(n_tasks=1, input_dim=mp.output_dim)
        for name in HEAD_KEYS
    }
    model = MultiHeadMPNN(mp=mp, agg=agg, heads=heads, scalers=scalers)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


#  Inference
def run_inference_multihead(model, loader):
    """Return {head_name: np.ndarray} predictions in original label space."""
    all_preds = {name: [] for name in HEAD_KEYS}
    with torch.no_grad():
        for bmg, _ in loader:
            preds = model.predict(bmg)
            for name in HEAD_KEYS:
                all_preds[name].append(preds[name].squeeze(-1).cpu().numpy())
    return {name: np.concatenate(vals) for name, vals in all_preds.items()}


#  Metrics
METRIC_NAMES = ["rmse", "mae", "r2", "pearson", "spearman"]


def compute_metrics(y_true, y_pred):
    mask   = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return {}

    pr, _ = pearsonr(y_true, y_pred)
    sp, _ = spearmanr(y_true, y_pred)
    return {
        "rmse":        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":         float(mean_absolute_error(y_true, y_pred)),
        "r2":          float(r2_score(y_true, y_pred)),
        "pearson_r":   float(pr),
        "spearman_r":  float(sp),
    }


def aggregate_fold_metrics(fold_results_list):
    """Average per-fold dicts over scalar metric keys."""
    agg = {}
    scalar_keys = ["rmse", "mae", "r2", "pearson_r", "spearman_r"]
    for key in scalar_keys:
        vals = [r[key] for r in fold_results_list if key in r]
        if vals:
            agg[f"mean_{key}"] = float(np.mean(vals))
            agg[f"std_{key}"]  = float(np.std(vals))
    return agg


#  Main
def resolve_actual_depth(actual_encoder, actual_pretrain_ckpt):
    if actual_encoder == "chemprop_random_init":
        return config.MPNN_DEPTH
    mp = load_encoder(actual_pretrain_ckpt)
    return mp.depth


def main():
    args = parse_args()

    run_cfg      = config.MULTIHEAD_RUN_CONFIG[args.run_id]
    run_name     = run_cfg["run_name"]
    subset_label = resolve_subset_label(args)
    ckpt_dir     = config.finetune_dir(subset_label, run_name)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== EVALUATE (multi-head) — CV + test ===")
    logger.info(f"Date/Job:      {_RUN_DATE} / {_JOB_ID}")
    logger.info(f"Run ID:        {args.run_id}  ({run_name})")
    logger.info(f"Pipeline mode: {config.PIPELINE_MODE}")
    logger.info(f"Subset label:  {subset_label}")
    logger.info(f"Checkpoints:   {ckpt_dir}")

    if config.PIPELINE_MODE == "chemprop":
        actual_encoder       = "chemprop_random_init"
        actual_pretrain_ckpt = "n/a (random init)"
    elif config.PIPELINE_MODE == "chemeleon":
        actual_encoder       = "chemeleon"
        actual_pretrain_ckpt = str(config.CHEMELEON_PATH)
    else:  # "pretrained"
        actual_encoder       = "pretrained"
        actual_pretrain_ckpt = args.pretrain_ckpt or str(config.pretrain_checkpoint(args.subset_name))

    logger.info(f"Encoder:       {actual_encoder}  ({actual_pretrain_ckpt})")

    actual_depth = resolve_actual_depth(actual_encoder, actual_pretrain_ckpt)
    logger.info(f"Actual depth:  {actual_depth} (config.MPNN_DEPTH={config.MPNN_DEPTH})")

    encoder_info_path = ckpt_dir / "encoder_info.json"

    if encoder_info_path.exists():
        with open(encoder_info_path) as f:
            encoder_info = json.load(f)
    else:
        encoder_info = {
            "split_strategy": config.SPLIT_STRATEGY,
            "run_id":         args.run_id,
        }

    merged_df, label_map, head_dfs = load_and_merge_datasets(run_cfg)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    groups     = merged_df[CLUSTER_COL].values
    smiles_arr = merged_df[config.SMILES_COL].values

    #  Load saved split indices
    split_path = ckpt_dir / "split_indices.json"
    if split_path.exists():
        logger.info(f"Loading split indices from {split_path}")
        with open(split_path) as f:
            split_meta = json.load(f)
        train_folds = [np.array(f, dtype=int) for f in split_meta["train_folds"]]
        val_folds   = [np.array(f, dtype=int) for f in split_meta["val_folds"]]
        test_idx    = np.array(split_meta["test_idx"], dtype=int)
        assert split_meta["n_total"] == len(merged_df), (
            f"Dataset size mismatch: split file has {split_meta['n_total']}, "
            f"current merge has {len(merged_df)}"
        )

        split_meta = {}
        per_dataset, fold_train_smiles, fold_val_smiles, test_smiles = make_multihead_splits(
            head_dfs, config.SMILES_COL, CLUSTER_COL, config.N_FOLDS,
            config.SPLIT_STRATEGY, config.SEED, test_frac=0.10,
        )
        train_folds, val_folds, test_idx = resolve_multihead_indices(
            smiles_arr, fold_train_smiles, fold_val_smiles, test_smiles, config.N_FOLDS)

    #  CV evaluation
    fold_results = {name: [] for name in HEAD_KEYS}

    for fold in range(config.N_FOLDS):
        logger.info(f"=== CV Fold {fold + 1}/{config.N_FOLDS} ===")
        ckpt_path    = ckpt_dir / f"fold_{fold + 1}" / "finetuned_model.pt"
        scalers_path = ckpt_dir / f"fold_{fold + 1}" / "scalers.pt"

        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found, skipping: {ckpt_path}")
            continue
        if not scalers_path.exists():
            logger.warning(f"Scalers not found, skipping: {scalers_path}")
            continue

        scalers  = load_scalers(scalers_path)
        val_idx  = val_folds[fold]
        val_smiles = smiles_arr[val_idx]

        dummy_targets = {name: np.full(len(val_idx), np.nan) for name in HEAD_KEYS}
        val_ds     = MultiHeadDataset(val_smiles, dummy_targets, featurizer)
        val_loader = build_loader(val_ds, featurizer, batch_size=64, shuffle=False)

        model = load_multihead_model(ckpt_path, scalers, depth=actual_depth)
        preds = run_inference_multihead(model, val_loader)

        for name in HEAD_KEYS:
            col    = f"label_{name}"
            y_true = merged_df[col].iloc[val_idx].values.astype(float)
            y_pred = preds[name]
            m      = compute_metrics(y_true, y_pred)
            m["fold"] = fold + 1
            fold_results[name].append(m)
            logger.info(f"  [{name}] Fold {fold+1}: r2={m.get('r2', float('nan')):.3f}  "
                        f"pearson={m.get('pearson_r', float('nan')):.3f}  "
                        f"rmse={m.get('rmse', float('nan')):.4f}")

    # Aggregate CV
    cv_agg = {name: aggregate_fold_metrics(fold_results[name]) for name in HEAD_KEYS}
    logger.info("=== CV AGGREGATED ===")
    for name, agg in cv_agg.items():
        logger.info(f"  [{name}] mean_r2={agg.get('mean_r2', float('nan')):.3f} ± "
                    f"{agg.get('std_r2', float('nan')):.3f}  "
                    f"mean_pearson={agg.get('mean_pearson_r', float('nan')):.3f}")

    #  Test-set evaluation (final model only)
    logger.info("=== TEST SET EVALUATION ===")
    final_ckpt_path    = ckpt_dir / "final" / "finetuned_model.pt"
    final_scalers_path = ckpt_dir / "final" / "scalers.pt"
    test_metrics = {}

    if not final_ckpt_path.exists():
        logger.warning(f"Final model checkpoint not found: {final_ckpt_path}")
    elif len(test_idx) == 0:
        logger.info("No test set found, skipping test evaluation.")
    else:
        final_scalers = load_scalers(final_scalers_path)
        test_smiles   = smiles_arr[test_idx]

        dummy_test_targets = {name: np.full(len(test_idx), np.nan) for name in HEAD_KEYS}
        test_ds     = MultiHeadDataset(test_smiles, dummy_test_targets, featurizer)
        test_loader = build_loader(test_ds, featurizer, batch_size=64, shuffle=False)

        final_model  = load_multihead_model(final_ckpt_path, final_scalers, depth=actual_depth)
        test_preds   = run_inference_multihead(final_model, test_loader)

        for name in HEAD_KEYS:
            col    = f"label_{name}"
            y_true = merged_df[col].iloc[test_idx].values.astype(float)
            y_pred = test_preds[name]
            m      = compute_metrics(y_true, y_pred)
            test_metrics[name] = m
            logger.info(f"  [{name}] TEST: r2={m.get('r2', float('nan')):.3f}  "
                        f"pearson={m.get('pearson_r', float('nan')):.3f}  "
                        f"rmse={m.get('rmse', float('nan')):.4f}  "
                        )

    #  Save full result record
    result_record = {
        "run_timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_date":              _RUN_DATE,
        "job_id":                _JOB_ID,
        "model":                 "chemprop_multihead",
        "model_arch":            "multi_head",
        "run_id":                args.run_id,
        "run_name":              run_name,
        "pipeline_mode":         config.PIPELINE_MODE,
        "subset_label":          subset_label,
        "cluster_col":           CLUSTER_COL,
        "split_strategy":        encoder_info.get("split_strategy", config.SPLIT_STRATEGY),
        "encoder":               actual_encoder,
        "pretrain_ckpt":         actual_pretrain_ckpt,
        "actual_depth":          actual_depth,
        "head_datasets":         {k: run_cfg[k] for k in HEAD_KEYS},
        "n_folds":               config.N_FOLDS,
        "finetune_epochs":       config.FINETUNE_EPOCHS,
        "finetune_lr":           config.FINETUNE_LR,
        "seed":                  config.SEED,
        "per_head_fold_results": fold_results,
        "cv_agg":                cv_agg,
        "test_metrics":          test_metrics,
    }

    json_path = (
        config.RESULTS_DIR
        / f"chemprop_{subset_label}_{run_name}_cv_test_metrics.json"
    )
    with open(json_path, "w") as f:
        json.dump(result_record, f, indent=2)

    #  Append subset of rows to chemprop_results.csv
    rows = []
    for name in HEAD_KEYS:
        cv_row = {
            "run_timestamp":  result_record["run_timestamp"],
            "run_date":       _RUN_DATE,
            "job_id":         _JOB_ID,
            "model_arch":     "multi_head",
            "run_name":       run_name,
            "head":           name,
            "dataset_name":   run_cfg[name],
            "pipeline_mode":  config.PIPELINE_MODE,
            "subset_label":   subset_label,
            "split_strategy": encoder_info.get("split_strategy", config.SPLIT_STRATEGY),
            **cv_agg[name],
        }
        rows.append(cv_row)

        if name in test_metrics and test_metrics[name]:
            tm = test_metrics[name]
            rows.append({
                "run_timestamp":  result_record["run_timestamp"],
                "run_date":       _RUN_DATE,
                "job_id":         _JOB_ID,
                "model_arch":     "multi_head",
                "run_name":       run_name,
                "head":           name,
                "dataset_name":   run_cfg[name],
                "pipeline_mode":  config.PIPELINE_MODE,
                "subset_label":   subset_label,
                "split_strategy": encoder_info.get("split_strategy", config.SPLIT_STRATEGY),
                "rmse":           tm.get("rmse"),
                "mae":            tm.get("mae"),
                "r2":             tm.get("r2"),
                "pearson_r":      tm.get("pearson_r"),
                "spearman_r":     tm.get("spearman_r"),
            })

    append_results(rows, config.CHEMPROP_RESULTS_CSV)

if __name__ == "__main__":
    main()
