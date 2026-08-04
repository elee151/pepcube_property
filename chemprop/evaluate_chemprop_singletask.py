"""
evaluate_chemprop_singletask.py
Cross-validation + test-set evaluation of a single-task finetuned Chemprop model.
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from chemprop import data, featurizers, nn
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
from pepcube_property.results import append_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHEMELEON_SUBSET_LABEL = "chemeleon"
CHEMPROP_SUBSET_LABEL = "chemprop"
_RUN_DATE = datetime.now().strftime("%Y-%m-%d")
_JOB_ID   = os.environ.get("PBS_JOBID", "local").split(".")[0]

CLUSTER_COL = config.CLUSTER_COL


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate single-task fine-tuned Chemprop model")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument("--subset_name", type=str, default=None,
                        help="Pretraining subset (required for PIPELINE_MODE=pretrained)")
    return parser.parse_args()


def resolve_subset_label(args):
    if config.PIPELINE_MODE == "chemprop":
        return CHEMPROP_SUBSET_LABEL
    elif config.PIPELINE_MODE == "chemeleon":
        return CHEMELEON_SUBSET_LABEL
    else:
        if args.subset_name is None:
            raise ValueError("--subset_name is required when PIPELINE_MODE=pretrained")
        return args.subset_name


def compute_metrics(y_true, y_pred, metric_names):
    results = {}
    if "rmse"     in metric_names:
        results["rmse"]      = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if "mae"      in metric_names:
        results["mae"]       = float(mean_absolute_error(y_true, y_pred))
    if "r2"       in metric_names:
        results["r2"]        = float(r2_score(y_true, y_pred))
    if "pearson"  in metric_names:
        r, p = pearsonr(y_true, y_pred)
        results["pearson_r"] = float(r)
        results["pearson_p"] = float(p)
    if "spearman" in metric_names:
        r, p = spearmanr(y_true, y_pred)
        results["spearman_r"] = float(r)
        results["spearman_p"] = float(p)
    return results


def load_scaler(scaler_path) -> StandardScaler:
    saved  = torch.load(scaler_path, map_location="cpu", weights_only=False)
    scaler = StandardScaler()
    scaler.mean_  = saved["mean"]
    scaler.scale_ = saved["scale"]
    return scaler


def load_model(checkpoint_path, scaler: StandardScaler) -> MPNN:
    state       = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hidden_size = state["message_passing.W_i.weight"].shape[0]
    mp  = nn.BondMessagePassing(d_h=hidden_size)
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn  = nn.RegressionFFN(n_tasks=1, output_transform=output_transform, input_dim=mp.output_dim)
    model = MPNN(message_passing=mp, agg=agg, predictor=ffn)
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model, eval_dataset) -> np.ndarray:
    loader = data.build_dataloader(eval_dataset, batch_size=64, shuffle=False, drop_last=False)
    preds  = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.bmg, batch.V_d, batch.X_d)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds).flatten()


def main():
    args = parse_args()

    ds_cfg       = config.DATASET_CONFIG[args.dataset_name]
    label_col    = ds_cfg["label_col"]
    metric_names = ds_cfg["metrics"]
    data_path    = ds_cfg["data_file"]
    subset_label = resolve_subset_label(args)
    ckpt_dir     = config.finetune_dir(subset_label, args.dataset_name)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== EVALUATE (single-task) — CV + test ===")
    logger.info(f"Date/Job:        {_RUN_DATE} / {_JOB_ID}")
    logger.info(f"Pipeline mode:   {config.PIPELINE_MODE}")
    logger.info(f"Subset label:    {subset_label}")
    logger.info(f"Dataset:         {args.dataset_name} | Label: {label_col}")
    logger.info(f"Cluster col:     {CLUSTER_COL}")
    logger.info(f"Split strategy:  {config.SPLIT_STRATEGY}")
    logger.info(f"Checkpoints:     {ckpt_dir}")

    df = pd.read_csv(data_path)
    for col in [config.SMILES_COL, label_col, CLUSTER_COL]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {data_path}")
    df = df[[config.SMILES_COL, label_col, CLUSTER_COL]].dropna().reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows | {df[CLUSTER_COL].nunique()} unique groups")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    #  Load split indices if saved, else recompute
    split_path = ckpt_dir / "split_indices.json"
    split_meta = {}
    if split_path.exists():
        logger.info(f"Loading split indices from {split_path}")
        with open(split_path) as f:
            split_meta = json.load(f)
        train_folds = [np.array(f, dtype=int) for f in split_meta["train_folds"]]
        val_folds   = [np.array(f, dtype=int) for f in split_meta["val_folds"]]
        test_idx    = np.array(split_meta["test_idx"], dtype=int)
        logger.info(f"Test set: {len(test_idx)} samples")
    else:
        logger.warning("split_indices.json not found — recomputing splits"
                       "on this dataset's own groups.")
        groups = df[CLUSTER_COL].values

        train_folds, val_folds, test_idx = make_train_val_test_indices(
            groups, config.N_FOLDS, config.SPLIT_STRATEGY, config.SEED, test_frac=0.10)
        logger.info(f"Test set: {len(test_idx)} samples (recomputed)")

    all_datapoints = [
        data.MoleculeDatapoint.from_smi(smi, [y])
        for smi, y in zip(df[config.SMILES_COL], df[label_col])
    ]

    #  CV folds 
    fold_results = []
    for fold in range(config.N_FOLDS):
        logger.info(f"=== Fold {fold + 1}/{config.N_FOLDS} ===")
        ckpt_path   = ckpt_dir / f"fold_{fold + 1}" / "finetuned_model.pt"
        scaler_path = ckpt_dir / f"fold_{fold + 1}" / "scaler.pt"

        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found, skipping: {ckpt_path}")
            continue
        if not scaler_path.exists():
            logger.warning(f"Scaler not found, skipping: {scaler_path}")
            continue

        scaler  = load_scaler(scaler_path)
        val_idx = val_folds[fold]
        val_dps = [all_datapoints[i] for i in val_idx]
        val_dataset = data.MoleculeDataset(val_dps, featurizer=featurizer)

        model  = load_model(ckpt_path, scaler)
        y_pred = run_inference(model, val_dataset)
        y_true = df[label_col].iloc[val_idx].values

        metrics = compute_metrics(y_true, y_pred, metric_names)
        metrics["fold"]          = fold + 1
        fold_results.append(metrics)
        logger.info(f"  Fold {fold + 1}: {metrics}")

    if not fold_results:
        logger.error("No fold results — check that fine-tuning ran successfully.")
        return

    scalar_keys = ["rmse", "mae", "r2", "pearson_r", "spearman_r"]
    agg_metrics = {}
    for key in scalar_keys:
        vals = [r[key] for r in fold_results if key in r]
        if vals:
            agg_metrics[f"mean_{key}"] = float(np.mean(vals))
            agg_metrics[f"std_{key}"]  = float(np.std(vals))
    logger.info(f"CV aggregated: {agg_metrics}")

    #  Test set 
    test_metrics = {}
    final_ckpt   = ckpt_dir / "final" / "finetuned_model.pt"
    final_scaler = ckpt_dir / "final" / "scaler.pt"
    if not final_ckpt.exists():
        logger.warning(f"Final model not found: {final_ckpt} — skipping test evaluation.")

    else:
        scaler       = load_scaler(final_scaler)
        test_dps     = [all_datapoints[i] for i in test_idx]
        test_dataset = data.MoleculeDataset(test_dps, featurizer=featurizer)
        model        = load_model(final_ckpt, scaler)
        y_pred       = run_inference(model, test_dataset)
        y_true       = df[label_col].iloc[test_idx].values
        test_metrics = compute_metrics(y_true, y_pred, metric_names)
        logger.info(f"TEST: {test_metrics}")

    #  Save JSON
    result_record = {
        "run_timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_date":        _RUN_DATE,
        "job_id":          _JOB_ID,
        "model_arch":      "single_task",
        "pipeline_mode":   config.PIPELINE_MODE,
        "subset_label":    subset_label,
        "dataset_name":    args.dataset_name,
        "cluster_col":     CLUSTER_COL,
        "split_strategy":  config.SPLIT_STRATEGY,
        "n_folds":         config.N_FOLDS,
        "finetune_epochs": config.FINETUNE_EPOCHS,
        "finetune_lr":     config.FINETUNE_LR,
        "seed":            config.SEED,
        "fold_results":    fold_results,
        "test_metrics":    test_metrics,
        **agg_metrics,
    }

    json_path = (
        config.RESULTS_DIR
        / f"{subset_label}_{args.dataset_name}_{CLUSTER_COL}_cv_test_metrics.json"
    )
    with open(json_path, "w") as f:
        json.dump(result_record, f, indent=2)
    logger.info(f"JSON saved: {json_path}")

    #  Append to chemprop_results.csv (trimmed schema) 
    # Single-task runs don't have a multi-head "run_name"/"head" split — the
    # dataset itself IS the head, so both are set to dataset_name for schema
    # consistency with the multi-head ledger.
    base_row = {
        "run_timestamp":  result_record["run_timestamp"],
        "run_date":       _RUN_DATE,
        "job_id":         _JOB_ID,
        "model_arch":     "single_task",
        "run_name":       args.dataset_name,
        "head":           args.dataset_name,
        "dataset_name":   args.dataset_name,
        "pipeline_mode":  config.PIPELINE_MODE,
        "subset_label":   subset_label,
        "split_strategy": config.SPLIT_STRATEGY,
    }

    cv_row = {**base_row, **agg_metrics}
    rows = [cv_row]

    if test_metrics:
        rows.append({
            **base_row,
            "rmse":       test_metrics.get("rmse"),
            "mae":        test_metrics.get("mae"),
            "r2":         test_metrics.get("r2"),
            "pearson_r":  test_metrics.get("pearson_r"),
            "spearman_r": test_metrics.get("spearman_r"),
        })

    append_results(rows, config.CHEMPROP_RESULTS_CSV)


if __name__ == "__main__":
    main()
