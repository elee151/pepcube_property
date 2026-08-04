"""
baseline_ml.py
Baseline ML models (RandomForest, XGBoost, CatBoost, SVM) on Morgan fingerprints.
Hyperparameter tuning via RandomizedSearchCV

Usage:
    python baseline_ml.py --dataset_name hemolysis --model random_forest
    python baseline_ml.py --dataset_name camsol_20k --model xgboost
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import sys as _sys
_p = Path(__file__).resolve().parent
while _p.name != "pepcube_property" and _p != _p.parent:
    _p = _p.parent
if _p.name == "pepcube_property" and str(_p.parent) not in _sys.path:
    _sys.path.insert(0, str(_p.parent))

import pepcube_property.config as config
from pepcube_property.utils import *
from pepcube_property.results import append_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

#  Paths
DATA_DIR    = config.DATA_DIR
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(config.BASE_DIR / "runs" / "baseline_ml")))
SMILES_COL  = config.SMILES_COL
CLUSTER_COL = config.CLUSTER_COL
N_FOLDS     = config.N_FOLDS
N_ITER      = int(os.environ.get("N_ITER", 30))   # RandomizedSearchCV iterations
SEED        = config.SEED
_RUN_DATE      = datetime.now().strftime("%Y-%m-%d")
_RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
_JOB_ID        = os.environ.get("PBS_JOBID", "local").split(".")[0]

#  Dataset registry 
DF_DIR = {
    "synthesize_chemprop.csv": "L : (L+M) Ratio (%)",
    "chemberta_hemolysis_chemprop_550.csv": "revised_50hemo",
    "even_chemberta_noncanonical_camsol_chemprop_10k.csv": "camsol_score",
    "test_syn.csv": "L : (L+M) Ratio (%)",
    "test_hemo.csv": "revised_50hemo",
    "test_camsol.csv": "camsol_score",
}

DATASET_NAME_MAP = {
    "synthesizability": "synthesize_chemprop.csv",
    "chemberta_hemolysis_chemprop_550": "chemberta_hemolysis_chemprop_550.csv",
    "even_chemberta_noncanonical_camsol_chemprop_10k": "even_chemberta_noncanonical_camsol_chemprop_10k.csv",
    "test_syn": "test_syn.csv",
    "test_hemo": "test_hemo.csv",
    "test_camsol": "test_camsol.csv",
}

MODEL_NAMES = ["random_forest", "xgboost", "catboost", "svm"]
#  Hyperparameter search spaces
PARAM_GRIDS = {
    "random_forest": {
        "n_estimators":     [200, 500, 1000],
        "max_depth":        [None, 10, 20, 30],
        "min_samples_split":[2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features":     ["sqrt", "log2", 0.3],
    },
    "xgboost": {
        "n_estimators":     [300, 500, 1000],
        "max_depth":        [3, 5, 6, 8],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma":            [0, 0.1, 0.5],
    },
    "catboost": {
        "iterations":   [500, 1000, 1500],
        "depth":        [4, 6, 8],
        "learning_rate":[0.01, 0.05, 0.1],
        "l2_leaf_reg":  [1, 3, 5, 7],
    },
    "svm": {
        "C":       [0.1, 1, 10, 100],
        "gamma":   ["scale", "auto", 0.001, 0.01, 0.1],
        "epsilon": [0.01, 0.1, 0.5],
    },
}

#  Fingerprint
fingergen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def smiles_to_fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return fingergen.GetFingerprintAsNumPy(mol)

#  Models
def get_base_model(model_name: str, use_gpu: bool = False):
    n_jobs = int(os.environ.get("NUM_WORKERS", 16))

    if model_name == "random_forest":
        if use_gpu:
            try:
                from cuml.ensemble import RandomForestRegressor as cuRF
                return cuRF(random_state=SEED)
            except ImportError:
                logger.warning("cuML not installed — random_forest falling back to CPU")
        return RandomForestRegressor(random_state=SEED, n_jobs=n_jobs)

    elif model_name == "xgboost":
        if use_gpu:
            return XGBRegressor(random_state=SEED, tree_method="hist",
                                device="cuda", verbosity=0)
        return XGBRegressor(random_state=SEED, n_jobs=n_jobs, verbosity=0)

    elif model_name == "catboost":
        if use_gpu:
            return CatBoostRegressor(random_seed=SEED, task_type="GPU",
                                     devices="0", verbose=0)
        return CatBoostRegressor(random_seed=SEED, thread_count=n_jobs, verbose=0)

    elif model_name == "svm":
        if use_gpu:
            try:
                from cuml.svm import SVR as cuSVR
                return cuSVR(kernel="rbf")
            except ImportError:
                logger.warning("cuML not installed — svm falling back to CPU")
        return SVR(kernel="rbf")

    else:
        raise ValueError(f"Unknown model: {model_name}")

#  Metrics
def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "pearson_r": float(pr), "spearman_r": float(sr),
    }

#  Multi-metric scorers used for the single tuning pass 
CV_SCORERS = {
    "rmse":       make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp))),
    "mae":        make_scorer(mean_absolute_error, greater_is_better=False),
    "r2":         make_scorer(r2_score),
    "pearson_r":  make_scorer(lambda yt, yp: pearsonr(yt, yp)[0]),
    "spearman_r": make_scorer(lambda yt, yp: spearmanr(yt, yp)[0]),
}

# One RandomizedSearchCV run with cv=splits, pick best hyperparameter set for the whole dataset
def tune_and_eval(X_fit, y_fit, splits, model_name: str, use_gpu: bool = False):
    base = get_base_model(model_name, use_gpu=use_gpu)
    grid = PARAM_GRIDS[model_name]

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=grid,
        n_iter=N_ITER,
        scoring=CV_SCORERS,
        cv=splits,
        refit="rmse",
        random_state=SEED,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_fit, y_fit)

    best_idx    = search.best_index_
    best_params = search.best_params_
    cvres       = search.cv_results_

    fold_metrics = []
    for fold_idx in range(len(splits)):
        fold_metrics.append({
            "fold":        fold_idx + 1,
            "rmse":        float(-cvres[f"split{fold_idx}_test_rmse"][best_idx]),
            "mae":         float(-cvres[f"split{fold_idx}_test_mae"][best_idx]),
            "r2":          float(cvres[f"split{fold_idx}_test_r2"][best_idx]),
            "pearson_r":   float(cvres[f"split{fold_idx}_test_pearson_r"][best_idx]),
            "spearman_r":  float(cvres[f"split{fold_idx}_test_spearman_r"][best_idx]),
        })

    return search.best_estimator_, best_params, fold_metrics

#  Core CV loop 
def run_cv(X, y, groups, model_name: str, dataset_name: str, out_dir: Path,
           train_folds, val_folds, test_idx,
           split_strategy: str = "group",
           use_gpu: bool = False):

    if len(test_idx) == 0:
        X_fit, y_fit, g_fit = X, y, groups
        splits = list(zip(train_folds, val_folds))
    else:
        trainval_idx = np.sort(np.concatenate([train_folds[0], val_folds[0]]))
        pos_of = -np.ones(len(y), dtype=int)
        pos_of[trainval_idx] = np.arange(len(trainval_idx))

        X_fit, y_fit, g_fit = X[trainval_idx], y[trainval_idx], groups[trainval_idx]
        splits = [(pos_of[tr], pos_of[va]) for tr, va in zip(train_folds, val_folds)]

    best_model, best_params, fold_metrics = tune_and_eval(
        X_fit, y_fit, splits, model_name, use_gpu=use_gpu)

    for m in fold_metrics:
        logger.info(
            f"    Fold {m['fold']}/{N_FOLDS} — "
            f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}  "
            f"Pearson={m['pearson_r']:.4f}  Spearman={m['spearman_r']:.4f}"
        )

    #  Aggregate metrics
    keys = ["rmse", "mae", "r2", "pearson_r", "spearman_r"]
    summary = {
        "run_timestamp":  _RUN_TIMESTAMP,
        "run_date":       _RUN_DATE,
        "job_id":         _JOB_ID,
        "dataset":        dataset_name,
        "model":          model_name,
        "n_folds":        N_FOLDS,
        "n_iter_tune":    N_ITER,
        "split_strategy": split_strategy,
        "fold_metrics":   fold_metrics,
        "tuned_params":   best_params,
    }
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        summary[f"mean_{k}"] = float(np.mean(vals))
        summary[f"std_{k}"]  = float(np.std(vals))

    logger.info(
        f"  CV — "
        f"RMSE={summary['mean_rmse']:.4f}±{summary['std_rmse']:.4f}  "
        f"MAE={summary['mean_mae']:.4f}±{summary['std_mae']:.4f}  "
        f"R²={summary['mean_r2']:.4f}±{summary['std_r2']:.4f}  "
        f"Pearson={summary['mean_pearson_r']:.4f}±{summary['std_pearson_r']:.4f}  "
        f"Spearman={summary['mean_spearman_r']:.4f}±{summary['std_spearman_r']:.4f}"
    )

    #  Test set evaluation
    test_metrics = {}
    logger.info(f"  Evaluating on held-out test ({len(test_idx)} samples) "
                    f"with the tuned CV params…")
    X_test, y_test = X[test_idx], y[test_idx]
    y_test_pred = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)
    test_metrics["n_samples"]    = int(len(test_idx))
    test_metrics["final_params"] = best_params
    logger.info(
        f"  TEST — RMSE={test_metrics['rmse']:.4f}  MAE={test_metrics['mae']:.4f}  "
        f"R2={test_metrics['r2']:.4f}  Pearson={test_metrics['pearson_r']:.4f}  "
        f"Spearman={test_metrics['spearman_r']:.4f}  n={len(test_idx)}"
        )

    summary["test_metrics"] = test_metrics

    #  Save JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{model_name}_{split_strategy}_cv_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: {json_path}")
    return summary


def _fmt_params(d: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in sorted(d.items()))


#  Compile all JSONs to baseline_results.csv
def compile_results():
    rows = []
    for p in sorted(RESULTS_DIR.glob("*/*_cv_summary.json")):
        s = json.load(open(p))
        base = {
            "run_timestamp":  s.get("run_timestamp"),
            "run_date":       s["run_date"],
            "job_id":         s["job_id"],
            "model_arch":     s["model"],
            "run_name":       f"{s['dataset']}_{s['model']}",
            "head":           s["dataset"],
            "dataset_name":   s["dataset"],
            "pipeline_mode":  "baseline_ml",
            "subset_label":   s["model"],
            "split_strategy": s.get("split_strategy", "group"),
        }
        rows.append({
            **base,
            "mean_rmse":       s.get("mean_rmse"),
            "std_rmse":        s.get("std_rmse"),
            "mean_mae":        s.get("mean_mae"),
            "std_mae":         s.get("std_mae"),
            "mean_r2":         s.get("mean_r2"),
            "std_r2":          s.get("std_r2"),
            "mean_pearson_r":  s.get("mean_pearson_r"),
            "std_pearson_r":   s.get("std_pearson_r"),
            "mean_spearman_r": s.get("mean_spearman_r"),
            "std_spearman_r":  s.get("std_spearman_r"),
        })
        tm = s.get("test_metrics", {})
        if tm:
            rows.append({
                **base,
                "rmse":       tm.get("rmse"),
                "mae":        tm.get("mae"),
                "r2":         tm.get("r2"),
                "pearson_r":  tm.get("pearson_r"),
                "spearman_r": tm.get("spearman_r"),
            })

    append_results(rows, config.BASELINE_RESULTS_CSV)
    logger.info(f"Results written: {config.BASELINE_RESULTS_CSV}")


#  Main 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline ML on Morgan fingerprints with hyperparam tuning")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=list(DATASET_NAME_MAP.keys()))
    parser.add_argument("--model", type=str, required=True,
                        choices=MODEL_NAMES)
    parser.add_argument("--split_strategy", type=str, default=config.SPLIT_STRATEGY,
                        choices=["group", "random"])
    parser.add_argument("--gpu", action="store_true",
                        default=os.environ.get("USE_GPU", "0") == "1")
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_filename = DATASET_NAME_MAP[args.dataset_name]
    label_col    = DF_DIR[csv_filename]
    csv_path     = DATA_DIR / csv_filename

    logger.info(f"=== BASELINE ML ===")
    logger.info(f"Date/Job:    {_RUN_DATE} / {_JOB_ID}")
    logger.info(f"Dataset:     {args.dataset_name}  ({csv_filename})")
    logger.info(f"Label:       {label_col}")
    logger.info(f"Model:       {args.model}")
    logger.info(f"Tuning:      RandomizedSearchCV  n_iter={N_ITER}")
    logger.info(f"Split:       strategy={args.split_strategy}")
    logger.info(f"GPU:         {args.gpu}")
    logger.info(f"Output dir:  {RESULTS_DIR}")

    df = pd.read_csv(csv_path)
    for col in [SMILES_COL, label_col, CLUSTER_COL]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_path}")

    df = df[[SMILES_COL, label_col, CLUSTER_COL]].dropna().reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows. {df[CLUSTER_COL].nunique()} unique clusters")

    fps = df[SMILES_COL].apply(smiles_to_fingerprint)
    invalid = fps.isna().sum()
    if invalid:
        logger.warning(f"{invalid} invalid SMILES")
        mask = fps.notna()
        df, fps = df[mask].reset_index(drop=True), fps[mask]

    X      = np.stack(fps.values)
    y      = df[label_col].values.astype(float)
    groups = df[CLUSTER_COL].values
    train_folds, val_folds, test_idx = make_train_val_test_indices(
            groups, N_FOLDS, args.split_strategy, SEED, test_frac=0.10)

    out_dir = RESULTS_DIR / args.dataset_name
    run_cv(X, y, groups, args.model, args.dataset_name, out_dir,
           train_folds, val_folds, test_idx,
           split_strategy=args.split_strategy,
           use_gpu=args.gpu)

    # Recompile master CSV after every job completes
    compile_results()


if __name__ == "__main__":
    main()
