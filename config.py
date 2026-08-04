"""
config.py
Central configuration for the peptide property prediction pipeline
"""

import os
from pathlib import Path

# Runtime settings
SPLIT_STRATEGY = os.environ.get("SPLIT_STRATEGY", "group")
PIPELINE_MODE  = os.environ.get("PIPELINE_MODE",  "chemprop")
ACCELERATOR    = os.environ.get("ACCELERATOR",    "gpu")
MODEL_ARCH     = os.environ.get("MODEL_ARCH",     "single_task")

# Paths
PACKAGE_DIR    = Path(__file__).resolve().parent
BASE_DIR       = Path(os.environ.get("BASE_DIR", str(PACKAGE_DIR)))
DATA_DIR       = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
CHEMELEON_PATH = Path(os.environ.get("CHEMELEON_PATH", str(BASE_DIR / "chemeleon_mp.pt")))

# HuggingFace cache root for HF models (pepdora, chemberta)
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", str(BASE_DIR / "hf_cache" / "hub")))

def get_checkpoint_root(pipeline_mode: str = None, model_arch: str = None) -> Path:
    """Return run root for a given pipeline_mode and model_arch"""
    pm   = pipeline_mode or PIPELINE_MODE
    arch = model_arch    or MODEL_ARCH
    tag  = f"{arch}_{pm}_{SPLIT_STRATEGY}"
    return BASE_DIR / "runs" / tag

_run_tag        = f"{MODEL_ARCH}_{PIPELINE_MODE}_{SPLIT_STRATEGY}"
CHECKPOINT_ROOT = BASE_DIR / "runs" / _run_tag
RESULTS_DIR     = CHECKPOINT_ROOT
LOG_DIR         = CHECKPOINT_ROOT / "logs"

# Results dir and CSVs
RESULTS_ROOT             = Path(os.environ.get("RESULTS_ROOT", str(BASE_DIR / "results")))
CHEMPROP_RESULTS_CSV     = RESULTS_ROOT / "chemprop_results.csv"
TRANSFORMER_RESULTS_CSV  = RESULTS_ROOT / "transformer_results.csv"
BASELINE_RESULTS_CSV     = RESULTS_ROOT / "baseline_results.csv"


# Path helpers
def pretrain_dir(subset_name: str, pipeline_mode: str = None, model_arch: str = None) -> Path:
    return get_checkpoint_root(pipeline_mode, model_arch) / "pretrain" / subset_name

def pretrain_checkpoint(subset_name: str, pipeline_mode: str = None, model_arch: str = None) -> Path:
    return pretrain_dir(subset_name, pipeline_mode, model_arch) / "final" / "pretrained_model.pt"

def finetune_dir(subset_label: str, run_name: str) -> Path:
    return CHECKPOINT_ROOT / "finetune" / subset_label / run_name

def finetune_checkpoint(subset_label: str, run_name: str) -> Path:
    return finetune_dir(subset_label, run_name) / "final" / "finetuned_model.pt"


# Column definitions
SMILES_COL   = "linear_SMILES"
CLUSTER_COL  = "cluster_umap_hdb"

PRETRAIN_LABEL_COLS = [
    "synthesizability_difficulty",
    "instability",
    "charge_at_pH_7.4",
    "gravy",
    "camsol_score",
]

# Single-task dataset definitions
DATASET_CONFIG = {
    "chemberta_hemolysis_chemprop_550": {
        "data_file": DATA_DIR / "chemberta_hemolysis_chemprop_550.csv",
        "label_col": "revised_50hemo",
        "metrics": ["rmse", "mae", "r2", "pearson", "spearman"],
    },
    "even_chemberta_noncanonical_camsol_chemprop_10k": {
        "data_file": DATA_DIR / "even_chemberta_noncanonical_camsol_chemprop_10k.csv",
        "label_col": "camsol_score",
        "metrics":   ["rmse", "mae", "r2", "pearson", "spearman"],
    },
    "synthesizability": {
        "data_file": DATA_DIR / "synthesize_chemprop.csv",
        "label_col": "L : (L+M) Ratio (%)",
        "metrics":   ["rmse", "mae", "r2", "pearson", "spearman"],
    },
    "trial_syn": {
        "data_file": DATA_DIR / "trial_syn.csv",
        "label_col": "L : (L+M) Ratio (%)",
        "metrics":   ["rmse", "mae", "r2", "pearson", "spearman"],
    },
    "trial_camsol": {
        "data_file": DATA_DIR / "trial_camsol.csv",
        "label_col": "camsol_score",
        "metrics":   ["rmse", "mae", "r2", "pearson", "spearman"],
    },
    "trial_hemo": {
        "data_file": DATA_DIR / "trial_hemo.csv",
        "label_col": "revised_50hemo",
        "metrics":   ["rmse", "mae", "r2", "pearson", "spearman"],
    },
}

# Multi-head run definitions
MULTIHEAD_RUN_CONFIG = {
    "trial": {
        "run_name":        "trial_synth_camsol_hemo",
        "synthesizability":"trial_syn",
        "camsol":          "trial_camsol",
        "hemolysis":       "trial_hemo",
    },
    "full_run": {
        "run_name":        "run3_new_chemberta_synth_camsol10k_hemo550",
        "synthesizability":"synthesizability",
        "camsol":          "even_chemberta_noncanonical_camsol_chemprop_10k",
        "hemolysis":       "chemberta_hemolysis_chemprop_550",
    }
}

# Hyperparameters
PRETRAIN_EPOCHS = int(os.environ.get("PRETRAIN_EPOCHS", 30))
PRETRAIN_BATCH  = int(os.environ.get("PRETRAIN_BATCH",  512))
PRETRAIN_LR     = float(os.environ.get("PRETRAIN_LR",  6e-4))

FINETUNE_EPOCHS = int(os.environ.get("FINETUNE_EPOCHS", 50))
FINETUNE_BATCH  = int(os.environ.get("FINETUNE_BATCH",  128))
FINETUNE_LR     = float(os.environ.get("FINETUNE_LR",   1e-4))
FREEZE_ENCODER  = os.environ.get("FREEZE_ENCODER", "false").lower() == "true"

#  Architecture hyperparameters
# To override, set env vars before running:
MPNN_DEPTH            = int(os.environ.get("MPNN_DEPTH",            3))
MPNN_D_H              = int(os.environ.get("MPNN_D_H",              300))
FFN_NUM_LAYERS        = int(os.environ.get("FFN_NUM_LAYERS",        2))
FFN_HIDDEN_DIM        = int(os.environ.get("FFN_HIDDEN_DIM",        300))
AGGREGATION           = os.environ.get("AGGREGATION",               "mean")
DROPOUT               = float(os.environ.get("DROPOUT",             0.0))
FREEZE_ENCODER_EPOCHS = int(os.environ.get("FREEZE_ENCODER_EPOCHS", 0))

NUM_WORKERS     = int(os.environ.get("NUM_WORKERS", 8))
N_FOLDS         = int(os.environ.get("N_FOLDS", 5))
SEED            = 42
