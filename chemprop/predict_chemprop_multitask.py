"""
predict_chemprop_multitask_full.py
Loads finetuned multi-head Chemprop checkpoint using evaluate_chemprop_multitask.py logic

Example:
    python predict_chemprop_multitask_load.py \
        --model_dir pretrained_chemprop_90/final \
        --subset_name new_1M \
        --input_csv data/experimental_sequences.csv \
        --output_csv preds_90.csv
"""

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MODEL_ARCH", "multi_head")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from chemprop import featurizers, nn as cpnn
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

import importlib, sys
_mh_path = Path(__file__).parent / "finetune_chemprop_multitask.py"
_spec     = importlib.util.spec_from_file_location("finetune_multihead", _mh_path)
_mh_mod   = importlib.util.module_from_spec(_spec)
sys.modules["finetune_multihead"] = _mh_mod
_spec.loader.exec_module(_mh_mod)

MultiHeadMPNN = _mh_mod.MultiHeadMPNN
HEAD_KEYS     = _mh_mod.HEAD_KEYS
load_encoder  = _mh_mod.load_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def bmg_to_device(bmg, device):
    for attr in ("V", "E", "edge_index", "batch"):
        val = getattr(bmg, attr, None)
        if isinstance(val, torch.Tensor):
            setattr(bmg, attr, val.to(device))
    return bmg


def get_device():
    if config.ACCELERATOR == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if config.ACCELERATOR == "gpu" and not torch.cuda.is_available():
        logger.warning("ACCELERATOR=gpu but no CUDA device is available — run on cpu")
    return torch.device("cpu")


#  Model loading, same as evaluate_chemprop_multitask.py
def load_scalers(scalers_path) -> dict:
    saved = torch.load(scalers_path, map_location="cpu", weights_only=False)
    scalers = {}
    for name, d in saved.items():
        sc = StandardScaler()
        sc.mean_  = d["mean"]
        sc.scale_ = d["scale"]
        scalers[name] = sc
    return scalers


def resolve_encoder_info(args):
    """Resolves in same way as evaluate_chemprop_multitask.py's in main()."""
    if config.PIPELINE_MODE == "chemprop":
        return "chemprop_random_init", "n/a (random init)"
    elif config.PIPELINE_MODE == "chemeleon":
        return "chemeleon", str(config.CHEMELEON_PATH)
    else:  # "pretrained"
        if args.subset_name is None:
            raise ValueError("--subset_name required when PIPELINE_MODE=pretrained")
        return "pretrained", str(config.pretrain_checkpoint(args.subset_name))


def resolve_actual_depth(actual_encoder, actual_pretrain_ckpt):
    if actual_encoder == "chemprop_random_init":
        return config.MPNN_DEPTH
    mp = load_encoder(actual_pretrain_ckpt)
    return mp.depth


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing finetuned_model.pt + scalers.pt")
    parser.add_argument("--subset_name", type=str, default=None,
                        help="Required when PIPELINE_MODE=pretrained, to help depth of the pretrain checkpoint")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="CSV with a linear_SMILES column to predict on")
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    logger.info(f"Device: {device}")

    model_dir    = Path(args.model_dir)
    model_path   = model_dir / "finetuned_model.pt"
    scalers_path = model_dir / "scalers.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"No model weights found at {model_path}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"No scalers found at {scalers_path}")

    actual_encoder, actual_pretrain_ckpt = resolve_encoder_info(args)
    actual_depth = resolve_actual_depth(actual_encoder, actual_pretrain_ckpt)
    logger.info(f"Encoder:      {actual_encoder}  ({actual_pretrain_ckpt})")
    logger.info(f"Actual depth: {actual_depth} (config.MPNN_DEPTH={config.MPNN_DEPTH})")

    scalers = load_scalers(scalers_path)
    model   = load_multihead_model(model_path, scalers, depth=actual_depth)
    model.to(device)
    model.eval()
    logger.info(f"Loaded model weights from {model_path}")

    #  Inference on input CSV
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    input_df = pd.read_csv(args.input_csv)
    smiles_col = config.SMILES_COL
    if smiles_col not in input_df.columns:
        raise ValueError(f"Input CSV must have a '{smiles_col}' column")

    pred_smiles = input_df[smiles_col].values

    valid_mask = [Chem.MolFromSmiles(s) is not None for s in pred_smiles]
    n_invalid = sum(not v for v in valid_mask)
    if n_invalid:
        logger.warning(f"{n_invalid} SMILES could not be parsed and will be skipped")

    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    valid_smiles = [pred_smiles[i] for i in valid_idx]

    PRED_CHUNK = 256
    all_preds = {name: [] for name in HEAD_KEYS}

    with torch.no_grad():
        for i in range(0, len(valid_smiles), PRED_CHUNK):
            chunk = valid_smiles[i : i + PRED_CHUNK]
            molgraphs = [featurizer(Chem.MolFromSmiles(s)) for s in chunk]
            bmg = BatchMolGraph(molgraphs)
            bmg = bmg_to_device(bmg, device)
            preds = model.predict(bmg)
            for name in HEAD_KEYS:
                all_preds[name].append(preds[name].squeeze(-1).cpu().numpy())
            if i % 10000 == 0:
                logger.info(f"  Predicted {i}/{len(valid_smiles)}")

    results = input_df.iloc[valid_idx].copy()
    for name in HEAD_KEYS:
        results[f"pred_{name}"] = np.concatenate(all_preds[name])

    results.to_csv(args.output_csv, index=False)
    logger.info(f"Predictions saved: {args.output_csv} ({len(results)} rows)")


if __name__ == "__main__":
    main()
