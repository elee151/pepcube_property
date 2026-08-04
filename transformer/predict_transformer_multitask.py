"""
predict_transformer_multitask.py
Load a saved TransformerMultiHead full-finetune model and run inference on a CSV
of novel peptides.

Expects:
    <model_dir>/
      model.pt      full TransformerMultiHead state dict (encoder + heads)
      scalers.pt    {task: {mean, scale}} for all three heads

Example:
    python predict_transformer_multitask.py --model_dir runs/finetune_transformer/chemberta-77m-mtr/run3/final_model --model_name chemberta-77m-mtr --input_csv experimental_sequences.csv --output_csv predictions.csv
"""

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEAD_KEYS = ["synthesizability", "camsol", "hemolysis"]

#  Local HF cache paths
_HF_HUB_ROOT = config.HF_HUB_CACHE

CHEMBERTA_MTR_DIR = _HF_HUB_ROOT / (
    "models--DeepChem--ChemBERTa-77M-MTR/snapshots/"
    "fc007d31c2fb774ab7a8e5a8d318e25cb01d2da1"
)

CHEMBERTA_MLM_WEIGHTS_DIR = _HF_HUB_ROOT / (
    "models--DeepChem--ChemBERTa-77M-MLM/snapshots/"
    "d62f784b9a0a3aab09c788a7fb95a8e1ce89116f"
)

CHEMBERTA_MLM_TOKENIZER_DIR = _HF_HUB_ROOT / (
    "models--DeepChem--ChemBERTa-77M-MLM/snapshots/"
    "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"
)

PEPDORA_ADAPTER_DIR = _HF_HUB_ROOT / (
    "models--ChatterjeeLab--PepDoRA/snapshots/"
    "e034544e8f2ab1c34fffcfd4984f4183db7f12ed"
)

def _check_local_dir(path: Path, required_files: list[str], label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{label}: directory not found: {path}\n"
            f"Check HF_HUB_CACHE / hf_cache layout — see script header."
        )
    missing = [f for f in required_files if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"{label}: missing {missing} in {path}")


#  Models
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

class TransformerMultiHeadInference(nn.Module):
    """Encoder + per-task linear heads, inference only"""
    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, 1) for name in HEAD_KEYS
        })

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)

        enc_inputs = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "position_ids":   position_ids,
        }
        if token_type_ids is not None:
            enc_inputs["token_type_ids"] = token_type_ids

        out = self.encoder(**enc_inputs)
        h = out.last_hidden_state[:, 0, :]  # [CLS] token, matching training
        return {name: head(h).squeeze(-1) for name, head in self.heads.items()}


def _load_encoder(model_cfg, pretrain_ckpt=None):
    trust_rc = model_cfg["trust_remote_code"]
    if pretrain_ckpt:
        return AutoModel.from_pretrained(pretrain_ckpt, trust_remote_code=trust_rc)
    elif model_cfg["is_pepdora"]:
        from peft import PeftModel
        base = AutoModel.from_pretrained(model_cfg["base_model"], trust_remote_code=trust_rc)
        enc  = PeftModel.from_pretrained(base, model_cfg["hf_name"])
        return enc.merge_and_unload()
    else:
        return AutoModel.from_pretrained(model_cfg["hf_name"], trust_remote_code=trust_rc)


def load_model(model_dir, model_name, pretrain_ckpt, device):
    model_dir   = Path(model_dir)
    model_path  = model_dir / "model.pt"
    scaler_path = model_dir / "scalers.pt"

    for p in (model_path, scaler_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing expected artifact: {p}")

    model_cfg = MODEL_REGISTRY[model_name]

    if not pretrain_ckpt:
        if model_name == "chemberta-77m-mtr":
            _check_local_dir(CHEMBERTA_MTR_DIR,
                ["config.json", "model.safetensors"], "ChemBERTa-77M-MTR")
        elif model_name == "pepdora":
            _check_local_dir(CHEMBERTA_MLM_WEIGHTS_DIR,
                ["config.json", "model.safetensors"], "ChemBERTa-77M-MLM weights")
            _check_local_dir(CHEMBERTA_MLM_TOKENIZER_DIR,
                ["tokenizer.json", "vocab.json", "merges.txt"], "ChemBERTa-77M-MLM tokenizer")
            _check_local_dir(PEPDORA_ADAPTER_DIR,
                ["adapter_config.json", "adapter_model.safetensors"], "PepDoRA adapter")

    logger.info(f"Base model: {model_cfg['hf_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["tokenizer_path"], trust_remote_code=model_cfg["trust_remote_code"])

    encoder     = _load_encoder(model_cfg, pretrain_ckpt)
    hidden_size = encoder.config.hidden_size
    model = TransformerMultiHeadInference(encoder, hidden_size)

    # Full finetune saves the whole model state dict (encoder + heads) for inference
    full_state = torch.load(model_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(full_state, strict=True)
    logger.info(f"Loaded model weights from {model_path}")

    saved_scalers = torch.load(scaler_path, map_location="cpu", weights_only=False)
    scalers = {}
    for name, d in saved_scalers.items():
        sc = StandardScaler()
        sc.mean_  = d["mean"]
        sc.scale_ = d["scale"]
        scalers[name] = sc

    model.to(device)
    model.eval()
    return model, tokenizer, scalers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  type=str, required=True,
                        help="Directory containing model.pt and scalers.pt "
                             "(the Stage-3 final_model dir from finetune_chemberta_multitask.py)")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Architecture the saved model was finetuned from")
    parser.add_argument("--input_csv",  type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="predictions_transformer.csv")
    parser.add_argument("--batch_size",            type=int,   default=16,
                        help="Sequences tokenized + forward-passed per chunk")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--pretrain_ckpt",        type=str,   default=None,
                        help="Path to pretrained backbone dir (overrides HF load)")

    args = parser.parse_args()
    return args


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, tokenizer, scalers = load_model(
        args.model_dir, args.model_name, args.pretrain_ckpt, device
    )

    input_df   = pd.read_csv(args.input_csv)
    smiles_col = config.SMILES_COL
    if smiles_col not in input_df.columns:
        raise ValueError(f"Input CSV must have a '{smiles_col}' column")

    all_smiles = input_df[smiles_col].tolist()
    n_total    = len(all_smiles)
    all_preds = {name: [] for name in HEAD_KEYS}

    with torch.no_grad():
        for i in range(0, n_total, args.batch_size):
            chunk = all_smiles[i : i + args.batch_size]

            enc = tokenizer(
                chunk,
                truncation=True,
                max_length=args.max_tokens,
                padding="longest",
                return_tensors="pt",
            )
            n_tokens = enc["input_ids"].shape[1]
            if n_tokens >= args.max_tokens:
                lengths = [len(tokenizer(s, truncation=False)["input_ids"]) for s in chunk]
                over = [j for j, L in enumerate(lengths) if L > args.max_tokens]

            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            token_type_ids = enc.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            preds = model(input_ids, attention_mask, token_type_ids)
            for name in HEAD_KEYS:
                z  = preds[name].cpu().numpy()
                sc = scalers[name]
                all_preds[name].append(z * sc.scale_[0] + sc.mean_[0])

            # free tensors before the next iteration
            del enc, input_ids, attention_mask, token_type_ids, preds

    results = input_df.copy()
    for name in HEAD_KEYS:
        results[f"pred_{name}"] = np.concatenate(all_preds[name])

    results.to_csv(args.output_csv, index=False)
    logger.info(f"Predictions saved: {args.output_csv}")

if __name__ == "__main__":
    main()
