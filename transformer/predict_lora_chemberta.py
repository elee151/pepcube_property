"""
predict_lora_chemberta.py
Load a saved LoRAMultiHeadModel (from lora_chemberta_multitask.py) and run
inference on a CSV of novel peptides

Expects:
    <model_dir>/
      model.pt          full LoRAMultiHeadModel state dict (backbone + heads)
      scalers.pt         {task: {mean, scale}} for all three heads
      lora_adapter/      PEFT adapter dir (adapter_config.json + weights)

Example:
    python predict_lora_chemberta.py --model_dir runs/chemberta_lora_group_raw/finetune/run3_new_chemberta_synth_camsol10k_hemo550/final_model --input_csv experimental_sequences.csv --output_csv predictions.csv
"""

import argparse
import json
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
from peft import PeftModel

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


def pool(backbone_out, attention_mask, pooling):
    if pooling == "cls":
        return backbone_out.last_hidden_state[:, 0, :]
    token_emb = backbone_out.last_hidden_state
    mask_exp  = attention_mask.unsqueeze(-1).float()
    summed    = (token_emb * mask_exp).sum(dim=1)
    counts    = mask_exp.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class LoRAMultiHeadInference(nn.Module):
    """Merged (adapter-free) backbone + per-task heads, inference only."""
    def __init__(self, backbone, hidden_size, pooling):
        super().__init__()
        self.backbone = backbone
        self.pooling  = pooling
        self.heads    = nn.ModuleDict({
            name: nn.Sequential(nn.Dropout(0.0), nn.Linear(hidden_size, 1))
            for name in HEAD_KEYS
        })

    def _encode(self, input_ids, attention_mask):
        seq_len      = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long,
                                    device=input_ids.device).unsqueeze(0)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids)
        return pool(out, attention_mask, self.pooling)

    def forward(self, input_ids, attention_mask):
        h = self._encode(input_ids, attention_mask)
        return {name: head(h).squeeze(-1) for name, head in self.heads.items()}


def load_model(model_dir, pooling, base_model_override, device):
    model_dir     = Path(model_dir)
    adapter_dir   = model_dir / "lora_adapter"
    model_path    = model_dir / "model.pt"
    scaler_path   = model_dir / "scalers.pt"

    for p in (adapter_dir, model_path, scaler_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing expected: {p}")

    if base_model_override:
        base_model_path = base_model_override
    else:
        with open(adapter_dir / "adapter_config.json") as f:
            base_model_path = json.load(f)["base_model_name_or_path"]
    logger.info(f"Base model: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    base    = AutoModel.from_pretrained(base_model_path)
    peft_bb = PeftModel.from_pretrained(base, str(adapter_dir))
    backbone = peft_bb.merge_and_unload()
    hidden_size = backbone.config.hidden_size

    model = LoRAMultiHeadInference(backbone, hidden_size, pooling)

    full_state = torch.load(model_path, map_location="cpu", weights_only=False)
    head_state = {k[len("heads."):]: v for k, v in full_state.items() if k.startswith("heads.")}
    missing, unexpected = model.heads.load_state_dict(head_state, strict=True)
    logger.info(f"Loaded {len(head_state)} head tensors from {model_path}")

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
                        help="Directory containing model.pt, scalers.pt, lora_adapter/")
    parser.add_argument("--input_csv",  type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="predictions_lora_chemberta.csv")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Sequences tokenized + forward-passed per chunk")
    parser.add_argument("--pooling",    type=str, default="mean", choices=["mean", "cls"])
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Override the base backbone path (default: read from adapter_config.json)")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, tokenizer, scalers = load_model(
        args.model_dir, args.pooling, args.base_model_path, device
    )

    input_df   = pd.read_csv(args.input_csv)
    smiles_col = config.SMILES_COL
    if smiles_col not in input_df.columns:
        raise ValueError(f"Input CSV must have a '{smiles_col}' column")

    all_smiles = input_df[smiles_col].tolist()
    n_total    = len(all_smiles)

    all_preds  = {name: [] for name in HEAD_KEYS}
    kept_mask  = []

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
                # if sequence(s) in this chunk hit the truncation limit
                lengths = [len(tokenizer(s, truncation=False)["input_ids"]) for s in chunk]
                over = [j for j, L in enumerate(lengths) if L > args.max_tokens]

            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            preds = model(input_ids, attention_mask)
            for name in HEAD_KEYS:
                z  = preds[name].cpu().numpy()
                sc = scalers[name]
                all_preds[name].append(z * sc.scale_[0] + sc.mean_[0])

            kept_mask.extend([True] * len(chunk))

            # free chunk tensors before the next iteration
            del enc, input_ids, attention_mask, preds

            if i % (args.batch_size * 20) == 0:
                logger.info(f"  Predicted {i}/{n_total}")

    results = input_df.copy()
    for name in HEAD_KEYS:
        results[f"pred_{name}"] = np.concatenate(all_preds[name])

    results.to_csv(args.output_csv, index=False)
    logger.info(f"Predictions saved: {args.output_csv}")


if __name__ == "__main__":
    main()
