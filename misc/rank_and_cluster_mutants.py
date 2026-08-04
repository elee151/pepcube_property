import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import hdbscan
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import rdFingerprintGenerator

NCAA = ['NLE', 'Cha', 'AIB', 'ivDde', 'Chg', 'GABA']

AA_CLASSES = {
    'aromatic':    ['F', 'W', 'Y'],
    'basic':       ['K', 'R', 'H'],
    'acidic':      ['D', 'E'],
    'polar':       ['S', 'T', 'N', 'Q'],
    'hydrophobic': ['A', 'I', 'L', 'V', 'M'],
    'small':       ['G', 'P'],
}
AA_TO_CLASS = {aa: cls for cls, aas in AA_CLASSES.items() for aa in aas}
ALL_CANONICAL = [aa for aas in AA_CLASSES.values() for aa in aas]
_NCAA_PATTERN = "|".join(sorted((re.escape(a) for a in NCAA), key=len, reverse=True))
_RESIDUE_RE = re.compile(f"{_NCAA_PATTERN}|[A-Z]")


def tokenize_seq(seq):
    """Split a peptide sequence into residue tokens, keeping NCAA codes intact."""
    if not isinstance(seq, str) or not seq:
        return []
    if any(sep in seq for sep in ("-", ",", " ")):
        return [t for t in re.split(r"[-, ]+", seq) if t]
    return _RESIDUE_RE.findall(seq)


def get_mutations(starter_seq, mutated_seq, positions):
    """Return the list of new (mutant) residues"""
    if isinstance(positions, str):
        try:
            positions = ast.literal_eval(positions)
        except (ValueError, SyntaxError):
            positions = [positions]
    if positions is None or (isinstance(positions, float) and pd.isna(positions)):
        return []
    if not isinstance(positions, (list, tuple)):
        positions = [positions]

    mutated_tokens = tokenize_seq(mutated_seq)
    new_aas = []
    for pos in positions:
        try:
            pos = int(pos)
        except (TypeError, ValueError):
            continue
        if 0 <= pos < len(mutated_tokens):
            new_aas.append(mutated_tokens[pos])
    return new_aas

# Paths
DATA_DIR = Path("./predictions")
OUT_DIR  = Path("./predictions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model mapping
MODEL_SUFFIX_MAP = {
    "lora": "lora_chemberta",
    "chemberta_def": "chemberta_default",
    "chemprop_scratch": "chemprop_scratch",
    "new1M": "chemprop_pretrained",
}

ALLOWED_SOURCE_GROUPS = [
    ("chemprop_pretrained",),
    ("lora_chemberta",),
    ("chemprop_pretrained", "lora_chemberta"),
]
ALLOWED_SOURCE_SETS = [frozenset(g) for g in ALLOWED_SOURCE_GROUPS]

def source_set(source_val):
    sources = source_val if isinstance(source_val, list) else [source_val]
    return frozenset(sources)

# File names
def orig_path(suffix):
    return DATA_DIR / f"experimental_sequences_{suffix}.csv"

def mut_path(suffix):
    return DATA_DIR / f"mutated_experimental_peptides_150k_{suffix}.csv"

# Columns
ORIG_ID_COL   = "Sample"
MUT_PARENT_COL = "peptide_name"
DEDUP_KEY      = "linear_SMILES"
SMILES_COL     = "linear_SMILES"

SCORE_COLS = ["pred_synthesizability", "pred_camsol", "pred_hemolysis"]

# +1 for larger is better, -1 for lower is better.
IMPROVE_SIGN = {
    "pred_synthesizability": 1,
    "pred_camsol": 1,
    "pred_hemolysis": 1,
}

# score columns to ignore
AGNOSTIC_COLS = {}

TOP_N    = 1000
MID_N    = 100
BOTTOM_N = 100


def rank_model(suffix, display_name):
    orig_df = pd.read_csv(orig_path(suffix))
    mut_df  = pd.read_csv(mut_path(suffix))

    missing_orig = [c for c in SCORE_COLS if c not in orig_df.columns]
    missing_mut  = [c for c in SCORE_COLS if c not in mut_df.columns]
    d_synth = "delta_pred_synthesizability_pct"
    d_cam   = "delta_pred_camsol_pct"

    directional_cols = [c for c in SCORE_COLS if c not in AGNOSTIC_COLS]
    d_cols_all = [f"delta_{c}_pct" for c in SCORE_COLS]
    d_cols_directional = [f"delta_{c}_pct" for c in directional_cols]
    required_positive_cols = [f"delta_{c}_pct" for c in directional_cols if c != "pred_synthesizability"]
    required_negative_cols = d_cols_directional

    per_peptide_selections = []

    for _, orig_row in orig_df.iterrows():
        parent_id = orig_row[ORIG_ID_COL]
        mask = mut_df[MUT_PARENT_COL] == parent_id
        subset = mut_df[mask].copy()
        if subset.empty:
            continue

        for col in SCORE_COLS:
            orig_val = orig_row[col]
            raw_pct = (subset[col] - orig_val) / abs(orig_val) * 100
            subset[f"delta_{col}_pct"] = raw_pct * IMPROVE_SIGN.get(col, 1)

        subset["total_abs_change"] = subset[d_cols_all].abs().sum(axis=1)
        subset["delta_avg_directional"] = subset[d_cols_directional].mean(axis=1)

        if required_positive_cols:
            top_mask = np.all([subset[c] > 0 for c in required_positive_cols], axis=0)
            top_pool = subset[top_mask]
        else:
            top_pool = subset
        top = top_pool.nlargest(TOP_N, d_synth).copy()
        top["category"] = "top_1000"

        # no change
        used_idx = set(top.index)
        mid_pool = subset.drop(index=used_idx, errors="ignore")
        middle = mid_pool.nsmallest(MID_N, "total_abs_change").copy()
        middle["category"] = "middle_no_change"

        # bottom and low tanimoto
        used_idx |= set(middle.index)
        bottom_pool = subset.drop(index=used_idx, errors="ignore")
        bottom_mask = np.all([bottom_pool[c] < 0 for c in required_negative_cols], axis=0)
        bottom_pool = bottom_pool[bottom_mask]
        bottom = bottom_pool.nsmallest(BOTTOM_N, "delta_avg_directional").copy()
        bottom["category"] = "bottom_100_worst"
        per_peptide_selections.append(pd.concat([top, middle, bottom], ignore_index=True))

    selected = pd.concat(per_peptide_selections, ignore_index=True)
    selected["source"] = display_name
    selected["pct_dif"] = selected[d_synth]  # primary ranking metric, for convenience
    selected["pct_dif_synth"] = selected[d_synth]
    selected["pct_dif_camsol"] = selected["delta_pred_camsol_pct"]
    selected["pct_dif_hemolysis"] = selected["delta_pred_hemolysis_pct"]
    return selected


# combine and create final df
def build_combined_selection():
    frames = []
    for suffix, display_name in MODEL_SUFFIX_MAP.items():
        frames.append(rank_model(suffix, display_name))

    combined = pd.concat(frames, ignore_index=True)

    def agg_group(g):
        first = g.iloc[0].copy()
        first["source"] = sorted(set(g["source"]))
        first["n_sources"] = len(first["source"])
        cats = sorted(set(g["category"]))
        first["category"] = cats[0] if len(cats) == 1 else " / ".join(cats)
        first["pct_dif_by_source"] = dict(zip(g["source"], g["pct_dif"]))
        first["pct_dif"] = g["pct_dif"].mean()
        first["pct_dif_synth"] = g["pct_dif_synth"].mean()
        first["pct_dif_camsol"] = g["pct_dif_camsol"].mean()
        first["pct_dif_hemolysis"] = g["pct_dif_hemolysis"].mean()

        #saves the average predicted scores
        first["pred_synthesizability"] = g["pred_synthesizability"].mean()
        first["pred_camsol"] = g["pred_camsol"].mean()
        first["pred_hemolysis"] = g["pred_hemolysis"].mean()
        return first

    deduped = combined.groupby(DEDUP_KEY, group_keys=False).apply(agg_group).reset_index(drop=True)
    return deduped

# umap and tanimoto for plotting
def build_all_vis(deduped_mutants):

    parent_frames = []
    for suffix, display_name in MODEL_SUFFIX_MAP.items():
        p = pd.read_csv(orig_path(suffix))
        p["source"] = display_name
        p["category"] = "experimental_parent"
        p["pct_dif"] = np.nan
        p["pct_dif_synth"] = np.nan
        p["pct_dif_camsol"] = np.nan
        p["pct_dif_hemolysis"] = np.nan
        p["peptide_group"] = p[ORIG_ID_COL]
        parent_frames.append(p)
    parents = pd.concat(parent_frames, ignore_index=True)
    if ORIG_ID_COL in parents.columns:
        parents = parents.drop_duplicates(subset=ORIG_ID_COL)

    deduped_mutants = deduped_mutants.copy()
    deduped_mutants["peptide_group"] = deduped_mutants[MUT_PARENT_COL]

    MUTANT_ONLY_COLS = ["starter_seq", "n_mutations", "positions", "mutated_seq"]

    common_cols = list(set(deduped_mutants.columns) & set(parents.columns))
    keep_cols = sorted(set(common_cols) | set(MUTANT_ONLY_COLS))

    for col in MUTANT_ONLY_COLS:
        if col not in parents.columns:
            parents[col] = np.nan

    all_vis = pd.concat(
        [deduped_mutants[keep_cols], parents[keep_cols]],
        ignore_index=True,
    )

    all_vis["mutations_made"] = all_vis.apply(
        lambda r: get_mutations(r.get("starter_seq"), r.get("mutated_seq"), r.get("positions")),
        axis=1,
    )

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    def _smi_to_fp(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return mfpgen.GetFingerprint(mol)

    all_vis["Morgan_linear"] = all_vis["linear_SMILES"].apply(_smi_to_fp)

    def _group_tanimoto(g):
        fps = list(g["Morgan_linear"])
        if len(fps) <= 1:
            return pd.Series(np.nan, index=g.index)
        sims = [(sum(BulkTanimotoSimilarity(fp, fps)) - 1) / (len(fps) - 1) for fp in fps]
        return pd.Series(sims, index=g.index)

    all_vis["avg_tanimoto_sim"] = all_vis.groupby("peptide_group", group_keys=False).apply(_group_tanimoto)
    return all_vis


# select the final sunsets of top1000, middle, bottom, create dict in category_n to modify # selected
def select_final_subset(all_vis, category_n=None):
    default_n = {"top_1000": 2, "middle_no_change": 0, "bottom_100_worst": 1}
    category_n = {**default_n, **(category_n or {})}

    def _in_category(cat_val, target):
        return target in str(cat_val)

    all_vis = all_vis.copy()
    all_vis["_source_set"] = all_vis["source"].apply(source_set)
    all_vis = all_vis[all_vis["_source_set"].isin(ALLOWED_SOURCE_SETS)]

    picks = []
    for (group_id, src_set), g in all_vis.groupby(["peptide_group", "_source_set"]):
        top_pool = g[g["category"].apply(lambda c: _in_category(c, "top_1000"))]
        picks.append(top_pool.nlargest(category_n["top_1000"], "pct_dif"))

        mid_pool = g[g["category"].apply(lambda c: _in_category(c, "middle_no_change"))]
        picks.append(mid_pool.nsmallest(category_n["middle_no_change"], "avg_tanimoto_sim"))

        bottom_pool = g[g["category"].apply(lambda c: _in_category(c, "bottom_100_worst"))].copy()
        if not bottom_pool.empty:
            bottom_pool["_pct_dif_rank"] = bottom_pool["pct_dif"].rank(ascending=True)
            bottom_pool["_tanimoto_rank"] = bottom_pool["avg_tanimoto_sim"].rank(ascending=True)
            bottom_pool["_combined_rank"] = bottom_pool["_pct_dif_rank"] + bottom_pool["_tanimoto_rank"]
            bottom_pick = bottom_pool.nsmallest(category_n["bottom_100_worst"], "_combined_rank")
            bottom_pick = bottom_pick.drop(columns=["_pct_dif_rank", "_tanimoto_rank", "_combined_rank"])
            picks.append(bottom_pick)

    final = pd.concat(picks, ignore_index=True)
    final = final.drop(columns=["_source_set"])
    final = final.drop_duplicates(subset=DEDUP_KEY)

    n_groups = all_vis["peptide_group"].nunique()
    n_combos = all_vis.groupby(["peptide_group", "_source_set"]).ngroups
    target = sum(category_n.values()) * n_combos
    return final