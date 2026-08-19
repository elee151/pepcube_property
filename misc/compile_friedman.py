"""
compile_friedman.py

Parse per-fold CV metrics from result JSONs, compute mean rank per model, run Friedman's test for significance, and
make a heatmap
"""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

METRIC_KEYS = ["rmse", "mae", "r2", "pearson_r", "spearman_r"]
HIGHER_IS_BETTER = {"r2", "pearson_r", "spearman_r"}

DATASET_TO_PROPERTY = {
    "chemberta_hemolysis_chemprop_550": "hemolysis",
    "even_chemberta_noncanonical_camsol_chemprop_10k": "camsol",
    "synthesizability": "synthesizability",
    "trial_syn": "synthesizability",
    "trial_camsol": "camsol",
    "trial_hemo": "hemolysis",
}


def infer_property(dataset_name):
    if dataset_name in DATASET_TO_PROPERTY:
        return DATASET_TO_PROPERTY[dataset_name]
    name = (dataset_name or "").lower()
    if "hemo" in name:
        return "hemolysis"
    if "camsol" in name:
        return "camsol"
    if "synth" in name:
        return "synthesizability"
    return dataset_name or "unknown"


MODEL_LABEL_MAP = {
    ("chemberta_lora_multi_head", "transformer", "lora_finetune"): "LoRA ChemBERTa-77M-MTR",
    ("transformer_multi_head_chemberta-77m-mtr", "transformer", "chemberta-77m-mtr"): "ChemBERTa-77M-MTR",
    ("transformer_multi_head_pepdora", "transformer", "pepdora"): "PepDoRA",
    ("transformer_single_task_chemberta-77m-mtr", "transformer", "chemberta-77m-mtr"): "ChemBERTa-77M-MTR (Single-task)",
    ("transformer_single_task_pepdora", "transformer", "pepdora"): "PepDoRA (Single-task)",

    ("multi_head", "chemprop", "chemprop"): "ChemProp (Multi-head)",
    ("single_task", "chemprop", "chemprop"): "ChemProp (Single-task)",
    ("multi_head", "chemeleon", "chemeleon"): "Chemeleon (Multi-head)",
    ("single_task", "chemeleon", "chemeleon"): "Chemeleon (Single-task)",
    ("multi_head", "chemprop", "new_1M"): "ChemProp (Multi-head Pretrained, Trainable Encoder)",
    ("multi_head", "chemprop", "new_1M_frozen_hpo_small_batch"): "ChemProp (Multi-head Pretrained, Frozen Encoder)",

    ("random_forest", "baseline_ml", "random_forest"): "Random Forest",
    ("catboost", "baseline_ml", "catboost"): "CatBoost",
    ("xgboost", "baseline_ml", "xgboost"): "XGBoost",
    ("svm", "baseline_ml", "svm"): "SVM",
}

BASELINE_MODELS = {"random_forest", "catboost", "xgboost", "svm"}


def classify_model(data):
    model = data.get("model", "")
    if model in BASELINE_MODELS:
        key = (model, "baseline_ml", model)
        return key, "classical", "baseline_ml"

    model_arch = data.get("model_arch", "")

    if "lora" in model_arch:
        key = (model_arch, "transformer", "lora_finetune")
        return key, "transformer", "lora_finetune"

    if model_arch.startswith("transformer"):
        model_name = data.get("model_name", "unknown")
        key = (f"{model_arch}_{model_name}", "transformer", model_name)
        return key, "transformer", model_name

    if model_arch in ("multi_head", "single_task"):
        pipeline_mode = data.get("pipeline_mode", "")
        subset_label = data.get("subset_label", pipeline_mode or "unknown")
        sub_family = "chemeleon" if pipeline_mode == "chemeleon" else "chemprop"
        key = (model_arch, sub_family, subset_label)
        return key, "chemprop", sub_family

    key = (model_arch or model or "unknown", "unknown", "unknown")
    return key, "unknown", "unknown"


def get_model_label(key):
    return MODEL_LABEL_MAP.get(key, "_".join(str(k) for k in key if k))


def rows_from_fold_list(fold_list, model_label, family, sub_family, prop, split_strategy, source_file):
    rows = []
    for entry in fold_list:
        row = {
            "source_file": source_file,
            "model": model_label,
            "family": family,
            "sub_family": sub_family,
            "property": prop,
            "split_strategy": split_strategy,
            "fold": entry.get("fold"),
        }
        for key in METRIC_KEYS:
            row[key] = entry.get(key)
        rows.append(row)
    return rows


def parse_file(path):
    with open(path) as f:
        data = json.load(f)

    key, family, sub_family = classify_model(data)
    model_label = get_model_label(key)
    if key not in MODEL_LABEL_MAP:
        print(f"unmapped model key {key} in {path.name}, using fallback label '{model_label}'")
    split_strategy = data.get("split_strategy", "unknown")
    source_file = path.name
    rows = []

    if "per_head_fold_results" in data:
        for prop, fold_list in data["per_head_fold_results"].items():
            rows.extend(rows_from_fold_list(fold_list, model_label, family, sub_family, prop, split_strategy, source_file))
        return rows

    fold_list = data.get("fold_results") or data.get("fold_metrics")
    if fold_list is None:
        print(f"skipping {source_file}: no per-fold data found")
        return rows

    dataset_name = data.get("dataset_name") or data.get("dataset")
    prop = infer_property(dataset_name)
    rows.extend(rows_from_fold_list(fold_list, model_label, family, sub_family, prop, split_strategy, source_file))
    return rows


def compile_folder(input_dir):
    all_rows = []
    for path in sorted(Path(input_dir).glob("*.json")):
        all_rows.extend(parse_file(path))
    return pd.DataFrame(all_rows)


def compute_mean_ranks(df, group_cols, metrics):
    rows = []
    for metric in metrics:
        higher_better = metric in HIGHER_IS_BETTER
        for key, group in df.groupby(group_cols):
            key = key if isinstance(key, tuple) else (key,)
            pivot = group.pivot_table(index="fold", columns="model", values=metric).dropna(axis=1, how="any")
            if pivot.shape[1] < 2:
                continue
            ranks = pivot.rank(axis=1, ascending=not higher_better)
            mean_rank = ranks.mean(axis=0)
            std_rank = ranks.std(axis=0)
            base = dict(zip(group_cols, key))
            for model in mean_rank.index:
                rows.append({
                    **base, "metric": metric, "model": model,
                    "mean_rank": mean_rank[model], "std_rank": std_rank[model],
                })
    return pd.DataFrame(rows)


def compute_friedman_significance(df, group_cols, metrics, alpha):
    rows = []
    for metric in metrics:
        for key, group in df.groupby(group_cols):
            key = key if isinstance(key, tuple) else (key,)
            pivot = group.pivot_table(index="fold", columns="model", values=metric).dropna(axis=1, how="any")
            n_models = pivot.shape[1]
            if n_models < 3:
                continue
            stat, p = friedmanchisquare(*[pivot[col] for col in pivot.columns])
            rows.append({
                **dict(zip(group_cols, key)), "metric": metric, "n_models": n_models,
                "statistic": stat, "p_value": p, "alpha": alpha, "significant": p < alpha,
            })
    return pd.DataFrame(rows)


def compute_within_family_significance(df, group_cols, metrics, alpha):
    rows = []
    for metric in metrics:
        for key, group in df.groupby(group_cols):
            key = key if isinstance(key, tuple) else (key,)
            pivot = group.pivot_table(index="fold", columns="model", values=metric).dropna(axis=1, how="any")
            n_models = pivot.shape[1]
            base = {**dict(zip(group_cols, key)), "metric": metric, "n_models": n_models, "alpha": alpha}
            if n_models >= 3:
                stat, p = friedmanchisquare(*[pivot[col] for col in pivot.columns])
                rows.append({**base, "test": "friedman", "statistic": stat, "p_value": p, "significant": p < alpha})
            elif n_models == 2:
                col_a, col_b = pivot.columns
                stat, p = wilcoxon(pivot[col_a], pivot[col_b])
                rows.append({**base, "test": "wilcoxon", "statistic": stat, "p_value": p, "significant": p < alpha})
    return pd.DataFrame(rows)


def _friedman_row(friedman_df, property_, split_strategy, metric):
    match = friedman_df[
        (friedman_df["property"] == property_) & (friedman_df["split_strategy"] == split_strategy)
        & (friedman_df["metric"] == metric)
    ]
    return match.iloc[0] if not match.empty else None

def _wrap_label(text, width=26):
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "<br>".join(lines)


BASE_COLORS = {
    "SVM":           "#FFE0B2",
    "XGBoost":       "#FFB74D",
    "Random Forest": "#FB8C00",
    "CatBoost":      "#E65100",
    "Chemeleon":     "#2E7D32",
    "ChemProp":      "#1565C0",
    "PepDoRA":       "#6A1B9A",
    "ChemBERTa":     "#AD1457",
}


def get_model_color(model_label):
    for key, color in BASE_COLORS.items():
        if key in model_label:
            return color
    return "black"


DEFAULT_METRIC_LABELS = {
    "rmse": "RMSE", "mae": "MAE", "r2": "R\u00b2",
    "pearson_r": "Pearson r", "spearman_r": "Spearman r",
}

PROPERTY_DISPLAY_LABELS = {
    "camsol": "Solubility (Non-natural CamSol-PTM)",
    "hemolysis": "Hemolytic Toxicity",
    "synthesizability": "Synthesisability",
}

FAMILY_DISPLAY_LABELS = {
    "classical": "Classical ML",
    "transformer": "Transformer",
    "chemprop": "ChemProp / Chemeleon",
}


def plot_rank_heatmap(mean_rank_df, friedman_df, split_strategy, properties, outfile,
                       metrics=("rmse", "mae", "pearson_r", "spearman_r"), top_n=15,
                       axes_size=16, axes_size_small=13, font_family="Aptos",
                       width=None, height=None, metric_labels=None, property_labels=None,
                       colorscale="RdYlGn_r", title=None):
    metric_labels = metric_labels or DEFAULT_METRIC_LABELS
    property_labels = property_labels or PROPERTY_DISPLAY_LABELS
    n_metrics = len(metrics)
    col_keys = [(p, m) for p in properties for m in metrics]

    pivot = mean_rank_df[
        (mean_rank_df["split_strategy"] == split_strategy)
        & (mean_rank_df["property"].isin(properties)) & (mean_rank_df["metric"].isin(metrics))
    ].pivot_table(index="model", columns=["property", "metric"], values="mean_rank")
    pivot = pivot.reindex(columns=col_keys)

    empty_cols = [key for key in col_keys if key not in pivot.columns or pivot[key].isna().all()]
    if empty_cols:
        print(f"warning: no data found for {empty_cols} at split_strategy='{split_strategy}' "
              f"-- these columns will render blank. Check mean_rank_df['property'].unique() "
              f"against the 'properties' argument for a name mismatch.")

    overall_order = pivot.mean(axis=1).sort_values().index
    pivot = pivot.loc[overall_order]
    if top_n:
        pivot = pivot.head(top_n)

    significant_cols = []
    p_values = []
    for p, m in col_keys:
        f_row = _friedman_row(friedman_df, p, split_strategy, m)
        significant_cols.append(bool(f_row is not None and bool(f_row["significant"])))
        p_values.append(f_row["p_value"] if f_row is not None else None)

    col_labels = [
        metric_labels.get(m, m) + ("*" if sig else "")
        for (p, m), sig in zip(col_keys, significant_cols)
    ]
    y_labels = [
        f'<span style="color:{get_model_color(m)}">{_wrap_label(m, 30)}</span>' for m in pivot.index
    ]

    n_cols_total, n_rows_total = len(col_keys), len(pivot)
    width = width or max(750, 80 * n_cols_total + 340)
    height = height or max(480, 42 * n_rows_total + 190)

    z = pivot.values
    x_pos, y_pos = list(range(n_cols_total)), list(range(n_rows_total))

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x=x_pos, y=y_pos,
        colorscale=colorscale, reversescale=False,
        xgap=3, ygap=3,
        colorbar=dict(title=dict(text="Mean rank<br>(1=best)", font=dict(size=axes_size_small, family=font_family)),
                       tickfont=dict(size=axes_size_small, family=font_family)),
    ))

    # cell value text
    finite_vals = z[~np.isnan(z)]
    vmin, vmax = (finite_vals.min(), finite_vals.max()) if finite_vals.size else (0.0, 1.0)
    span = (vmax - vmin) or 1.0
    text_x, text_y, text_vals, text_colors = [], [], [], []
    for yi in range(n_rows_total):
        for xi in range(n_cols_total):
            val = z[yi, xi]
            if np.isnan(val):
                continue
            norm = (val - vmin) / span
            text_x.append(xi)
            text_y.append(yi)
            text_vals.append(f"{val:.1f}")
            text_colors.append("white" if (norm < 0.18 or norm > 0.82) else "black")

    fig.add_trace(go.Scatter(
        x=text_x, y=text_y, mode="text", text=text_vals,
        textfont=dict(color=text_colors, size=axes_size_small, family=font_family),
        hoverinfo="skip", showlegend=False,
    ))

    # thick separators between property groups
    for i in range(1, len(properties)):
        fig.add_shape(
            type="line", xref="x", yref="paper",
            x0=i * n_metrics - 0.5, x1=i * n_metrics - 0.5, y0=0, y1=1,
            line=dict(color="black", width=3),
        )

    # p-value, once per column, italicized and small, just under the metric label
    for xi, pval in zip(x_pos, p_values):
        if pval is None:
            continue
        fig.add_annotation(
            text=f"<i>p = {pval:.3g}</i>", showarrow=False, xref="x", yref="paper",
            x=xi, y=-0.035, xanchor="center", yanchor="top",
            font=dict(size=axes_size_small - 3, family=font_family),
        )

    # property group headers, once per group, bold and close under the p-values
    for i, property_ in enumerate(properties):
        center = i * n_metrics + (n_metrics - 1) / 2
        label = property_labels.get(property_, property_)
        fig.add_annotation(
            text=f"<b>{label}</b>", showarrow=False, xref="x", yref="paper",
            x=center, y=-0.1, xanchor="center", yanchor="top",
            font=dict(size=axes_size, family=font_family),
        )

    fig.update_layout(
        title=dict(text=title or f"Mean Rank by Property and Metric ({split_strategy.title()} Split)",
                    font=dict(size=axes_size + 4, family=font_family), x=0.5, xanchor="center"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        width=width, height=height,
        margin=dict(t=70, l=20, r=20, b=100),
        font=dict(family=font_family),
        xaxis=dict(showline=True, linecolor="black", mirror=True, ticks="outside", showgrid=False,
                   tickmode="array", tickvals=x_pos, ticktext=col_labels,
                   tickfont=dict(size=axes_size_small, family=font_family), side="bottom"),
        yaxis=dict(showline=True, linecolor="black", mirror=True, ticks="outside", showgrid=False,
                   tickmode="array", tickvals=y_pos, ticktext=y_labels,
                   tickfont=dict(size=axes_size_small, family=font_family),
                   autorange="reversed", automargin=True),
    )

    fig.write_image(outfile)
    return fig


def plot_within_family_heatmap(within_family_df, split_strategy, properties, outfile,
                                metrics=("rmse", "mae", "pearson_r", "spearman_r"),
                                axes_size=16, axes_size_small=13, font_family="Aptos",
                                width=None, height=None, metric_labels=None, property_labels=None,
                                family_labels=None, family_order=None, title=None):
    metric_labels = metric_labels or DEFAULT_METRIC_LABELS
    property_labels = property_labels or PROPERTY_DISPLAY_LABELS
    family_labels = family_labels or FAMILY_DISPLAY_LABELS
    n_metrics = len(metrics)
    col_keys = [(p, m) for p in properties for m in metrics]

    sub = within_family_df[
        (within_family_df["split_strategy"] == split_strategy)
        & (within_family_df["property"].isin(properties)) & (within_family_df["metric"].isin(metrics))
    ]
    families = family_order or sorted(sub["family"].unique())

    z = np.full((len(families), len(col_keys)), np.nan)
    p_text = [["" for _ in col_keys] for _ in families]
    for fi, family in enumerate(families):
        for ci, (p, m) in enumerate(col_keys):
            row = sub[(sub["family"] == family) & (sub["property"] == p) & (sub["metric"] == m)]
            if row.empty:
                continue
            r = row.iloc[0]
            z[fi, ci] = 1.0 if bool(r["significant"]) else 0.0
            p_text[fi][ci] = f"p = {r['p_value']:.3g}"

    y_labels = [family_labels.get(f, f) for f in families]
    col_labels = [metric_labels.get(m, m) for (p, m) in col_keys]

    n_cols_total, n_rows_total = len(col_keys), len(families)
    width = width or max(750, 80 * n_cols_total + 340)
    height = height or max(320, 70 * n_rows_total + 190)

    x_pos, y_pos = list(range(n_cols_total)), list(range(n_rows_total))
    colorscale = [[0.0, "#EF9A9A"], [0.5, "#EF9A9A"], [0.5, "#A5D6A7"], [1.0, "#A5D6A7"]]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x=x_pos, y=y_pos,
        zmin=0, zmax=1,
        colorscale=colorscale, showscale=False,
        xgap=3, ygap=3,
    ))

    # cell text: significant/not significant plus the underlying p-value
    text_x, text_y, text_vals = [], [], []
    for yi in range(n_rows_total):
        for xi in range(n_cols_total):
            val = z[yi, xi]
            if np.isnan(val):
                continue
            text_x.append(xi)
            text_y.append(yi)
            text_vals.append(f"{p_text[yi][xi]}")

    fig.add_trace(go.Scatter(
        x=text_x, y=text_y, mode="text", text=text_vals,
        textfont=dict(color="black", size=axes_size_small - 2, family=font_family),
        hoverinfo="skip", showlegend=False,
    ))

    # thick separators between property groups
    for i in range(1, len(properties)):
        fig.add_shape(
            type="line", xref="x", yref="paper",
            x0=i * n_metrics - 0.5, x1=i * n_metrics - 0.5, y0=0, y1=1,
            line=dict(color="black", width=3),
        )

    # property group headers, once per group, bold and close under the axis
    for i, property_ in enumerate(properties):
        center = i * n_metrics + (n_metrics - 1) / 2
        label = property_labels.get(property_, property_)
        fig.add_annotation(
            text=f"<b>{label}</b>", showarrow=False, xref="x", yref="paper",
            x=center, y=-0.15, xanchor="center", yanchor="top",
            font=dict(size=axes_size, family=font_family),
        )

    fig.update_layout(
        title=dict(text=title or f"Within-Family Significance ({split_strategy.title()} Split)",
                    font=dict(size=axes_size + 4, family=font_family), x=0.5, xanchor="center"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        width=width, height=height,
        margin=dict(t=70, l=20, r=20, b=90),
        font=dict(family=font_family),
        xaxis=dict(showline=True, linecolor="black", mirror=True, ticks="outside", showgrid=False,
                   tickmode="array", tickvals=x_pos, ticktext=col_labels,
                   tickfont=dict(size=axes_size_small, family=font_family), side="bottom"),
        yaxis=dict(showline=True, linecolor="black", mirror=True, ticks="outside", showgrid=False,
                   tickmode="array", tickvals=y_pos, ticktext=y_labels,
                   tickfont=dict(size=axes_size_small, family=font_family),
                   autorange="reversed", automargin=True),
    )

    fig.write_image(outfile)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", default="compiled_folds.csv")
    parser.add_argument("--mean-ranks-csv", default="mean_ranks.csv")
    parser.add_argument("--friedman-csv", default="friedman_results.csv")
    parser.add_argument("--friedman-within-family-csv", default="friedman_within_family.csv")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    df = compile_folder(args.input_dir)
    df.to_csv(args.output_csv, index=False)

    group_cols = ["property", "split_strategy"]
    mean_ranks = compute_mean_ranks(df, group_cols, METRIC_KEYS)
    mean_ranks.to_csv(args.mean_ranks_csv, index=False)

    friedman_df = compute_friedman_significance(df, group_cols, METRIC_KEYS, args.alpha)
    friedman_df.to_csv(args.friedman_csv, index=False)

    within_family_group_cols = ["family", "property", "split_strategy"]
    friedman_within_family_df = compute_within_family_significance(df, within_family_group_cols, METRIC_KEYS, args.alpha)
    friedman_within_family_df.to_csv(args.friedman_within_family_csv, index=False)

    properties = ["camsol", "hemolysis", "synthesizability"]
    for split in ["random", "group"]:
        plot_rank_heatmap(mean_ranks, friedman_df, split, properties, f"heatmap_{split}.svg", top_n=args.top_n)
        plot_within_family_heatmap(friedman_within_family_df, split, properties, f"within_family_heatmap_{split}.svg")

if __name__ == "__main__":
    main()
