"""
results.py
Shared results-CSV writer used by every finetune/evaluate script.
"""

from __future__ import annotations
import pandas as pd

RESULT_COLUMNS = [
    "run_timestamp", "run_date", "job_id", "model_arch", "run_name", "head",
    "dataset_name", "pipeline_mode", "subset_label", "split_strategy",
    "mean_rmse", "std_rmse", "mean_mae", "std_mae", "mean_r2", "std_r2",
    "mean_pearson_r", "std_pearson_r", "mean_spearman_r", "std_spearman_r",
    "rmse", "mae", "r2", "pearson_r", "spearman_r", "n_samples",
]

def append_results(rows: list[dict], csv_path) -> None:
    if not rows:
        raise ValueError("append_results() called with no rows")

    new_df = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    if file_exists and csv_path.stat().st_size > 0:
        with open(csv_path, "rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")

    new_df.to_csv(
        csv_path,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig" if not file_exists else "utf-8",
    )
