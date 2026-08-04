"""
utils.py
Shared utilities for train/val/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, GroupShuffleSplit, ShuffleSplit

# Core split helpers (single dataset / single pool)
def make_train_val_test_indices(
    groups,
    n_cv_folds: int,
    strategy: str,
    seed: int,
    test_frac: float = 0.10,
):
    """Set aside held-out test set, then produce CV folds on the remainder"""
    n = len(groups)
    groups = np.asarray(groups)
    all_idx = np.arange(n)

    #  Set aside test set
    if strategy == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        trainval_idx, test_idx = next(splitter.split(all_idx, groups=groups))
    elif strategy == "random":
        splitter = ShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        trainval_idx, test_idx = next(splitter.split(all_idx))
    else:
        raise ValueError(f"Unknown split strategy: '{strategy}'. Use 'group' or 'random'.")

    #  Make CV splits
    trainval_groups = groups[trainval_idx]
    dummy = np.zeros(len(trainval_idx))

    train_folds, val_folds = [], []

    if strategy == "group":
        cv = GroupKFold(n_splits=n_cv_folds)
        for tr, va in cv.split(dummy, groups=trainval_groups):
            train_folds.append(trainval_idx[tr])
            val_folds.append(trainval_idx[va])
    else:
        cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
        for tr, va in cv.split(dummy):
            train_folds.append(trainval_idx[tr])
            val_folds.append(trainval_idx[va])

    return train_folds, val_folds, test_idx

#  Multihead: per-dataset split, unioned per fold to match with single-task splits
def make_multihead_splits(
    datasets: dict,
    smiles_col: str,
    cluster_col: str,
    n_folds: int,
    strategy: str,
    seed: int,
    test_frac: float = 0.10,
):
    per_dataset = {}
    for name, df in datasets.items():
        groups = df[cluster_col].values
        train_folds, val_folds, test_idx = make_train_val_test_indices(
            groups, n_folds, strategy, seed, test_frac=test_frac)
        per_dataset[name] = {
            "train_folds": train_folds,
            "val_folds":   val_folds,
            "test_idx":    test_idx,
            "smiles":      df[smiles_col].values,
        }

    fold_train_smiles, fold_val_smiles = [], []
    for fold_i in range(n_folds):
        train_s, val_s = set(), set()
        for d in per_dataset.values():
            train_s.update(d["smiles"][d["train_folds"][fold_i]].tolist())
            val_s.update(d["smiles"][d["val_folds"][fold_i]].tolist())
        fold_train_smiles.append(train_s)
        fold_val_smiles.append(val_s)

    test_smiles = set()
    for d in per_dataset.values():
        test_smiles.update(d["smiles"][d["test_idx"]].tolist())

    return per_dataset, fold_train_smiles, fold_val_smiles, test_smiles

def resolve_multihead_indices(
    merged_smiles_array, fold_train_smiles, fold_val_smiles, test_smiles, n_folds
):
    """Map the per-fold SMILES sets from make_multihead_splits() on merged dataframe"""
    merged_smiles_array = np.asarray(merged_smiles_array)
    is_test  = np.array([s in test_smiles for s in merged_smiles_array])
    test_idx = np.where(is_test)[0]

    train_folds, val_folds = [], []
    for fold_i in range(n_folds):
        val_mask   = np.array([s in fold_val_smiles[fold_i]   for s in merged_smiles_array])
        train_mask = np.array([s in fold_train_smiles[fold_i] for s in merged_smiles_array])
        val_folds.append(np.where(val_mask & ~is_test)[0])
        train_folds.append(np.where(train_mask & ~is_test & ~val_mask)[0])

    return train_folds, val_folds, test_idx

