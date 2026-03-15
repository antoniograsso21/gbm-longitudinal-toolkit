"""
src/training/training_utils.py
================================
Shared utility functions for the LUMIERE training pipeline.

These functions are used by all run_*.py entry points (LR, LightGBM, LSTM)
and are centralised here to prevent divergence across models.

Deliberately separate from src/utils/lumiere_io.py — that module handles
LUMIERE-specific I/O and constants. This module handles training mechanics
(normalisation, splits, config loading) that are dataset-agnostic.

Functions:
    fit_transform_fold  — StandardScaler fit on train fold only
    split_train_val     — stratified train/val split for early stopping
    load_seed           — load random seed from YAML
    load_random_config  — load seed + n_jobs from YAML
"""

import numpy as np
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import pandas as pd


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def fit_transform_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit StandardScaler on train fold only, transform both splits.

    Scaler is fit exclusively on the training fold — never on the full
    dataset or the test fold. This is enforced structurally: the scaler
    object is local to this function and never returned.

    Args:
        df_train:     training DataFrame (raw, unnormalised).
        df_test:      test DataFrame (raw, unnormalised).
        feature_cols: columns to select and normalise.

    Returns:
        Tuple (X_train_scaled, X_test_scaled) as numpy arrays.

    Raises:
        ValueError: if any feature_col is absent from df_train or df_test.
    """
    missing_train = [c for c in feature_cols if c not in df_train.columns]
    missing_test  = [c for c in feature_cols if c not in df_test.columns]
    if missing_train:
        raise ValueError(f"Columns missing from df_train: {missing_train}")
    if missing_test:
        raise ValueError(f"Columns missing from df_test: {missing_test}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[feature_cols])
    X_test_scaled  = scaler.transform(df_test[feature_cols])
    return X_train_scaled, X_test_scaled


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/val split for early stopping.

    The val set is NEVER used for metric evaluation — only for early stopping
    in LightGBM and LSTM. Stratified to preserve class balance in the small
    val set (~18 examples on n=185 train fold).

    Args:
        X:            feature array, shape (n, d).
        y:            integer label array, shape (n,).
        val_fraction: fraction of data to use as val (default 0.1).
        seed:         random seed.

    Returns:
        Tuple (X_tr, y_tr, X_val, y_val).
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_idx, val_idx = next(sss.split(X, y))
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_seed(config_path: str = "configs/random_state.yaml") -> int:
    """
    Load the global random seed from random_state.yaml.

    Convenience wrapper around load_random_config.

    Args:
        config_path: path to random_state.yaml.

    Returns:
        Integer seed value.

    Raises:
        KeyError: if 'seed' key is missing.
        FileNotFoundError: if config_path does not exist.
    """
    seed, _ = load_random_config(config_path)
    return seed


def load_random_config(config_path: str = "configs/random_state.yaml") -> tuple[int, int]:
    """
    Load seed and n_jobs from random_state.yaml.

    Used by all run_*.py entry points. Centralised here (DRY) to prevent
    divergence across models.

    Args:
        config_path: path to random_state.yaml.

    Returns:
        Tuple (seed, n_jobs).
        n_jobs defaults to -1 if not present in the YAML.

    Raises:
        KeyError: if 'seed' key is missing.
        FileNotFoundError: if config_path does not exist.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "seed" not in cfg:
        raise KeyError(
            f"'seed' key not found in {config_path}. "
            "Expected format: 'seed: 42'"
        )
    return int(cfg["seed"]), int(cfg.get("n_jobs", -1))
