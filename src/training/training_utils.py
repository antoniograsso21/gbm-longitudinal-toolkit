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
from __future__ import annotations
import numpy as np
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import pandas as pd

import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path

from typing import TYPE_CHECKING

from src.training.feature_selector import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_REPLICATES_FAST,
    MRMR_N_SELECT,
    MRMR_N_SELECT_FAST,
    STABILITY_THRESHOLD,
    STABILITY_THRESHOLD_FAST,
    VARIANCE_THRESHOLD,
    select_features_fold_anchored
)

if TYPE_CHECKING:
    from src.training.feature_selector import (
        AnchoredFoldSelectionResult
    )

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


# ---------------------------------------------------------------------------
# Cached feature selection wrapper
# ---------------------------------------------------------------------------

FEATURE_SELECTION_CACHE_DIR = Path("data/processed/feature_selection_cache")


def select_features_fold_anchored_cached(
    X_train: "pd.DataFrame",
    y_train: "np.ndarray",
    fold: int,
    **kwargs,
) -> "AnchoredFoldSelectionResult":
    """
    Cached wrapper around select_features_fold_anchored.

    Computes a deterministic cache key from data content + parameters.
    On cache hit: loads and returns the cached result.
    On cache miss: runs selection, saves result, returns it.

    Cache lives in data/processed/feature_selection_cache/ — excluded from
    DVC tracking (add to .dvcignore). Safe to delete to force recomputation.

    Side effects are intentional here — this is a utility wrapper,
    not a pure function. Pure logic lives in feature_selector.py.
    {'seed': 42, 'fast': False, 'n_jobs': 6, 'verbose': True}
     cache_kwargs = {
        "fold": fold, "n_select": effective_n_select, "B": effective_B,
        "tau": effective_tau, "k_mi": k_mi, "seed": seed, "fast": fast,
        "variance_threshold": VARIANCE_THRESHOLD
    }
    """
    # Build cache key from parameters + data fingerprint
    fast = kwargs.get("fast", False)
    seed = kwargs.get("seed", 42)
    
    effective_B = kwargs.get("B") or (BOOTSTRAP_REPLICATES_FAST if fast else BOOTSTRAP_REPLICATES)
    effective_n_select = kwargs.get("n_select") or (MRMR_N_SELECT_FAST if fast else MRMR_N_SELECT)
    tau = kwargs.get("tau", STABILITY_THRESHOLD)
    effective_tau = (STABILITY_THRESHOLD_FAST if fast and tau == STABILITY_THRESHOLD else tau)
    
    cache_kwargs = {
        "fold": fold,
        "B": effective_B,
        "n_select": effective_n_select,
        "tau": effective_tau,
        "k_mi": kwargs.get("k_mi", 3),
        "seed": seed,
        "fast": fast,
        "variance_threshold": VARIANCE_THRESHOLD,
    }
    param_str = "_".join(f"{k}={v}" for k, v in sorted(cache_kwargs.items()))
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    data_hash = hashlib.md5(
        X_train.values.tobytes() + y_train.tobytes()
    ).hexdigest()[:8]
    filename = f"anchored_fold{fold}_data{data_hash}_params{param_hash}.pkl"
    print(filename)

    FEATURE_SELECTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FEATURE_SELECTION_CACHE_DIR / filename

    if cache_path.exists():
        print(f"  [cache] fold={fold} ♻️  loading from {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    result: AnchoredFoldSelectionResult = select_features_fold_anchored(
        X_train=X_train, y_train=y_train, fold=fold, **kwargs
    )

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  [cache] fold={fold} 💾  saved to {cache_path.name}")

    return result


# ---------------------------------------------------------------------------
# Run metadata helpers (pure)
# ---------------------------------------------------------------------------

def build_run_info(
    *,
    seed: int,
    parquet_path: str,
    n_rows: int,
    n_patients: int,
    script_path: str,
) -> dict:
    """
    Minimal run provenance for JSON outputs.

    Intentionally lightweight: MLflow carries the full parameter surface area.
    JSON stores only enough to tie a results file to a concrete dataset snapshot
    and execution context.
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": script_path,
        "seed": int(seed),
        "parquet": parquet_path,
        "n_rows": int(n_rows),
        "n_patients": int(n_patients),
    }