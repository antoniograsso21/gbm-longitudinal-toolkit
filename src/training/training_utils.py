"""
src/training/training_utils.py
================================
Shared utility functions for the LUMIERE training pipeline.

Changes from previous version:
    - select_features_fold_anchored_cached now routes through
      feature_selector.py wrapper (method-agnostic cache key)
    - Cache key includes 'method' from feature_selector.yaml so changing
      method automatically invalidates cached results
    - All other functions unchanged
"""
from __future__ import annotations

import hashlib
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Import wrapper (routes to MI or mRMR based on config)
from src.training.feature_selector import (
    FEATURE_SELECTOR_CONFIG,
    VARIANCE_THRESHOLD,
    load_feature_selector_config,
    select_features_fold_anchored,
)

# Re-export for callers that import these constants from training_utils
from src.training.feature_selector_mrmr import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_REPLICATES_FAST,
    MRMR_N_SELECT,
    MRMR_N_SELECT_FAST,
    STABILITY_THRESHOLD,
    STABILITY_THRESHOLD_FAST,
)

if TYPE_CHECKING:
    from src.training.feature_selector import SelectionResult


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

    Args:
        df_train:     training DataFrame (raw, unnormalised).
        df_test:      test DataFrame (raw, unnormalised).
        feature_cols: columns to select and normalise.

    Returns:
        Tuple (X_train_scaled, X_test_scaled) as numpy arrays.
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
    """Stratified train/val split for early stopping only."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_idx, val_idx = next(sss.split(X, y))
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_seed(config_path: str = "configs/random_state.yaml") -> int:
    seed, _ = load_random_config(config_path)
    return seed


def load_random_config(config_path: str = "configs/random_state.yaml") -> tuple[int, int]:
    """Load seed and n_jobs from random_state.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "seed" not in cfg:
        raise KeyError(f"'seed' key not found in {config_path}.")
    return int(cfg["seed"]), int(cfg.get("n_jobs", -1))


# ---------------------------------------------------------------------------
# Cached feature selection wrapper
# ---------------------------------------------------------------------------

FEATURE_SELECTION_CACHE_DIR = Path("data/processed/feature_selection_cache")


def select_features_fold_anchored_cached(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    fold: int,
    config_path: str = FEATURE_SELECTOR_CONFIG,
    **kwargs,
) -> "SelectionResult":
    """
    Cached wrapper around select_features_fold_anchored.

    Cache key includes method + all relevant parameters so changing method
    or any parameter automatically produces a different cache key.
    Delete data/processed/feature_selection_cache/ to force full recomputation.

    Args:
        X_train:     normalised training features DataFrame.
        y_train:     integer label array.
        fold:        fold index.
        config_path: path to feature_selector.yaml.
        **kwargs:    override any selector parameter.

    Returns:
        MIFoldSelectionResult or AnchoredFoldSelectionResult depending on method.
    """
    cfg = load_feature_selector_config(config_path)
    method = kwargs.get("method", cfg.get("method", "mi_univariate"))

    # Build cache key: method + relevant parameters + data fingerprint
    if method == "mi_univariate":
        cache_params = {
            "method": method,
            "fold": fold,
            "percentile": kwargs.get("percentile", cfg.get("percentile", 5.0)),
            "n_neighbors": kwargs.get("n_neighbors", cfg.get("n_neighbors", 5)),
            "seed": kwargs.get("seed", cfg.get("seed", 42)),
            "variance_threshold": VARIANCE_THRESHOLD,
        }
    else:  # mrmr
        fast = kwargs.get("fast", False)
        effective_B = kwargs.get("B", cfg.get("B", BOOTSTRAP_REPLICATES_FAST if fast else BOOTSTRAP_REPLICATES))
        effective_n_select = kwargs.get("n_select", cfg.get("n_select", MRMR_N_SELECT_FAST if fast else MRMR_N_SELECT))
        tau = kwargs.get("tau", cfg.get("tau", STABILITY_THRESHOLD))
        cache_params = {
            "method": method,
            "fold": fold,
            "B": effective_B,
            "n_select": effective_n_select,
            "tau": tau,
            "k_mi": kwargs.get("k_mi", cfg.get("k_mi", 3)),
            "seed": kwargs.get("seed", 42),
            "fast": fast,
            "variance_threshold": VARIANCE_THRESHOLD,
        }

    param_str = "_".join(f"{k}={v}" for k, v in sorted(cache_params.items()))
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    data_hash = hashlib.md5(
        X_train.values.tobytes() + y_train.tobytes()
    ).hexdigest()[:8]
    filename = f"{method}_fold{fold}_data{data_hash}_params{param_hash}.pkl"

    FEATURE_SELECTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FEATURE_SELECTION_CACHE_DIR / filename

    if cache_path.exists():
        print(f"  [cache] fold={fold} method={method} ♻️  loading from {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    result = select_features_fold_anchored(
        X_train=X_train,
        y_train=y_train,
        fold=fold,
        config_path=config_path,
        **kwargs,
    )

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  [cache] fold={fold} method={method} 💾  saved to {cache_path.name}")

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
    """Minimal run provenance for JSON outputs."""
    cfg = load_feature_selector_config()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": script_path,
        "seed": int(seed),
        "parquet": parquet_path,
        "n_rows": int(n_rows),
        "n_patients": int(n_patients),
        "feature_selector_method": cfg.get("method", "mi_univariate"),
        "feature_selector_justification": cfg.get("justification", ""),
    }
