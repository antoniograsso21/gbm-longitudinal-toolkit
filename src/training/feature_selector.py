"""
src/training/feature_selector.py
==================================
Unified feature selector wrapper — routes to MI univariate or mRMR
based on the method specified in configs/feature_selector.yaml.

This module is the single import point for all run_*.py callers.
The underlying selectors live in:
    feature_selector_mi.py   — production path (MI univariate, stable on small n)
    feature_selector_mrmr.py — reference path (mRMR + stability, large n)

Decision rule (from diagnostic analysis on LUMIERE):
    ρ < 0.30  → use mi_univariate
    ρ ≥ 0.60  → mRMR is reliable
    0.30–0.60 → borderline; prefer mi_univariate and declare in Methods

The method is fixed in configs/feature_selector.yaml and does not change
between folds. Changing method mid-pipeline invalidates the feature
selection cache — delete data/processed/feature_selection_cache/ after
any method change.

Backward compatibility:
    AnchoredFoldSelectionResult and FoldSelectionResult are re-exported
    from this module so existing callers (run_lgbm_baseline.py, etc.)
    require zero changes to their imports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml

# Re-export for backward compatibility — callers import from here
from src.training.feature_selector_mrmr import (
    AnchoredFoldSelectionResult,
    AggregatedSelection,
    FoldSelectionResult,
    aggregate_fold_selections,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_REPLICATES_FAST,
    MRMR_N_SELECT,
    MRMR_N_SELECT_FAST,
    STABILITY_THRESHOLD,
    STABILITY_THRESHOLD_FAST,
    VARIANCE_THRESHOLD,
)
from src.training.feature_selector_mi import MIFoldSelectionResult

logger = logging.getLogger(__name__)

# Default config path
FEATURE_SELECTOR_CONFIG: str = "configs/feature_selector.yaml"

# Union type for return — both result types have the same fields accessed by callers
SelectionResult = Union[AnchoredFoldSelectionResult, MIFoldSelectionResult]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_feature_selector_config(
    config_path: str = FEATURE_SELECTOR_CONFIG,
) -> dict:
    """
    Load feature selector configuration from YAML.

    Returns defaults if file not found (backward compatibility).
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(
            f"{config_path} not found — using defaults: method=mi_univariate, "
            f"percentile=5.0, n_neighbors=5"
        )
        return {
            "method": "mi_univariate",
            "percentile": 5.0,
            "n_neighbors": 5,
            "justification": "",
        }
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def select_features_fold_anchored(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    fold: int,
    config_path: str = FEATURE_SELECTOR_CONFIG,
    **kwargs,
) -> SelectionResult:
    """
    Select features for a single CV fold using the configured method.

    Reads method from configs/feature_selector.yaml. All method-specific
    parameters (percentile, n_neighbors, tau, B, k_mi) are taken from the
    config file; kwargs override config values if provided.

    Args:
        X_train:     normalised training features DataFrame.
        y_train:     integer label array.
        fold:        fold index for traceability.
        config_path: path to feature_selector.yaml.
        **kwargs:    override any config parameter.

    Returns:
        MIFoldSelectionResult  (method=mi_univariate)
        AnchoredFoldSelectionResult  (method=mrmr)

    Both types expose: selected_radiomic, full_feature_set, n_radiomic_selected,
    n_delta_anchored — all fields accessed by run_*.py callers.
    """
    cfg = load_feature_selector_config(config_path)
    method = kwargs.pop("method", cfg.get("method", "mi_univariate"))
    justification = cfg.get("justification", "")

    logger.info(
        f"[feature_selector fold={fold}] method={method} | {justification}"
    )

    if method == "mi_univariate":
        from src.training.feature_selector_mi import select_features_fold_mi

        percentile = kwargs.pop("percentile", cfg.get("percentile", 5.0))
        n_neighbors = kwargs.pop("n_neighbors", cfg.get("n_neighbors", 5))
        seed = kwargs.pop("seed", cfg.get("seed", 42))
        verbose = kwargs.pop("verbose", False)

        return select_features_fold_mi(
            X_train=X_train,
            y_train=y_train,
            fold=fold,
            percentile=percentile,
            n_neighbors=n_neighbors,
            seed=seed,
            verbose=verbose,
            justification=justification,
        )

    elif method == "mrmr":
        from src.training.feature_selector_mrmr import select_features_fold_anchored_mrmr

        # Pass all mRMR-specific kwargs through
        return select_features_fold_anchored_mrmr(
            X_train=X_train,
            y_train=y_train,
            fold=fold,
            n_select=kwargs.pop("n_select", cfg.get("n_select", MRMR_N_SELECT)),
            B=kwargs.pop("B", cfg.get("B", BOOTSTRAP_REPLICATES)),
            tau=kwargs.pop("tau", cfg.get("tau", STABILITY_THRESHOLD)),
            k_mi=kwargs.pop("k_mi", cfg.get("k_mi", 3)),
            seed=kwargs.pop("seed", cfg.get("seed", 42)),
            fast=kwargs.pop("fast", False),
            n_jobs=kwargs.pop("n_jobs", -1),
            verbose=kwargs.pop("verbose", False),
            check_consistency=cfg.get("check_consistency", True),
        )

    else:
        raise ValueError(
            f"Unknown feature selection method: '{method}'. "
            "Must be 'mi_univariate' or 'mrmr'. "
            f"Check {config_path}."
        )
