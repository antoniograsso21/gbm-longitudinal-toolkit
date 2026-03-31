"""
src/training/feature_selector_mi.py
=====================================
Univariate Mutual Information feature selector — production path for LUMIERE.

Replaces mRMR + Stability Selection as the primary selector after diagnostic
analysis revealed low mRMR rank consistency (Spearman ρ=0.226) on n~93 per
bootstrap replicate. See diagnose_feature_selection_report.json for details.

Algorithm:
    1. Variance threshold (label-free, same as mRMR path)
    2. sklearn mutual_info_classif on radiomic-only subset
       - n_neighbors=5 for reduced variance on small n (vs default 3)
       - Avoids discretisation bias of ANOVA/F-test on continuous radiomic features
    3. Select top percentile% by MI score
    4. Anchored delta: include delta_f for each selected radiomic f
    5. full_feature_set = selected_radiomic + temporal + nadir + anchored_delta + delta_derived

Design rationale (declare in paper Methods):
    mRMR was considered but discarded due to high rank instability
    (Spearman ρ=0.226 across bootstrap replicates at n~93 per replicate),
    which would have introduced selection noise rather than biological signal.
    Univariate MI is a standard radiomics filter (consistent with IBSI guidelines),
    stable on small n, and leaves multivariate interaction learning to the model.

Design:
    All functions are pure — no side effects, no file I/O.
    The cached wrapper lives in training_utils.py (shared with mRMR path).
    Caller (run_*.py) owns configuration loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (defaults — overridden by feature_selector.yaml via caller)
# ---------------------------------------------------------------------------

PERCENTILE: float = 5.0          # top 5% of radiomic features by MI score
N_NEIGHBORS: int = 5             # k for MI estimator — 5 for stability on small n
VARIANCE_THRESHOLD: float = 1e-6

TEMPORAL_COLS: frozenset[str] = frozenset({
    "interval_weeks", "scan_index", "time_from_diagnosis_weeks",
})

NADIR_COLS: frozenset[str] = frozenset({
    "CE_vs_nadir",
    "weeks_since_nadir",
})

DELTA_DERIVED_COLS: frozenset[str] = frozenset({
    "delta_CE_NC_ratio",
    "delta_CE_vs_nadir",
})

RADIOMIC_PREFIX: str = "original_"


# ---------------------------------------------------------------------------
# Typed result (identical structure to AnchoredFoldSelectionResult for
# drop-in compatibility with all run_*.py callers)
# ---------------------------------------------------------------------------

@dataclass
class MIFoldSelectionResult:
    """
    Feature selection result from univariate MI selector.

    Compatible with AnchoredFoldSelectionResult: all run_*.py callers
    access the same fields (selected_radiomic, full_feature_set, etc.).
    """
    fold: int
    selected_radiomic: list[str]
    anchored_delta: list[str]
    temporal_cols: list[str]
    full_feature_set: list[str]
    mi_scores: dict[str, float]        # MI score per radiomic candidate (all, not just selected)
    mi_threshold: float                # actual MI cutoff used
    n_radiomic_candidates: int
    n_radiomic_selected: int
    n_delta_anchored: int
    percentile_used: float
    n_neighbors_used: int
    method: str = "mi_univariate"


# ---------------------------------------------------------------------------
# Core selection function (pure)
# ---------------------------------------------------------------------------

def select_features_fold_mi(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    fold: int,
    percentile: float = PERCENTILE,
    n_neighbors: int = N_NEIGHBORS,
    seed: int = 42,
    verbose: bool = False,
    justification: str = "",
) -> MIFoldSelectionResult:
    """
    Select radiomic features using univariate Mutual Information.

    Follows the same 4-step flow as the mRMR path for architectural consistency:
        1. Variance threshold on full set (label-free)
        2. MI scoring on radiomic-only subset → selected_radiomic
        3. anchored_delta = {delta_f : f in selected_radiomic AND variance OK}
        4. full_feature_set = selected_radiomic + temporal + nadir
                              + anchored_delta + delta_derived

    Args:
        X_train:       normalised training features, DataFrame with named columns.
                       Must be a DataFrame — column names drive all filtering.
        y_train:       integer label array, shape (n_train,).
        fold:          fold index for traceability.
        percentile:    fraction of top radiomic features to select (default 5%).
                       5% of 1284 ≈ 64 features on LUMIERE.
        n_neighbors:   k for sklearn MI estimator (default 5 for small-n stability).
        seed:          random_state for MI estimator (controls jitter in ties).
        verbose:       if True, print top-20 features by MI score.
        justification: free-text reason for method choice — logged at INFO level.
                       Passed from feature_selector.yaml for paper traceability.

    Returns:
        MIFoldSelectionResult with all selection details.

    Raises:
        ValueError: if X_train has no radiomic columns after filtering.
        ValueError: if percentile produces zero selected features.
    """
    method_tag = f"[MI fold={fold}]"

    if justification:
        logger.info(f"{method_tag} method=mi_univariate | {justification}")
    else:
        logger.info(f"{method_tag} method=mi_univariate | n_neighbors={n_neighbors} | percentile={percentile}%")

    all_feature_names = list(X_train.columns)
    X_full = X_train.values.astype(float)

    # Step 1 — variance threshold (label-free)
    variances = X_full.var(axis=0)
    kept_mask = variances > VARIANCE_THRESHOLD
    n_dropped = int((~kept_mask).sum())
    variance_passed: set[str] = {n for n, k in zip(all_feature_names, kept_mask) if k}
    if n_dropped > 0:
        logger.info(f"{method_tag} variance filter: {n_dropped} removed")

    # Step 2 — MI scoring on radiomic-only subset
    radiomic_names = [
        f for f in all_feature_names
        if f in variance_passed
        and not f.startswith("delta_")
        and f not in TEMPORAL_COLS
        and f not in NADIR_COLS
        and f not in DELTA_DERIVED_COLS
        and RADIOMIC_PREFIX in f
    ]

    if not radiomic_names:
        raise ValueError(
            f"{method_tag} No radiomic candidates after variance filter. "
            "Check preprocessing."
        )

    radiomic_indices = [all_feature_names.index(f) for f in radiomic_names]
    X_radiomic = X_full[:, radiomic_indices]

    mi_scores_array = mutual_info_classif(
        X_radiomic,
        y_train,
        n_neighbors=n_neighbors,
        random_state=seed,
    )
    mi_scores: dict[str, float] = {
        name: float(score)
        for name, score in zip(radiomic_names, mi_scores_array)
    }

    # Select top percentile%
    n_select = max(1, int(np.ceil(len(radiomic_names) * percentile / 100.0)))
    threshold = np.percentile(mi_scores_array, 100.0 - percentile)
    # Use argsort to get exactly n_select features (avoids tie ambiguity at threshold)
    sorted_idx = np.argsort(mi_scores_array)[::-1][:n_select]
    selected_radiomic = [radiomic_names[i] for i in sorted_idx]

    actual_threshold = float(mi_scores_array[sorted_idx[-1]])

    logger.info(
        f"{method_tag} radiomic: {len(radiomic_names)} candidates → "
        f"{len(selected_radiomic)} selected (top {percentile}%, MI threshold={actual_threshold:.4f})"
    )

    if verbose:
        print(f"\n  {method_tag} Top-20 radiomic features by MI score:")
        print(f"  {'Rank':>4}  {'MI Score':>8}  Feature")
        print(f"  {'─'*4}  {'─'*8}  {'─'*50}")
        for rank, idx in enumerate(sorted_idx[:20], 1):
            fname = radiomic_names[idx]
            score = mi_scores_array[idx]
            marker = "✅" if rank <= n_select else "  "
            print(f"  {marker}{rank:>3}  {score:>8.4f}  {fname[:60]}")

    # Step 3 — anchor delta features
    anchored_delta = [
        f"delta_{f}" for f in selected_radiomic
        if f"delta_{f}" in variance_passed
    ]
    logger.info(f"{method_tag} anchored delta: {len(anchored_delta)}")

    # Step 4 — build full feature set
    temporal = [c for c in all_feature_names if c in TEMPORAL_COLS]
    nadir = [c for c in all_feature_names if c in NADIR_COLS and c in variance_passed]
    delta_derived = [c for c in all_feature_names if c in DELTA_DERIVED_COLS and c in variance_passed]
    full_feature_set = (
        selected_radiomic
        + sorted(temporal)
        + sorted(nadir)
        + sorted(anchored_delta)
        + sorted(delta_derived)
    )

    logger.info(
        f"{method_tag} nadir={len(nadir)} | delta_derived={len(delta_derived)} | "
        f"total full_feature_set={len(full_feature_set)}"
    )

    return MIFoldSelectionResult(
        fold=fold,
        selected_radiomic=selected_radiomic,
        anchored_delta=anchored_delta,
        temporal_cols=temporal,
        full_feature_set=full_feature_set,
        mi_scores=mi_scores,
        mi_threshold=actual_threshold,
        n_radiomic_candidates=len(radiomic_names),
        n_radiomic_selected=len(selected_radiomic),
        n_delta_anchored=len(anchored_delta),
        percentile_used=percentile,
        n_neighbors_used=n_neighbors,
    )
