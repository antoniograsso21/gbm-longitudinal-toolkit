"""
src/training/feature_selector_mrmr.py
=======================================
mRMR + Stability Selection — reference implementation.

This module is the original feature_selector.py refactored with:
    - k_mi as a runtime parameter (was hardcoded to 3)
    - Automatic rank-consistency check (Spearman ρ) before full bootstrap run
    - Warning if ρ < 0.3: mRMR orderings are unstable, switch to MI univariate
    - All other logic unchanged for reproducibility of past runs

Intended use:
    - Datasets with n >> 200 per fold where Kraskov MI is reliable
    - Research exploration of multivariate interaction-aware selection
    - NOT recommended for LUMIERE (n~185/fold, ρ=0.226 — use feature_selector_mi.py)

Decision rule (declare in paper if used):
    ρ < 0.30  → use mi_univariate (feature_selector_mi.py)
    ρ ≥ 0.30  → mRMR may be used; declare ρ in Methods

Design:
    All functions are pure — no side effects, no file I/O.
    Cached wrapper lives in training_utils.py.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOOTSTRAP_REPLICATES: int = 100
BOOTSTRAP_REPLICATES_FAST: int = 10
STABILITY_THRESHOLD: float = 0.4
STABILITY_THRESHOLD_FAST: float = 0.3
MRMR_N_SELECT: int = 50
MRMR_N_SELECT_FAST: int = 20
FOLD_MAJORITY_THRESHOLD: int = 3
VARIANCE_THRESHOLD: float = 1e-6
RHO_CONSISTENCY_THRESHOLD: float = 0.30   # below this → warn and recommend MI path
RHO_PROBE_REPLICATES: int = 10            # fast probe before full bootstrap

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
# Typed results
# ---------------------------------------------------------------------------

@dataclass
class FoldSelectionResult:
    """Radiomic-only selection result — used for YAML aggregation."""
    fold: int
    selected_features: list[str]
    bootstrap_stability: dict[str, float]
    n_candidates: int
    n_selected: int
    fast_mode: bool


@dataclass
class AnchoredFoldSelectionResult:
    """Full feature selection result with anchored delta features."""
    fold: int
    selected_radiomic: list[str]
    anchored_delta: list[str]
    temporal_cols: list[str]
    full_feature_set: list[str]
    bootstrap_stability: dict[str, float]
    n_radiomic_candidates: int
    n_radiomic_selected: int
    n_delta_anchored: int
    fast_mode: bool
    rank_consistency_rho: float | None = None   # Spearman ρ from probe run
    method: str = "mrmr"


@dataclass
class AggregatedSelection:
    """Aggregated selection across all CV folds via majority vote."""
    selected_features: list[str]
    fold_stability: dict[str, float]
    fold_results: list[FoldSelectionResult]


# ---------------------------------------------------------------------------
# MI estimation (pure)
# ---------------------------------------------------------------------------

def _compute_mi_feature_target(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    from npeet.entropy_estimators import mi as kraskov_mi
    estimate = kraskov_mi(
        x.reshape(-1, 1).astype(float),
        y.reshape(-1, 1).astype(float),
        k=k,
    )
    return max(0.0, float(estimate))


def _compute_mi_feature_feature(xi: np.ndarray, xj: np.ndarray, k: int = 3) -> float:
    from npeet.entropy_estimators import mi as kraskov_mi
    estimate = kraskov_mi(
        xi.reshape(-1, 1).astype(float),
        xj.reshape(-1, 1).astype(float),
        k=k,
    )
    return max(0.0, float(estimate))


# ---------------------------------------------------------------------------
# mRMR (pure)
# ---------------------------------------------------------------------------

def run_mrmr(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_select: int = MRMR_N_SELECT,
    k_mi: int = 3,
) -> list[str]:
    """
    Select top-n features using mRMR (Maximum Relevance Minimum Redundancy).

    Args:
        k_mi: k for Kraskov MI estimator. Use 5–7 for n < 200 to reduce
              estimation variance. Default 3 (original paper value).
    """
    n_features = X.shape[1]
    if len(feature_names) != n_features:
        raise ValueError(f"feature_names length {len(feature_names)} != X columns {n_features}.")
    if n_select > n_features:
        raise ValueError(f"n_select={n_select} exceeds n_features={n_features}.")

    relevance = np.array([
        _compute_mi_feature_target(X[:, i], y, k=k_mi)
        for i in range(n_features)
    ])

    selected_indices: list[int] = []
    remaining: set[int] = set(range(n_features))
    mi_cache: dict[tuple[int, int], float] = {}

    for _ in range(n_select):
        best_idx, best_score = -1, -np.inf
        for idx in remaining:
            if not selected_indices:
                score = relevance[idx]
            else:
                redundancy_vals = []
                for j in selected_indices:
                    key = (min(idx, j), max(idx, j))
                    if key not in mi_cache:
                        mi_cache[key] = _compute_mi_feature_feature(X[:, idx], X[:, j], k=k_mi)
                    redundancy_vals.append(mi_cache[key])
                score = relevance[idx] - float(np.mean(redundancy_vals))
            if score > best_score:
                best_score, best_idx = score, idx
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [feature_names[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Rank consistency probe (fast pre-check before full bootstrap)
# ---------------------------------------------------------------------------

def _probe_rank_consistency(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_select: int,
    k_mi: int,
    seed: int,
    n_probe: int = RHO_PROBE_REPLICATES,
) -> float:
    """
    Run n_probe bootstrap replicates and compute mean Spearman ρ of mRMR rankings.

    Returns mean ρ across replicate pairs. Used to decide whether full bootstrap
    is worth running or whether MI univariate should be used instead.

    Returns:
        float: mean Spearman ρ in [-1, 1]. Values < RHO_CONSISTENCY_THRESHOLD
               indicate mRMR rankings are unstable on this dataset/fold.
    """
    from scipy.stats import spearmanr

    sss = StratifiedShuffleSplit(n_splits=n_probe, test_size=0.5, random_state=seed)
    replicate_indices = [idx for idx, _ in sss.split(X, y)]

    rankings: list[list[str]] = []
    for rep_idx, idx in enumerate(replicate_indices):
        np.random.seed(seed + rep_idx)
        selected = run_mrmr(X[idx], y[idx], feature_names, n_select=n_select, k_mi=k_mi)
        rankings.append(selected)

    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    rank_matrix = np.full((len(feature_names), n_probe), np.nan)
    for rep_idx, sel in enumerate(rankings):
        for rank, feat in enumerate(sel):
            rank_matrix[feat_to_idx[feat], rep_idx] = rank

    rho_vals = []
    for a in range(n_probe):
        for b in range(a + 1, n_probe):
            mask = ~np.isnan(rank_matrix[:, a]) & ~np.isnan(rank_matrix[:, b])
            if mask.sum() < 5:
                continue
            rho, _ = spearmanr(rank_matrix[mask, a], rank_matrix[mask, b])
            rho_vals.append(float(rho))

    return float(np.mean(rho_vals)) if rho_vals else 0.0


# ---------------------------------------------------------------------------
# Stability Selection (pure)
# ---------------------------------------------------------------------------

def run_stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_select: int = MRMR_N_SELECT,
    B: int = BOOTSTRAP_REPLICATES,
    tau: float = STABILITY_THRESHOLD,
    k_mi: int = 3,
    seed: int = 42,
    n_jobs: int = -1,
) -> tuple[list[str], dict[str, float]]:
    """
    Run Stability Selection over B bootstrap replicates.

    Uses StratifiedShuffleSplit to preserve class distribution per replicate.
    Each replicate uses np.random.seed(seed + rep_idx) for reproducibility.
    """
    sss = StratifiedShuffleSplit(n_splits=B, test_size=0.5, random_state=seed)
    replicate_indices = [idx for idx, _ in sss.split(X, y)]

    def _run_replicate(idx: np.ndarray, rep_idx: int) -> list[str]:
        np.random.seed(seed + rep_idx)
        return run_mrmr(X[idx], y[idx], feature_names, n_select=n_select, k_mi=k_mi)

    results: list[list[str]] = Parallel(n_jobs=n_jobs)(
        delayed(_run_replicate)(idx, i)
        for i, idx in enumerate(tqdm(replicate_indices, desc=f"  Bootstrap (B={B})", leave=False))
    )

    selection_counts: dict[str, int] = {name: 0 for name in feature_names}
    for selected in results:
        for name in selected:
            selection_counts[name] += 1

    bootstrap_stability = {name: count / B for name, count in selection_counts.items()}
    stable_features = [name for name, score in bootstrap_stability.items() if score >= tau]

    return stable_features, bootstrap_stability


# ---------------------------------------------------------------------------
# Main entry point (pure)
# ---------------------------------------------------------------------------

def select_features_fold_anchored_mrmr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    fold: int,
    n_select: int | None = None,
    B: int | None = None,
    tau: float = STABILITY_THRESHOLD,
    k_mi: int = 3,
    seed: int = 42,
    fast: bool = False,
    n_jobs: int = -1,
    verbose: bool = False,
    check_consistency: bool = True,
) -> AnchoredFoldSelectionResult:
    """
    mRMR + Stability Selection with rank-consistency pre-check.

    Args:
        k_mi:              k for Kraskov MI estimator. Default 3 (original).
                           Use 5–7 for n < 200 to reduce estimation variance.
        check_consistency: if True, run a fast probe (10 replicates) to compute
                           Spearman ρ before the full bootstrap. If ρ < 0.30,
                           emit a warning recommending MI univariate path.
                           Does not abort — caller decides.

    Returns:
        AnchoredFoldSelectionResult with rank_consistency_rho field populated
        if check_consistency=True.
    """
    if fast:
        effective_B = B if B is not None else BOOTSTRAP_REPLICATES_FAST
        effective_n_select = n_select if n_select is not None else MRMR_N_SELECT_FAST
        effective_tau = tau if tau != STABILITY_THRESHOLD else STABILITY_THRESHOLD_FAST
    else:
        effective_B = B if B is not None else BOOTSTRAP_REPLICATES
        effective_n_select = n_select if n_select is not None else MRMR_N_SELECT
        effective_tau = tau

    fold_seed = seed + fold
    mode_label = "FAST" if fast else "PRODUCTION"
    logger.info(
        f"  [mRMR fold={fold} {mode_label}] "
        f"B={effective_B} | n_select={effective_n_select} | "
        f"tau={effective_tau} | k_mi={k_mi}"
    )

    all_feature_names = list(X_train.columns)
    X_full = X_train.values.astype(float)

    # Step 1 — variance threshold
    variances = X_full.var(axis=0)
    kept_mask = variances > VARIANCE_THRESHOLD
    n_dropped = int((~kept_mask).sum())
    variance_passed: set[str] = {n for n, k in zip(all_feature_names, kept_mask) if k}
    if n_dropped > 0:
        logger.info(f"  [mRMR fold={fold}] variance filter: {n_dropped} removed")

    radiomic_names = [
        f for f in all_feature_names
        if f in variance_passed
        and not f.startswith("delta_")
        and f not in TEMPORAL_COLS
        and f not in NADIR_COLS
        and f not in DELTA_DERIVED_COLS
        and RADIOMIC_PREFIX in f
    ]
    radiomic_indices = [all_feature_names.index(f) for f in radiomic_names]
    X_radiomic = X_full[:, radiomic_indices]

    # Rank consistency probe
    rank_consistency_rho: float | None = None
    if check_consistency and not fast:
        logger.info(f"  [mRMR fold={fold}] running rank consistency probe ({RHO_PROBE_REPLICATES} replicates)...")
        rank_consistency_rho = _probe_rank_consistency(
            X_radiomic, y_train, radiomic_names,
            n_select=min(effective_n_select, len(radiomic_names)),
            k_mi=k_mi, seed=fold_seed,
        )
        logger.info(f"  [mRMR fold={fold}] Spearman ρ = {rank_consistency_rho:.3f}")
        if rank_consistency_rho < RHO_CONSISTENCY_THRESHOLD:
            warnings.warn(
                f"[mRMR fold={fold}] Low rank consistency (ρ={rank_consistency_rho:.3f} < "
                f"{RHO_CONSISTENCY_THRESHOLD}). mRMR stability scores may reflect noise. "
                f"Consider switching to mi_univariate (feature_selector_mi.py). "
                f"Continuing with mRMR as requested.",
                UserWarning,
                stacklevel=2,
            )

    # Step 2 — Stability Selection
    stable_radiomic, bootstrap_stability = run_stability_selection(
        X=X_radiomic,
        y=y_train,
        feature_names=radiomic_names,
        n_select=min(effective_n_select, len(radiomic_names)),
        B=effective_B,
        tau=effective_tau,
        k_mi=k_mi,
        seed=fold_seed,
        n_jobs=n_jobs,
    )

    if verbose:
        top_n = 50
        sorted_stab = sorted(bootstrap_stability.items(), key=lambda x: x[1], reverse=True)[:top_n]
        print(f"\n  [mRMR fold={fold}] Top-{top_n} radiomic by bootstrap stability:")
        for rank, (fname, score) in enumerate(sorted_stab, 1):
            marker = "✅" if score >= effective_tau else "  "
            print(f"    {marker} {rank:3d}. {fname:65s} : {score:.2f}")

    logger.info(
        f"  [mRMR fold={fold}] radiomic: {len(radiomic_names)} → {len(stable_radiomic)} selected"
    )

    # Step 3 — anchor delta features
    anchored_delta = [
        f"delta_{f}" for f in stable_radiomic
        if f"delta_{f}" in variance_passed
    ]

    # Step 4 — full feature set
    temporal = [c for c in all_feature_names if c in TEMPORAL_COLS]
    nadir = [c for c in all_feature_names if c in NADIR_COLS and c in variance_passed]
    delta_derived = [c for c in all_feature_names if c in DELTA_DERIVED_COLS and c in variance_passed]
    full_feature_set = (
        stable_radiomic
        + sorted(temporal)
        + sorted(nadir)
        + sorted(anchored_delta)
        + sorted(delta_derived)
    )

    return AnchoredFoldSelectionResult(
        fold=fold,
        selected_radiomic=stable_radiomic,
        anchored_delta=anchored_delta,
        temporal_cols=temporal,
        full_feature_set=full_feature_set,
        bootstrap_stability=bootstrap_stability,
        n_radiomic_candidates=len(radiomic_names),
        n_radiomic_selected=len(stable_radiomic),
        n_delta_anchored=len(anchored_delta),
        fast_mode=fast,
        rank_consistency_rho=rank_consistency_rho,
    )


# ---------------------------------------------------------------------------
# Cross-fold aggregation (pure) — unchanged from original
# ---------------------------------------------------------------------------

def aggregate_fold_selections(
    fold_results: list[FoldSelectionResult],
    fold_majority_threshold: int = FOLD_MAJORITY_THRESHOLD,
) -> AggregatedSelection:
    """Majority vote aggregation across CV folds."""
    if not fold_results:
        raise ValueError("fold_results is empty.")

    fast_folds = [fr.fold for fr in fold_results if fr.fast_mode]
    if fast_folds:
        raise ValueError(
            f"Folds {fast_folds} in fast mode — selected_features.yaml would be invalid."
        )

    all_features: set[str] = set()
    for fr in fold_results:
        all_features.update(fr.selected_features)

    fold_counts = {name: 0 for name in all_features}
    for fr in fold_results:
        for name in fr.selected_features:
            fold_counts[name] += 1

    n_folds = len(fold_results)
    fold_stability = {name: count / n_folds for name, count in fold_counts.items()}
    selected_features = sorted([
        name for name, count in fold_counts.items()
        if count >= fold_majority_threshold
    ])

    return AggregatedSelection(
        selected_features=selected_features,
        fold_stability=fold_stability,
        fold_results=fold_results,
    )
