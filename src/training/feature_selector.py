"""
src/training/feature_selector.py
==================================
mRMR + Stability Selection for the LUMIERE baseline pipeline.

Executed fold-by-fold inside the CV loop of each model — never on the
full dataset. There is no standalone entry point for this module:
running feature selection before the models would be data leakage.

Call pattern inside every run_*.py:
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=all_feature_cols)
    selection = select_features_fold(X_train=X_train_scaled_df, y_train=y_train, fold=k)

selected_features.yaml is produced exclusively by run_lgbm_baseline.py
(LightGBM ablation D) via aggregate_fold_selections() — it is not written
by LR or LSTM. Rationale: Full set D, most stable on small n, SHAP provides
independent validation of the mRMR selection.

Algorithm
---------
mRMR (Maximum Relevance Minimum Redundancy):
    score(xi) = I(xi; y) - (1/|S|) * sum_{xj in S} I(xi; xj)

MI estimator: Kraskov k-NN via npeet.
    - Correct for continuous variables with small n.
    - sklearn.mutual_info_classif uses a discretisation-based estimator
      inappropriate for the radiomic feature distribution.

Stability Selection (bootstrap dimension):
    B=100 bootstrap replicates on the training fold (full run).
    B=10  bootstrap replicates in fast mode (smoke test only).
    Feature bootstrap stability = fraction of replicates selected.
    Threshold tau=0.7.

Cross-fold stability:
    Feature fold stability = fraction of CV folds passing bootstrap threshold.
    Features stable in both dimensions are the primary biological interpretation basis.

Aggregation into selected_features.yaml:
    Majority vote: feature included if bootstrap-stable in >= 3/5 folds.
    Majority vote is more robust than strict intersection on n=231.

Fast mode (smoke test):
    Pass fast=True to select_features_fold() to use B=10 and n_select=20.
    Fast mode is for local smoke tests only — never for production runs.

Design:
    All functions are pure — no side effects, no file I/O, no MLflow calls.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from npeet.entropy_estimators import mi as kraskov_mi
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOOTSTRAP_REPLICATES: int = 100       # production
BOOTSTRAP_REPLICATES_FAST: int = 10   # smoke test only
STABILITY_THRESHOLD: float = 0.7
FOLD_MAJORITY_THRESHOLD: int = 3      # out of 5 folds
MRMR_N_SELECT: int = 50              # production
MRMR_N_SELECT_FAST: int = 20         # smoke test only


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------

@dataclass
class FoldSelectionResult:
    """Feature selection result for a single CV fold."""
    fold: int
    selected_features: list[str]
    bootstrap_stability: dict[str, float]   # feature -> fraction selected over B replicates
    n_candidates: int                        # features entering mRMR
    n_selected: int                          # features passing stability threshold
    fast_mode: bool                          # True if run with reduced B/n_select


@dataclass
class AggregatedSelection:
    """
    Aggregated feature selection result across all CV folds.

    selected_features: features passing majority vote (stable in >= FOLD_MAJORITY_THRESHOLD folds).
    fold_stability:    feature -> fraction of folds in which it passed bootstrap threshold.
    fold_results:      per-fold FoldSelectionResult for traceability.
    """
    selected_features: list[str]
    fold_stability: dict[str, float]
    fold_results: list[FoldSelectionResult]


# ---------------------------------------------------------------------------
# MI estimation
# ---------------------------------------------------------------------------

def _compute_mi_feature_target(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 3,
) -> float:
    """
    Estimate I(x; y) using the Kraskov k-NN estimator (npeet).

    x is treated as continuous; y is discrete (class labels).
    npeet handles mixed continuous/discrete via the standard Kraskov approach.

    Args:
        x: 1-D continuous feature array, shape (n,).
        y: 1-D integer label array, shape (n,).
        k: number of nearest neighbours for MI estimation (default 3).

    Returns:
        MI estimate as float. Clipped to 0.0 from below (MI >= 0 by definition;
        negative estimates are numerical artefacts of the k-NN estimator).
    """
    x_col = x.reshape(-1, 1).astype(float)
    y_col = y.reshape(-1, 1).astype(float)
    estimate = kraskov_mi(x_col, y_col, k=k)
    return max(0.0, float(estimate))


def _compute_mi_feature_feature(
    xi: np.ndarray,
    xj: np.ndarray,
    k: int = 3,
) -> float:
    """
    Estimate I(xi; xj) between two continuous features using Kraskov k-NN.

    Args:
        xi: 1-D array, shape (n,).
        xj: 1-D array, shape (n,).
        k:  nearest neighbours (default 3).

    Returns:
        MI estimate clipped to 0.0.
    """
    xi_col = xi.reshape(-1, 1).astype(float)
    xj_col = xj.reshape(-1, 1).astype(float)
    estimate = kraskov_mi(xi_col, xj_col, k=k)
    return max(0.0, float(estimate))


# ---------------------------------------------------------------------------
# mRMR
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

    Greedy forward selection:
        score(xi) = I(xi; y) - (1/|S|) * sum_{xj in S} I(xi; xj)

    At the first step |S|=0 — the redundancy term is 0 and selection is
    purely by relevance I(xi; y).

    Args:
        X:             feature matrix, shape (n_samples, n_features).
        y:             integer label array, shape (n_samples,).
        feature_names: list of feature names aligned with X columns.
        n_select:      number of features to select (default MRMR_N_SELECT).
        k_mi:          k for Kraskov MI estimator (default 3).

    Returns:
        Ordered list of selected feature names (most relevant first).

    Raises:
        ValueError: if n_select > number of features.
        ValueError: if X columns and feature_names lengths differ.
    """
    n_features = X.shape[1]
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but X has "
            f"{n_features} columns."
        )
    if n_select > n_features:
        raise ValueError(
            f"n_select={n_select} exceeds available features ({n_features})."
        )

    relevance: np.ndarray = np.array([
        _compute_mi_feature_target(X[:, i], y, k=k_mi)
        for i in range(n_features)
    ])

    selected_indices: list[int] = []
    remaining: set[int] = set(range(n_features))

    # Cache MI(feature_i, feature_j) within this replicate.
    # The same pair is requested O(n_select) times across greedy steps —
    # caching avoids redundant Kraskov calls within a single bootstrap replicate.
    # Cache is local to run_mrmr (not shared across replicates) because each
    # replicate operates on a different subsample — cross-replicate reuse
    # would give incorrect MI estimates.
    mi_cache: dict[tuple[int, int], float] = {}

    for _ in range(n_select):
        best_idx: int = -1
        best_score: float = -np.inf

        for idx in remaining:
            if not selected_indices:
                score = relevance[idx]
            else:
                redundancy_vals: list[float] = []
                for j in selected_indices:
                    key = (min(idx, j), max(idx, j))
                    if key not in mi_cache:
                        mi_cache[key] = _compute_mi_feature_feature(
                            X[:, idx], X[:, j], k=k_mi
                        )
                    redundancy_vals.append(mi_cache[key])
                redundancy = float(np.mean(redundancy_vals))
                score = relevance[idx] - redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [feature_names[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Stability Selection (bootstrap dimension)
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
    Run Stability Selection over B bootstrap replicates of the training data.

    Each replicate draws a 50% subsample without replacement (standard
    Stability Selection protocol — Meinshausen & Bühlmann, 2010).
    mRMR is run on each subsample. Feature bootstrap stability is the
    fraction of replicates in which it is selected.

    Args:
        X:             feature matrix, shape (n_samples, n_features).
        y:             integer label array, shape (n_samples,).
        feature_names: list of feature names aligned with X columns.
        n_select:      mRMR candidates per replicate.
        B:             number of bootstrap replicates.
                       Use BOOTSTRAP_REPLICATES (100) for production.
                       Use BOOTSTRAP_REPLICATES_FAST (10) for smoke tests.
        tau:           stability threshold (default 0.7).
        k_mi:          k for Kraskov MI estimator (default 3).
        seed:          random seed for bootstrap sampling.
        n_jobs:        number of parallel jobs for bootstrap replicates.
                       -1 uses all available cores. Set to 6 on laptops to
                       reduce thermal load without significant runtime penalty.

    Returns:
        Tuple of:
            - stable_features: list of feature names with stability > tau.
            - bootstrap_stability: dict feature_name -> stability score (0.0–1.0).
    """
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    subsample_size = max(1, n_samples // 2)

    selection_counts: dict[str, int] = {name: 0 for name in feature_names}

    # Pre-generate all bootstrap indices deterministically before parallelisation.
    # Each replicate gets its own seed derived from the global seed to ensure
    # reproducibility regardless of execution order across parallel workers.
    replicate_indices = [
        rng.choice(n_samples, size=subsample_size, replace=False)
        for _ in range(B)
    ]

    def _run_replicate(idx: np.ndarray) -> list[str]:
        return run_mrmr(
            X[idx], y[idx], feature_names,
            n_select=n_select, k_mi=k_mi,
        )

    results: list[list[str]] = Parallel(n_jobs=n_jobs)(
        delayed(_run_replicate)(idx)
        for idx in tqdm(replicate_indices, desc=f"  Bootstrap replicates (B={B})", leave=False)
    )

    for selected in results:
        for name in selected:
            selection_counts[name] += 1

    bootstrap_stability: dict[str, float] = {
        name: count / B
        for name, count in selection_counts.items()
    }

    stable_features = [
        name for name, score in bootstrap_stability.items()
        if score > tau
    ]

    return stable_features, bootstrap_stability


# ---------------------------------------------------------------------------
# Per-fold entry point
# ---------------------------------------------------------------------------

def select_features_fold(
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
) -> FoldSelectionResult:
    """
    Run mRMR + Stability Selection on a single training fold.

    X_train must already be normalised (StandardScaler applied upstream,
    inside the CV loop — never fit on the full dataset).

    Args:
        X_train:  normalised training features, shape (n_train, n_features).
        y_train:  integer labels for the training fold, shape (n_train,).
        fold:     fold index for traceability.
        n_select: mRMR candidates. None → MRMR_N_SELECT (production) or
                  MRMR_N_SELECT_FAST (fast mode).
        B:        bootstrap replicates. None → BOOTSTRAP_REPLICATES (production)
                  or BOOTSTRAP_REPLICATES_FAST (fast mode).
        tau:      stability threshold (default 0.7).
        k_mi:     k for Kraskov MI estimator (default 3).
        seed:     random seed.
        fast:     if True, use reduced B=10 and n_select=20 for smoke tests.
                  Never use fast=True for production runs — results are not
                  scientifically valid.

    Returns:
        FoldSelectionResult with selected features, bootstrap stability scores,
        and fast_mode flag for downstream validation.
    """
    if fast:
        effective_B = B if B is not None else BOOTSTRAP_REPLICATES_FAST
        effective_n_select = n_select if n_select is not None else MRMR_N_SELECT_FAST
    else:
        effective_B = B if B is not None else BOOTSTRAP_REPLICATES
        effective_n_select = n_select if n_select is not None else MRMR_N_SELECT

    feature_names = list(X_train.columns)
    X_arr = X_train.values.astype(float)

    stable_features, bootstrap_stability = run_stability_selection(
        X=X_arr,
        y=y_train,
        feature_names=feature_names,
        n_select=effective_n_select,
        B=effective_B,
        tau=tau,
        k_mi=k_mi,
        seed=seed,
        n_jobs=n_jobs,
    )

    return FoldSelectionResult(
        fold=fold,
        selected_features=stable_features,
        bootstrap_stability=bootstrap_stability,
        n_candidates=len(feature_names),
        n_selected=len(stable_features),
        fast_mode=fast,
    )


# ---------------------------------------------------------------------------
# Cross-fold aggregation
# ---------------------------------------------------------------------------

def aggregate_fold_selections(
    fold_results: list[FoldSelectionResult],
    fold_majority_threshold: int = FOLD_MAJORITY_THRESHOLD,
) -> AggregatedSelection:
    """
    Aggregate per-fold selections into a final feature set via majority vote.

    A feature is included in the final set if it was selected (bootstrap-stable)
    in >= fold_majority_threshold folds out of len(fold_results).

    Called exclusively by run_lgbm_baseline.py after LightGBM D CV completes.
    Not called by run_logistic_baseline.py or run_lstm_baseline.py.

    Raises ValueError if any fold_result has fast_mode=True — aggregation
    on fast-mode results is not scientifically valid and must be blocked.

    Args:
        fold_results:            list of FoldSelectionResult, one per fold.
        fold_majority_threshold: minimum folds required for inclusion (default 3).

    Returns:
        AggregatedSelection with final selected_features and fold_stability scores.

    Raises:
        ValueError: if fold_results is empty.
        ValueError: if any fold_result was produced in fast mode.
    """
    if not fold_results:
        raise ValueError("fold_results is empty — nothing to aggregate.")

    fast_folds = [fr.fold for fr in fold_results if fr.fast_mode]
    if fast_folds:
        raise ValueError(
            f"Folds {fast_folds} were run in fast mode. "
            "aggregate_fold_selections() must not be called on fast-mode results — "
            "selected_features.yaml would not be scientifically valid."
        )

    all_features: set[str] = set()
    for fr in fold_results:
        all_features.update(fr.selected_features)

    fold_counts: dict[str, int] = {name: 0 for name in all_features}
    for fr in fold_results:
        for name in fr.selected_features:
            fold_counts[name] += 1

    n_folds = len(fold_results)
    fold_stability: dict[str, float] = {
        name: count / n_folds
        for name, count in fold_counts.items()
    }

    selected_features = sorted([
        name for name, count in fold_counts.items()
        if count >= fold_majority_threshold
    ])

    return AggregatedSelection(
        selected_features=selected_features,
        fold_stability=fold_stability,
        fold_results=fold_results,
    )