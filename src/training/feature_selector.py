"""
src/training/feature_selector.py
==================================
mRMR + Stability Selection for the LUMIERE baseline pipeline.

Executed fold-by-fold inside the CV loop of each model — never on the
full dataset. There is no standalone entry point for this module:
running feature selection before the models would be data leakage.

Call pattern inside every run_*.py (via training_utils):
    from src.training.training_utils import select_features_fold_anchored_cached
    selection = select_features_fold_anchored_cached(X_train_df, y_train, fold, ...)

selected_features.yaml is produced exclusively by run_lgbm_baseline.py
(LightGBM ablation D) via aggregate_fold_selections().

Algorithm
---------
mRMR (Maximum Relevance Minimum Redundancy):
    score(xi) = I(xi; y) - (1/|S|) * sum_{xj in S} I(xi; xj)

MI estimator: Kraskov k-NN via npeet.
    - Correct for continuous variables with small n.
    - sklearn.mutual_info_classif uses discretisation — inappropriate here.

Stability Selection:
    Bootstrap with StratifiedShuffleSplit (preserves class balance per replicate).
    Feature stability = fraction of replicates in which it is selected.

Delta anchoring:
    Delta features are not passed to mRMR. Instead, delta_f is included in
    full_feature_set if and only if its base radiomic f was selected by mRMR
    AND delta_f passes variance threshold. Label-free, no leakage.

Design:
    All functions in this module are pure — no side effects, no file I/O.
    The cached wrapper lives in training_utils.py.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from npeet.entropy_estimators import mi as kraskov_mi
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


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


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------

@dataclass
class FoldSelectionResult:
    """Radiomic-only selection result — used for YAML aggregation in T3.3."""
    fold: int
    selected_features: list[str]
    bootstrap_stability: dict[str, float]
    n_candidates: int
    n_selected: int
    fast_mode: bool


@dataclass
class AnchoredFoldSelectionResult:
    """
    Full feature selection result with anchored delta features.

    selected_radiomic:   radiomic features passing mRMR + Stability Selection.
    anchored_delta:      delta_f where f in selected_radiomic AND variance OK.
    temporal_cols:       temporal features (always included, never selected).
    full_feature_set:    selected_radiomic + temporal + nadir + anchored_delta
                         + delta_derived. Used by LightGBM/LSTM/GNN.
    bootstrap_stability: stability scores for radiomic features.
    """
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
    """Estimate I(x; y) via Kraskov k-NN. Clipped to 0.0 from below."""
    estimate = kraskov_mi(x.reshape(-1, 1).astype(float), y.reshape(-1, 1).astype(float), k=k)
    return max(0.0, float(estimate))


def _compute_mi_feature_feature(xi: np.ndarray, xj: np.ndarray, k: int = 3) -> float:
    """Estimate I(xi; xj) via Kraskov k-NN. Clipped to 0.0 from below."""
    estimate = kraskov_mi(xi.reshape(-1, 1).astype(float), xj.reshape(-1, 1).astype(float), k=k)
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

    Greedy forward selection:
        score(xi) = I(xi; y) - (1/|S|) * mean_{xj in S} I(xi; xj)

    At the first step |S|=0 — redundancy term is 0, selection is purely
    by relevance I(xi; y).

    MI cache is local to this call — correct because each bootstrap replicate
    operates on a different subsample, so cross-replicate reuse would give
    incorrect MI estimates.

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
    Run Stability Selection over B bootstrap replicates of the training data.

    Uses StratifiedShuffleSplit (not rng.choice) to preserve class distribution
    per replicate — critical on 76/11/13 imbalanced dataset where random
    subsampling could produce replicates with zero minority-class examples.

    Each replicate gets np.random.seed(seed + rep_idx) to fix npeet's internal
    jittering — ensures reproducibility across parallel workers.

    Args:
        X:             feature matrix, shape (n_samples, n_features).
        y:             integer label array, shape (n_samples,).
        feature_names: list of feature names aligned with X columns.
        n_select:      mRMR candidates per replicate (default MRMR_N_SELECT).
        B:             number of bootstrap replicates.
                       BOOTSTRAP_REPLICATES (100) for production.
                       BOOTSTRAP_REPLICATES_FAST (10) for smoke tests.
        tau:           stability threshold — feature must appear in >= tau
                       fraction of replicates to be selected.
        k_mi:          k for Kraskov MI estimator (default 3).
        seed:          random seed for StratifiedShuffleSplit.
        n_jobs:        parallel jobs for bootstrap replicates.
                       -1 uses all cores. Set to 6 on laptops to reduce
                       thermal load without significant runtime penalty.

    Returns:
        Tuple of:
            stable_features:      list of feature names with stability >= tau.
            bootstrap_stability:  dict feature_name -> stability score (0.0–1.0).
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
# Main entry point (pure) — single unified function
# ---------------------------------------------------------------------------

def select_features_fold_anchored(
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
) -> AnchoredFoldSelectionResult:
    """
    Feature selection with anchored delta features. Four-step fold-wise flow:

        1. Variance threshold on full set (label-free)
        2. mRMR + Stability Selection on radiomic-only subset → selected_radiomic
        3. anchored_delta = {delta_f : f in selected_radiomic AND variance OK}
        4. full_feature_set = selected_radiomic + temporal + nadir
                              + anchored_delta + delta_derived

    This is the single unified entry point — select_features_fold is removed.
    The cached wrapper is in training_utils.select_features_fold_anchored_cached().

    LR reads: result.selected_radiomic
    LightGBM/LSTM/GNN read: result.full_feature_set

    Args:
        X_train:  normalised training features (full set D), shape (n_train, n_features).
                  Must be a DataFrame with named columns — column names drive
                  all filtering logic (delta_*, TEMPORAL_COLS, NADIR_COLS).
        y_train:  integer labels for the training fold, shape (n_train,).
        fold:     fold index for traceability and fold_seed derivation.
        n_select: mRMR candidates on radiomic subset.
                  None → MRMR_N_SELECT (production) or MRMR_N_SELECT_FAST (fast).
        B:        bootstrap replicates.
                  None → BOOTSTRAP_REPLICATES (production) or BOOTSTRAP_REPLICATES_FAST.
        tau:      stability threshold (default STABILITY_THRESHOLD).
                  Calibrated empirically: start at 0.6, lower to 0.5 if fold 0
                  gives 0 features. tau=0.7 requires n>>200 per fold.
        k_mi:     k for Kraskov MI estimator (default 3).
        seed:     base random seed. fold_seed = seed + fold is used internally
                  to ensure diversity across folds and reproducibility within.
        fast:     smoke test mode — uses BOOTSTRAP_REPLICATES_FAST and
                  STABILITY_THRESHOLD_FAST. Never use for production runs.
        n_jobs:   parallel jobs for bootstrap. -1 uses all cores.
        verbose:  if True, print top-50 radiomic features by bootstrap stability
                  before applying tau. Useful for tau calibration.

    Returns:
        AnchoredFoldSelectionResult with:
            selected_radiomic:   radiomic features passing mRMR + Stability Selection.
            anchored_delta:      delta features anchored to selected_radiomic.
            temporal_cols:       temporal features (always included).
            full_feature_set:    union of all above + nadir + delta_derived.
            bootstrap_stability: stability scores for all radiomic candidates.
            n_radiomic_candidates, n_radiomic_selected, n_delta_anchored: counts.
            fast_mode:           True if run with reduced B/tau.
    """
    if fast:
        effective_B = B if B is not None else BOOTSTRAP_REPLICATES_FAST
        effective_n_select = n_select if n_select is not None else MRMR_N_SELECT_FAST
        effective_tau = tau if tau != STABILITY_THRESHOLD else STABILITY_THRESHOLD_FAST
    else:
        effective_B = B if B is not None else BOOTSTRAP_REPLICATES
        effective_n_select = n_select if n_select is not None else MRMR_N_SELECT
        effective_tau = tau

    fold_seed = seed + fold  # diversity across folds, reproducibility within

    mode_label = "FAST" if fast else "PRODUCTION"
    print(
        f"  [feature_selector fold={fold} {mode_label}] "
        f"B={effective_B} | n_select={effective_n_select} | "
        f"tau={effective_tau} | variance_threshold={VARIANCE_THRESHOLD}"
    )

    all_feature_names = list(X_train.columns)
    X_full = X_train.values.astype(float)

    # Step 1 — variance threshold (label-free)
    variances = X_full.var(axis=0)
    kept_mask = variances > VARIANCE_THRESHOLD
    n_dropped = int((~kept_mask).sum())
    variance_passed: set[str] = {n for n, k in zip(all_feature_names, kept_mask) if k}
    if n_dropped > 0:
        print(f"  [feature_selector fold={fold}] variance filter: {n_dropped} removed")

    # Step 2 — mRMR + Stability Selection on radiomic-only subset
    radiomic_names = [
        f for f in all_feature_names
        if f in variance_passed
        and not f.startswith("delta_")
        and f not in TEMPORAL_COLS
        and f not in NADIR_COLS
        and f not in DELTA_DERIVED_COLS
    ]
    radiomic_indices = [all_feature_names.index(f) for f in radiomic_names]
    X_radiomic = X_full[:, radiomic_indices]

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
        print(f"\n  [feature_selector fold={fold}] Top-{top_n} radiomic by bootstrap stability:")
        for rank, (fname, score) in enumerate(sorted_stab, 1):
            marker = "✅" if score >= effective_tau else "  "
            print(f"    {marker} {rank:3d}. {fname:65s} : {score:.2f}")
        n_above = sum(1 for _, s in sorted_stab if s >= effective_tau)
        print(f"  [feature_selector fold={fold}] {n_above} above tau={effective_tau:.2f} in top-{top_n}\n")

    print(f"  [feature_selector fold={fold}] radiomic: {len(radiomic_names)} → {len(stable_radiomic)} selected")

    # Step 3 — anchor delta features
    anchored_delta = [
        f"delta_{f}" for f in stable_radiomic
        if f"delta_{f}" in variance_passed
    ]
    print(f"  [feature_selector fold={fold}] anchored delta: {len(anchored_delta)}")

    # Step 4 — build full feature set
    temporal = [c for c in all_feature_names if c in TEMPORAL_COLS]
    nadir = [c for c in all_feature_names if c in NADIR_COLS and c in variance_passed]
    delta_derived = [c for c in all_feature_names if c in DELTA_DERIVED_COLS and c in variance_passed]
    full_feature_set = stable_radiomic + temporal + nadir + anchored_delta + delta_derived

    print(
        f"  [feature_selector fold={fold}] "
        f"nadir: {len(nadir)} | delta_derived: {len(delta_derived)} | "
        f"total full_feature_set: {len(full_feature_set)}"
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
    )


# ---------------------------------------------------------------------------
# Cross-fold aggregation (pure)
# ---------------------------------------------------------------------------

def aggregate_fold_selections(
    fold_results: list[FoldSelectionResult],
    fold_majority_threshold: int = FOLD_MAJORITY_THRESHOLD,
) -> AggregatedSelection:
    """
    Majority vote aggregation across CV folds.
    Called exclusively by run_lgbm_baseline.py after ablation D completes.
    Raises if any fold was run in fast mode.
    """
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