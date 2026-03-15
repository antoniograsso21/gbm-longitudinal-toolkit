"""
src/models/gbm_baseline.py
===========================
LightGBM baseline for the LUMIERE pipeline (Step 3 — T3.3).

Four temporal ablations quantify the contribution of each feature group:
    A: Radiomic only      — static snapshot, no temporal context
    B: Temporal only      — 3 columns: interval_weeks, scan_index,
                            time_from_diagnosis_weeks
    C: Radiomic + Temporal
    D: Full set           — Radiomic + Temporal + Delta (primary model)

Decision rules (evaluated in run_lgbm_baseline.py, declared in paper):
    If macro_F1(B) ≈ macro_F1(C): weak radiomic signal
    If macro_F1(B) > 0.33:        temporal leakage confirmed

selected_features.yaml is written by run_lgbm_baseline.py after ablation D
CV completes — this is the only model that produces this artefact.

SHAP is run on the best-fold model of ablation D only.

Design:
    All functions are pure — no MLflow, no file I/O, no printing.
    Side effects live exclusively in run_lgbm_baseline.py.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import shap
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.training.metrics import FoldMetrics, compute_metrics


# ---------------------------------------------------------------------------
# Ablation type
# ---------------------------------------------------------------------------

AblationType = Literal["A", "B", "C", "D"]

TEMPORAL_COLS: list[str] = [
    "interval_weeks",
    "scan_index",
    "time_from_diagnosis_weeks",
]


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------

@dataclass
class LGBMFoldResult:
    """Full result for a single CV fold of one LightGBM ablation."""
    fold: int
    ablation: AblationType
    best_params: dict
    metrics: FoldMetrics
    n_train: int
    n_test: int
    n_features: int
    model: object   # fitted LGBMClassifier — used for SHAP in ablation D


@dataclass
class SHAPResult:
    """SHAP output for the best fold of ablation D."""
    fold: int
    feature_names: list[str]
    mean_abs_shap: list[float]          # mean |SHAP| per feature, aligned with feature_names
    interval_weeks_rank: int | None     # rank of interval_weeks by mean |SHAP| (1-based, None if absent)


# ---------------------------------------------------------------------------
# Feature set builders
# ---------------------------------------------------------------------------

def build_ablation_feature_set(
    selected_features: list[str],
    ablation: AblationType,
) -> list[str]:
    """
    Return the feature list for a given ablation from the fold's selected features.

    Ablation B (temporal only) uses TEMPORAL_COLS directly — mRMR is not
    run on 3 features, so selected_features is ignored for ablation B.

    Args:
        selected_features: features selected by mRMR + Stability Selection
                           on the Full set D for this fold.
        ablation:          one of A, B, C, D.

    Returns:
        List of feature column names for this ablation.

    Raises:
        ValueError: if ablation is not one of A/B/C/D.
        ValueError: if the resulting feature list is empty (ablation A/C/D).
    """
    if ablation == "B":
        return TEMPORAL_COLS.copy()

    # For A, C, D: start from selected features
    radiomic = [
        f for f in selected_features
        if not f.startswith("delta_")
        and f not in TEMPORAL_COLS
    ]
    delta = [
        f for f in selected_features
        if f.startswith("delta_")
    ]

    if ablation == "A":
        result = radiomic
    elif ablation == "C":
        result = radiomic + TEMPORAL_COLS
    elif ablation == "D":
        result = radiomic + TEMPORAL_COLS + delta
    else:
        raise ValueError(f"Unknown ablation: '{ablation}'. Must be one of A/B/C/D.")

    if not result:
        raise ValueError(
            f"Ablation {ablation}: empty feature list after filtering. "
            "Check that mRMR selected at least some radiomic features."
        )
    return result


# ---------------------------------------------------------------------------
# Core training function (pure)
# ---------------------------------------------------------------------------

def train_lgbm_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold: int,
    ablation: AblationType,
    param_grid: dict,
    n_iter: int = 30,
    seed: int = 42,
) -> LGBMFoldResult:
    """
    Train LightGBM with RandomizedSearchCV + early stopping on a single fold.

    Three splits are used:
        train: hyperparameter search + final training
        val:   early stopping only (10% of train fold, never used for metrics)
        test:  evaluation only

    RandomizedSearchCV runs on (train+val) with inner 3-fold CV for param
    search. The best estimator is then retrained on train only with early
    stopping monitored on val.

    Args:
        X_train:    normalised training features, shape (n_train, n_features).
        y_train:    integer labels for training.
        X_val:      normalised early-stopping validation set.
        y_val:      integer labels for validation.
        X_test:     normalised test features.
        y_test:     integer labels for test.
        fold:       fold index for traceability.
        ablation:   one of A/B/C/D.
        param_grid: hyperparameter search space dict.
        n_iter:     RandomizedSearchCV iterations (default 30).
        seed:       random seed.

    Returns:
        LGBMFoldResult with best params and all evaluation metrics.
    """
    # Ablation B has only 3 features — RandomizedSearch is not meaningful
    use_search = ablation != "B"

    base_model = LGBMClassifier(
        class_weight="balanced",
        random_state=seed,
        verbose=-1,
    )

    if use_search:
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=3,
            scoring="f1_macro",
            refit=False,           # we refit manually with early stopping
            random_state=seed,
            n_jobs=-1,
        )
        # Search on train only (not val — val is for early stopping only)
        search.fit(X_train, y_train)
        best_params = search.best_params_
    else:
        best_params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
        }

    # Refit with early stopping on val set
    final_model = LGBMClassifier(
        **best_params,
        class_weight="balanced",
        random_state=seed,
        verbose=-1,
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(20, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    y_pred  = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)

    fold_metrics = compute_metrics(
        fold=fold,
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    return LGBMFoldResult(
        fold=fold,
        ablation=ablation,
        best_params=best_params,
        metrics=fold_metrics,
        n_train=len(y_train),
        n_test=len(y_test),
        n_features=X_train.shape[1],
        model=final_model,
    )


# ---------------------------------------------------------------------------
# SHAP (ablation D best fold only)
# ---------------------------------------------------------------------------

def compute_shap(
    model: "LGBMClassifier",
    X_test: np.ndarray,
    feature_names: list[str],
    fold: int,
) -> SHAPResult:
    """
    Compute SHAP values for the best-fold LightGBM D model.

    Uses TreeExplainer (exact, fast for tree models).
    mean |SHAP| is computed across all samples and all classes,
    then averaged — consistent with the beeswarm plot convention.

    Args:
        model:         fitted LGBMClassifier.
        X_test:        normalised test features, shape (n_test, n_features).
        feature_names: feature names aligned with X_test columns.
        fold:          fold index for traceability.

    Returns:
        SHAPResult with mean |SHAP| per feature and interval_weeks rank.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values is list of arrays (one per class) for multiclass
    # shape per class: (n_samples, n_features)
    # mean |SHAP| per feature: average over classes and samples
    mean_abs = np.mean(
        [np.abs(sv).mean(axis=0) for sv in shap_values],
        axis=0,
    )

    # Rank by mean |SHAP| descending (1-based)
    ranked_indices = np.argsort(mean_abs)[::-1]
    rank_map = {feature_names[i]: rank + 1 for rank, i in enumerate(ranked_indices)}

    interval_weeks_rank = rank_map.get("interval_weeks", None)

    return SHAPResult(
        fold=fold,
        feature_names=feature_names,
        mean_abs_shap=mean_abs.tolist(),
        interval_weeks_rank=interval_weeks_rank,
    )
