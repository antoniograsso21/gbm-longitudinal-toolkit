"""
src/models/logistic_baseline.py
================================
Logistic Regression baseline for the LUMIERE pipeline (Step 3 — T3.2).

Cross-sectional model: uses the radiomic feature vector at a single
timepoint T (no delta, no temporal features). This is intentional —
LR is the lower bound and the direct test of Assumption A3:
"does temporal dynamics add signal beyond a static snapshot?"

Feature set used: radiomic set only (selected features from T3.1).
Delta features and temporal features are excluded by design.

Hyperparameter search: GridSearchCV over C inside each CV fold.
class_weight='balanced' is mandatory — with 76% Progressive, an
unweighted classifier collapses to the majority class.

Design:
    All functions are pure — no MLflow, no file I/O, no printing.
    Side effects live exclusively in run_logistic_baseline.py.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.training.metrics import FoldMetrics, compute_metrics


# ---------------------------------------------------------------------------
# Constants — defaults used if YAML not provided (should not happen in practice)
# ---------------------------------------------------------------------------

C_GRID_DEFAULT: list[float] = [0.01, 0.1, 1.0, 10.0]
INNER_CV_SPLITS_DEFAULT: int = 3


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------

@dataclass
class LRFoldResult:
    """Full result for a single CV fold of the LR baseline."""
    fold: int
    best_C: float
    metrics: FoldMetrics
    n_train: int
    n_test: int
    n_features: int


# ---------------------------------------------------------------------------
# Feature set selector
# ---------------------------------------------------------------------------

def select_radiomic_features(
    df: pd.DataFrame,
    selected_features: list[str],
) -> list[str]:
    """
    Return the intersection of selected_features with radiomic columns only.

    LR is cross-sectional: delta_* and temporal columns are excluded.
    This makes LR the correct lower bound for Assumption A3.

    Args:
        df:                full feature DataFrame.
        selected_features: feature names from configs/selected_features.yaml.

    Returns:
        List of column names: selected features that are radiomic (no delta_,
        no interval_weeks, no scan_index, no time_from_diagnosis_weeks).

    Raises:
        ValueError: if the resulting list is empty.
    """
    temporal = {
        "interval_weeks", "scan_index", "time_from_diagnosis_weeks",
        "CE_vs_nadir", "weeks_since_nadir",  # nadir features require patient history
        # — not cross-sectional, excluded from LR static baseline
    }
    radiomic_selected = [
        f for f in selected_features
        if f in df.columns
        and not f.startswith("delta_")
        and f not in temporal
    ]
    if not radiomic_selected:
        raise ValueError(
            "No radiomic features remain after filtering selected_features. "
            "Check that selected_features.yaml contains radiomic columns."
        )
    return radiomic_selected


# ---------------------------------------------------------------------------
# Core training function (pure)
# ---------------------------------------------------------------------------

def train_lr_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold: int,
    seed: int = 42,
    c_grid: list[float] | None = None,
    inner_cv_splits: int = INNER_CV_SPLITS_DEFAULT,
) -> LRFoldResult:
    """
    Train Logistic Regression with GridSearchCV on a single CV fold.

    Normalization must be applied by the caller before passing arrays here.
    This function is pure: it receives arrays and returns a typed result.

    GridSearchCV uses inner 3-fold CV on the training set only.
    Scoring metric: macro F1 (consistent with the paper's primary metric).

    Args:
        X_train: normalised training features, shape (n_train, n_features).
        y_train: integer labels for training, shape (n_train,).
        X_test:  normalised test features, shape (n_test, n_features).
        y_test:  integer labels for test, shape (n_test,).
        fold:           fold index for traceability.
        seed:           random seed for LogisticRegression solver.
        c_grid:         list of C values for GridSearchCV (default C_GRID_DEFAULT).
        inner_cv_splits: inner CV folds for GridSearchCV (default 3).

    Returns:
        LRFoldResult with best hyperparameters and all evaluation metrics.

    Raises:
        ValueError: if X_train and X_test have different numbers of columns.
    """
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature count mismatch: X_train has {X_train.shape[1]} columns, "
            f"X_test has {X_test.shape[1]}."
        )

    base_model = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    )

    effective_c_grid = c_grid if c_grid is not None else C_GRID_DEFAULT
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid={"C": effective_c_grid},
        cv=inner_cv_splits,
        scoring="f1_macro",
        refit=True,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model: LogisticRegression = grid_search.best_estimator_
    best_C: float = float(grid_search.best_params_["C"])

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    fold_metrics = compute_metrics(
        fold=fold,
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    return LRFoldResult(
        fold=fold,
        best_C=best_C,
        metrics=fold_metrics,
        n_train=len(y_train),
        n_test=len(y_test),
        n_features=X_train.shape[1],
    )