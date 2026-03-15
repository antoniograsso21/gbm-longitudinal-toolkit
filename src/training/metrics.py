"""
src/training/metrics.py
========================
Metric computation for the LUMIERE baseline pipeline.

Metrics reported (accuracy is never computed):
    macro_f1      — primary metric, equal weight across all 3 classes
    mcc           — Matthews Correlation Coefficient
    auroc_{class} — one-vs-rest AUROC per class
    prauc_{class} — one-vs-rest PR-AUC per class (primary for minority classes)

Class names follow LABEL_ENCODING from lumiere_io:
    0 = Progressive  (76% of examples)
    1 = Stable       (11%)
    2 = Response     (13%)

PR-AUC is the primary lens for Stable and Response given heavy class imbalance.
A trivial classifier achieves macro F1 ≈ 0.29 and MCC ≈ 0.0 on this distribution.

Design:
    compute_metrics()       — pure function, single fold
    aggregate_cv_results()  — pure function, list of fold dicts → mean ± std
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
)

# Class index → human-readable label (mirrors LABEL_ENCODING in lumiere_io)
CLASS_NAMES: dict[int, str] = {
    0: "Progressive",
    1: "Stable",
    2: "Response",
}

N_CLASSES: int = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Typed result
# ---------------------------------------------------------------------------

@dataclass
class FoldMetrics:
    """All metrics for a single CV fold."""
    fold: int
    macro_f1: float
    mcc: float
    auroc_progressive: float
    auroc_stable: float
    auroc_response: float
    prauc_progressive: float
    prauc_stable: float
    prauc_response: float


@dataclass
class AggregatedMetrics:
    """Mean ± std across all CV folds."""
    macro_f1_mean: float
    macro_f1_std: float
    mcc_mean: float
    mcc_std: float
    auroc_progressive_mean: float
    auroc_progressive_std: float
    auroc_stable_mean: float
    auroc_stable_std: float
    auroc_response_mean: float
    auroc_response_std: float
    prauc_progressive_mean: float
    prauc_progressive_std: float
    prauc_stable_mean: float
    prauc_stable_std: float
    prauc_response_mean: float
    prauc_response_std: float


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_metrics(
    fold: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> FoldMetrics:
    """
    Compute all evaluation metrics for a single CV fold.

    Accuracy is intentionally not computed. With 76% Progressive,
    a trivial classifier achieves 76% accuracy — the metric is meaningless.

    Args:
        fold:    fold index (0-based), stored for traceability.
        y_true:  ground-truth integer labels, shape (n,).
        y_pred:  predicted integer labels, shape (n,).
        y_proba: predicted class probabilities, shape (n, 3).
                 Column order must match CLASS_NAMES: [Progressive, Stable, Response].

    Returns:
        FoldMetrics dataclass with all computed metrics.

    Raises:
        ValueError: if y_proba does not have exactly N_CLASSES columns.
        ValueError: if y_true, y_pred, y_proba have inconsistent lengths.
    """
    n = len(y_true)
    if not (n == len(y_pred) == y_proba.shape[0]):
        raise ValueError(
            f"Length mismatch: y_true={n}, y_pred={len(y_pred)}, "
            f"y_proba rows={y_proba.shape[0]}."
        )
    if y_proba.shape[1] != N_CLASSES:
        raise ValueError(
            f"y_proba must have {N_CLASSES} columns (one per class). "
            f"Got shape {y_proba.shape}."
        )

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # One-vs-rest per class
    auroc: dict[str, float] = {}
    prauc: dict[str, float] = {}
    for class_idx, class_name in CLASS_NAMES.items():
        binary_true = (y_true == class_idx).astype(int)
        proba_col = y_proba[:, class_idx]

        # If only one class present in this fold's test set, metric is undefined
        if binary_true.sum() == 0 or binary_true.sum() == n:
            auroc[class_name] = float("nan")
            prauc[class_name] = float("nan")
        else:
            auroc[class_name] = roc_auc_score(binary_true, proba_col)
            prauc[class_name] = average_precision_score(binary_true, proba_col)

    return FoldMetrics(
        fold=fold,
        macro_f1=macro_f1,
        mcc=mcc,
        auroc_progressive=auroc["Progressive"],
        auroc_stable=auroc["Stable"],
        auroc_response=auroc["Response"],
        prauc_progressive=prauc["Progressive"],
        prauc_stable=prauc["Stable"],
        prauc_response=prauc["Response"],
    )


def aggregate_cv_results(fold_metrics: list[FoldMetrics]) -> AggregatedMetrics:
    """
    Compute mean and std across CV folds for all metrics.

    NaN values (from folds where a class was absent) are excluded from
    mean/std computation via np.nanmean / np.nanstd.

    Args:
        fold_metrics: list of FoldMetrics, one per fold.

    Returns:
        AggregatedMetrics dataclass with mean and std for each metric.

    Raises:
        ValueError: if fold_metrics is empty.
    """
    if not fold_metrics:
        raise ValueError("fold_metrics is empty — nothing to aggregate.")

    def _mean(vals: list[float]) -> float:
        return float(np.nanmean(vals))

    def _std(vals: list[float]) -> float:
        return float(np.nanstd(vals))

    fields = [
        "macro_f1", "mcc",
        "auroc_progressive", "auroc_stable", "auroc_response",
        "prauc_progressive", "prauc_stable", "prauc_response",
    ]

    kwargs: dict[str, float] = {}
    for field in fields:
        vals = [getattr(fm, field) for fm in fold_metrics]
        kwargs[f"{field}_mean"] = _mean(vals)
        kwargs[f"{field}_std"] = _std(vals)

    return AggregatedMetrics(**kwargs)
