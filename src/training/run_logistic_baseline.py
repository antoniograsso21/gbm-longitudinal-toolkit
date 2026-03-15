"""
src/training/run_logistic_baseline.py
=======================================
Entry point for the Logistic Regression baseline (Step 3 — T3.2).

Reads dataset_engineered.parquet, runs StratifiedGroupKFold CV with
feature selection (mRMR + Stability Selection) inside each fold,
logs results to MLflow.

Feature selection runs fold-by-fold on the training split only —
never on the full dataset. This is the architecturally correct pattern:
the scaler is fit on train, then the normalised train fold is passed
to select_features_fold(). The test fold uses only the features selected
on the corresponding train fold.

LR is cross-sectional: it receives the radiomic subset of selected
features (no delta_*, no temporal columns). This makes it the correct
lower bound for Assumption A3.

selected_features.yaml is NOT read here — it is produced by
run_lgbm_baseline.py (T3.3) after LightGBM D CV completes.

Side effects are confined to main(). All computation lives in
logistic_baseline.py, feature_selector.py, cross_validation.py,
and metrics.py (pure functions).

Output artifacts
----------------
data/processed/baselines/lr_results.json
    Per-fold and aggregated metrics.

MLflow experiment: baselines/logistic_regression

Usage
-----
    uv run python src/training/run_logistic_baseline.py
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from src.models.logistic_baseline import (
    select_radiomic_features,
    train_lr_fold,
)
from src.training.cross_validation import build_cv_splits
from src.training.feature_selector import select_features_fold
from src.training.metrics import AggregatedMetrics, FoldMetrics, aggregate_cv_results
from src.utils.lumiere_io import build_full_feature_set, fit_transform_fold, load_random_config, print_section

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PARQUET_PATH: Path = Path("data/processed/preprocessing/dataset_engineered.parquet")
OUTPUT_DIR: Path = Path("data/processed/baselines")
RANDOM_STATE_PATH: str = "configs/random_state.yaml"
LR_CONFIG_PATH: str = "configs/logistic_baseline.yaml"
MLFLOW_EXPERIMENT: str = "baselines/logistic_regression"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_lr_config(config_path: str) -> tuple[list[float], int]:
    """
    Load LR hyperparameter grid from YAML.

    Returns:
        Tuple (c_grid, inner_cv_splits).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["C"], int(cfg.get("inner_cv_splits", 3))


# ---------------------------------------------------------------------------
# MLflow logging helpers
# ---------------------------------------------------------------------------

def _log_fold_metrics(fold_metrics: FoldMetrics) -> None:
    """Log all metrics for a single fold to the active MLflow run."""
    k = fold_metrics.fold
    mlflow.log_metric(f"fold_{k}_macro_f1",          fold_metrics.macro_f1)
    mlflow.log_metric(f"fold_{k}_mcc",               fold_metrics.mcc)
    mlflow.log_metric(f"fold_{k}_auroc_progressive", fold_metrics.auroc_progressive)
    mlflow.log_metric(f"fold_{k}_auroc_stable",      fold_metrics.auroc_stable)
    mlflow.log_metric(f"fold_{k}_auroc_response",    fold_metrics.auroc_response)
    mlflow.log_metric(f"fold_{k}_prauc_progressive", fold_metrics.prauc_progressive)
    mlflow.log_metric(f"fold_{k}_prauc_stable",      fold_metrics.prauc_stable)
    mlflow.log_metric(f"fold_{k}_prauc_response",    fold_metrics.prauc_response)


def _log_aggregated_metrics(agg: AggregatedMetrics) -> None:
    """Log mean and std metrics for the full CV run to the active MLflow run."""
    mlflow.log_metric("macro_f1_mean",          agg.macro_f1_mean)
    mlflow.log_metric("macro_f1_std",           agg.macro_f1_std)
    mlflow.log_metric("mcc_mean",               agg.mcc_mean)
    mlflow.log_metric("mcc_std",                agg.mcc_std)
    mlflow.log_metric("auroc_progressive_mean", agg.auroc_progressive_mean)
    mlflow.log_metric("auroc_stable_mean",      agg.auroc_stable_mean)
    mlflow.log_metric("auroc_response_mean",    agg.auroc_response_mean)
    mlflow.log_metric("prauc_progressive_mean", agg.prauc_progressive_mean)
    mlflow.log_metric("prauc_stable_mean",      agg.prauc_stable_mean)
    mlflow.log_metric("prauc_response_mean",    agg.prauc_response_mean)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False) -> None:
    print_section("T3.2 — Logistic Regression Baseline")
    if fast:
        print("  ⚠️  FAST MODE — B=10, n_select=20. Smoke test only, not production.")

    seed, n_jobs = load_random_config(RANDOM_STATE_PATH)
    c_grid, inner_cv_splits = _load_lr_config(LR_CONFIG_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Input parquet not found: {PARQUET_PATH}. "
            "Run Step 2 (features_builder.py) first."
        )
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Loaded {PARQUET_PATH}: {df.shape[0]} rows, {df.shape[1]} columns")

    all_feature_cols = build_full_feature_set(df)
    y = df["target_encoded"].values
    groups = df["Patient"]

    print(f"  Full feature set: {len(all_feature_cols)} columns")
    print(f"  n_effective: {len(df)} | n_patients: {groups.nunique()}")

    # --- CV splits ---
    cv_splits = build_cv_splits(
        X=df[all_feature_cols],
        y=pd.Series(y),
        groups=groups,
        n_splits=5,
        seed=seed,
    )

    # --- MLflow ---
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    fold_metrics_list: list[FoldMetrics] = []
    fold_results_raw: list[dict] = []

    with mlflow.start_run(run_name="lr_cv"):
        mlflow.log_param("seed", seed)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_param("n_splits", cv_splits.n_splits)
        mlflow.log_param("C_grid", str(c_grid))
        mlflow.log_param("inner_cv_splits", inner_cv_splits)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("feature_set", "radiomic_only_selected")

        for fold_split in cv_splits.folds:
            print_section(f"  Fold {fold_split.fold}")

            X_train_df = df.iloc[fold_split.train_idx]
            X_test_df  = df.iloc[fold_split.test_idx]
            y_train    = y[fold_split.train_idx]
            y_test     = y[fold_split.test_idx]

            # 1 — normalise: scaler fit on train only, never on full dataset
            X_train_scaled, X_test_scaled = fit_transform_fold(
                X_train_df, X_test_df, all_feature_cols
            )

            # 2 — feature selection on normalised train fold (Full set D)
            X_train_scaled_df = pd.DataFrame(
                X_train_scaled, columns=all_feature_cols
            )
            selection = select_features_fold(
                X_train=X_train_scaled_df,
                y_train=y_train,
                fold=fold_split.fold,
                seed=seed,
                fast=fast,
                n_jobs=n_jobs,
            )

            # 3 — restrict to radiomic-only for LR
            # LR is cross-sectional: delta_* and temporal excluded by design
            radiomic_cols = select_radiomic_features(
                df=X_train_scaled_df,
                selected_features=selection.selected_features,
            )

            if not radiomic_cols:
                raise RuntimeError(
                    f"Fold {fold_split.fold}: no radiomic features remain after "
                    "mRMR + Stability Selection on Full set D. "
                    "The selection is too aggressive for this fold — "
                    "consider lowering tau or increasing n_select."
                )

            print(
                f"  {selection.n_selected} features selected | "
                f"{len(radiomic_cols)} radiomic for LR"
            )
            mlflow.log_metric(f"fold_{fold_split.fold}_n_selected",  selection.n_selected)
            mlflow.log_metric(f"fold_{fold_split.fold}_n_radiomic",  len(radiomic_cols))

            # 4 — select radiomic columns by name, not by position index
            # DataFrame-based indexing is robust to column order changes
            X_train_radiomic = X_train_scaled_df[radiomic_cols].values
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=all_feature_cols)
            X_test_radiomic  = X_test_scaled_df[radiomic_cols].values

            result = train_lr_fold(
                X_train=X_train_radiomic,
                y_train=y_train,
                X_test=X_test_radiomic,
                y_test=y_test,
                fold=fold_split.fold,
                seed=seed,
                c_grid=c_grid,
                inner_cv_splits=inner_cv_splits,
            )

            print(
                f"  best_C={result.best_C} | "
                f"macro_f1={result.metrics.macro_f1:.4f} | "
                f"mcc={result.metrics.mcc:.4f}"
            )

            _log_fold_metrics(result.metrics)
            mlflow.log_metric(f"fold_{fold_split.fold}_best_C", result.best_C)

            fold_metrics_list.append(result.metrics)
            fold_results_raw.append(asdict(result))

        # --- Aggregate ---
        aggregated = aggregate_cv_results(fold_metrics_list)
        _log_aggregated_metrics(aggregated)

        print_section("Aggregated Results")
        print(f"  macro_f1        : {aggregated.macro_f1_mean:.4f} ± {aggregated.macro_f1_std:.4f}")
        print(f"  mcc             : {aggregated.mcc_mean:.4f} ± {aggregated.mcc_std:.4f}")
        print(f"  PR-AUC Response : {aggregated.prauc_response_mean:.4f} ± {aggregated.prauc_response_std:.4f}")
        print(f"  PR-AUC Stable   : {aggregated.prauc_stable_mean:.4f} ± {aggregated.prauc_stable_std:.4f}")

        # --- Save JSON report ---
        output = {
            "model": "logistic_regression",
            "feature_set": "radiomic_only_selected",
            "seed": seed,
            "fold_results": fold_results_raw,
            "aggregated": asdict(aggregated),
        }
        report_path = OUTPUT_DIR / "lr_results.json"
        with open(report_path, "w") as f:
            json.dump(output, f, indent=2)

        mlflow.log_artifact(str(report_path))
        print(f"\n  Report → {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression baseline (T3.2)")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smoke test mode: B=10 bootstrap replicates, n_select=20. Never use for production.",
    )
    args = parser.parse_args()
    main(fast=args.fast)