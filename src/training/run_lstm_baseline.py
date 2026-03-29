"""
src/training/run_lstm_baseline.py
===================================
Entry point for the LSTM baseline (Step 3 — T3.4).

Reads dataset_engineered.parquet, runs StratifiedGroupKFold CV with
feature selection (mRMR + Stability Selection) inside each fold,
identical pattern to run_lgbm_baseline.py ablation D.

LSTM operates at patient level: each patient's paired examples become
a chronological sequence. The sequence label is the last timepoint's
target (already label-shifted in the parquet).

Val split: patient-level stratified (StratifiedGroupKFold n=10) — avoids
splitting patient sequences across train/val, preserves class balance.

Grid search: manual grid over hidden_size × num_layers × dropout × lr.
Best config per fold selected by best_val_loss (30-epoch proxy).
Final model trained with full patience on the best config.

Side effects are confined to main(). All computation lives in
lstm_baseline.py, feature_selector.py, cross_validation.py,
and metrics.py (pure functions).

Output artifacts
----------------
data/processed/baselines/lstm_results.json
MLflow experiment: baselines/lstm

Usage
-----
    uv run python src/training/run_lstm_baseline.py
    uv run python src/training/run_lstm_baseline.py --fast
    uv run python src/training/run_lstm_baseline.py --verbose
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from itertools import product
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from src.models.lstm_baseline import (
    LSTMFoldResult,
    build_patient_sequences,
    compute_class_weights,
    train_lstm_fold,
)
from src.training.cross_validation import CVSplits, build_cv_splits
from src.training.feature_selector import (
    AnchoredFoldSelectionResult,
    BOOTSTRAP_REPLICATES_FAST,
    MRMR_N_SELECT_FAST,
    STABILITY_THRESHOLD_FAST,
)
from src.training.metrics import AggregatedMetrics, FoldMetrics, aggregate_cv_results
from src.training.training_utils import (
    fit_transform_fold,
    load_random_config,
    select_features_fold_anchored_cached,
)
from src.utils.lumiere_io import build_full_feature_set, print_section

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PARQUET_PATH: Path = Path("data/processed/preprocessing/dataset_engineered.parquet")
OUTPUT_DIR: Path = Path("data/processed/baselines")
LSTM_CONFIG_PATH: str = "configs/lstm_baseline.yaml"
RANDOM_STATE_PATH: str = "configs/random_state.yaml"
MLFLOW_EXPERIMENT: str = "baselines/lstm"

VAL_FRACTION: float = 0.1


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_lstm_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Patient-level stratified val split (pure)
# ---------------------------------------------------------------------------

def _patient_train_val_split(
    patient_ids: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split scan-level indices into train/val ensuring whole patients go to val.

    Random patient-level split (no stratification). Stratification is not
    feasible here: with ~50 patients in the train fold and 3 imbalanced classes
    (76/11/13), the minority class has at most 3-4 patients — too few for any
    stratified splitter. Declare this limitation in paper Methods.

    Args:
        patient_ids:  patient ID per scan row, shape (n_scans,).
        y:            integer label per scan row, shape (n_scans,) — unused,
                      kept for API consistency with callers.
        val_fraction: approximate fraction of patients for val.
        seed:         random seed.

    Returns:
        train_idx, val_idx — scan-level indices into patient_ids array.
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    n_val = max(1, round(len(unique_patients) * val_fraction))
    val_patients = set(rng.choice(unique_patients, size=n_val, replace=False))
    train_idx = np.where(~np.isin(patient_ids, list(val_patients)))[0]
    val_idx   = np.where( np.isin(patient_ids, list(val_patients)))[0]
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Grid search helper (pure — selects best config by best_val_loss)
# ---------------------------------------------------------------------------

def _grid_search_lstm(
    train_sequences: list[dict],
    val_sequences: list[dict],
    config: dict,
    fold: int,
    class_weights: "torch.Tensor",
    seed: int,
) -> tuple[int, int, float, float]:
    """
    Manual grid search over hidden_size × num_layers × dropout × lr.
    Runs each config for max_epochs=30 (fast proxy).
    Selects best config by best_val_loss — consistent with early stopping criterion.

    Returns:
        (best_hidden_size, best_num_layers, best_dropout, best_lr)
    """
    best_val_loss = float("inf")
    best_cfg = (
        config["hidden_size"][0],
        config["num_layers"][0],
        config["dropout"][0],
        config["learning_rate"][0],
        config["weight_decay"][0]
    )

    for hidden, layers, drop, lr, wd in product(
        config["hidden_size"],
        config["num_layers"],
        config["dropout"],
        config["learning_rate"],
        config["weight_decay"]
    ):
        _, val_loss = train_lstm_fold(
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            test_sequences=val_sequences,   # dummy — only val_loss used
            fold=fold,
            hidden_size=hidden,
            num_layers=layers,
            dropout=drop,
            learning_rate=lr,
            weight_decay=float(wd),
            batch_size=config.get("batch_size", 16),
            max_epochs=30,
            patience=config.get("patience", 15),
            class_weights=class_weights,
            seed=seed,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_cfg = (hidden, layers, drop, lr, float(wd))

    return best_cfg


# ---------------------------------------------------------------------------
# MLflow logging helpers
# ---------------------------------------------------------------------------

def _log_fold_metrics(fold_metrics: FoldMetrics) -> None:
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

def main(fast: bool = False, verbose: bool = False) -> None:
    random.seed(42)
    np.random.seed(42)

    print_section("T3.4 — LSTM Baseline")
    if fast:
        print(
            f"  ⚠️  FAST MODE — "
            f"B={BOOTSTRAP_REPLICATES_FAST} | n_select={MRMR_N_SELECT_FAST} | "
            f"tau={STABILITY_THRESHOLD_FAST}. Smoke test only."
        )

    seed, n_jobs = load_random_config(RANDOM_STATE_PATH)
    config = _load_lstm_config(LSTM_CONFIG_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    cv_splits = build_cv_splits(
        X=df[all_feature_cols],
        y=pd.Series(y),
        groups=groups,
        n_splits=5,
        seed=seed,
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    fold_metrics_list: list[FoldMetrics] = []
    fold_results_raw: list[dict] = []

    with mlflow.start_run(run_name="lstm_cv"):
        mlflow.log_param("seed", seed)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_param("n_splits", cv_splits.n_splits)
        mlflow.log_param("fast_mode", fast)
        mlflow.log_param("feature_set", "full_set_D_anchored")
        mlflow.log_param("val_split", "patient_stratified")

        for fold_split in cv_splits.folds:
            print_section(f"  Fold {fold_split.fold}")

            X_train_df = df.iloc[fold_split.train_idx]
            X_test_df  = df.iloc[fold_split.test_idx]
            y_train    = y[fold_split.train_idx]
            y_test     = y[fold_split.test_idx]

            # 1 — normalise: scaler fit on train only
            X_train_scaled, X_test_scaled = fit_transform_fold(
                X_train_df, X_test_df, all_feature_cols
            )

            # 2 — feature selection (cached — shared with LightGBM D)
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=all_feature_cols)
            selection: AnchoredFoldSelectionResult = select_features_fold_anchored_cached(
                X_train=X_train_scaled_df,
                y_train=y_train,
                fold=fold_split.fold,
                seed=seed,
                fast=fast,
                n_jobs=n_jobs,
                verbose=verbose,
            )

            feature_cols = selection.full_feature_set
            if not feature_cols:
                raise RuntimeError(
                    f"Fold {fold_split.fold}: empty full_feature_set. "
                    "Check feature_selector.py."
                )

            print(f"  {len(feature_cols)} features selected")
            mlflow.log_metric(f"fold_{fold_split.fold}_n_features", len(feature_cols))

            # 3 — extract selected features
            X_train_feat = X_train_scaled_df[feature_cols].values
            X_test_feat  = pd.DataFrame(
                X_test_scaled, columns=all_feature_cols
            )[feature_cols].values

            # 4 — patient-level stratified val split
            train_patient_ids = df.iloc[fold_split.train_idx]["Patient"].values
            train_idx_inner, val_idx_inner = _patient_train_val_split(
                patient_ids=train_patient_ids,
                y=y_train,
                val_fraction=VAL_FRACTION,
                seed=seed,
            )

            X_tr  = X_train_feat[train_idx_inner]
            y_tr  = y_train[train_idx_inner]
            X_val = X_train_feat[val_idx_inner]
            y_val = y_train[val_idx_inner]

            patients_tr   = train_patient_ids[train_idx_inner]
            patients_val  = train_patient_ids[val_idx_inner]
            patients_test = df.iloc[fold_split.test_idx]["Patient"].values

            scan_idx_tr   = df.iloc[fold_split.train_idx]["scan_index"].values[train_idx_inner]
            scan_idx_val  = df.iloc[fold_split.train_idx]["scan_index"].values[val_idx_inner]
            scan_idx_test = df.iloc[fold_split.test_idx]["scan_index"].values

            # 5 — build patient-level sequences
            train_seqs = build_patient_sequences(X_tr,       y_tr,    patients_tr,   scan_idx_tr)
            val_seqs   = build_patient_sequences(X_val,      y_val,   patients_val,  scan_idx_val)
            test_seqs  = build_patient_sequences(X_test_feat, y_test, patients_test, scan_idx_test)

            print(
                f"  Sequences — train: {len(train_seqs)} | "
                f"val: {len(val_seqs)} | test: {len(test_seqs)} patients"
            )

            # 6 — class weights from train fold only
            class_weights_fold = compute_class_weights(y_tr)

            # 7 — grid search (30-epoch proxy, best_val_loss criterion)
            best_hidden, best_layers, best_drop, best_lr, best_wd = _grid_search_lstm(
                train_sequences=train_seqs,
                val_sequences=val_seqs,
                config=config,
                fold=fold_split.fold,
                class_weights=class_weights_fold,
                seed=seed,
            )
            print(
                f"  Best config: hidden={best_hidden} layers={best_layers} "
                f"dropout={best_drop} lr={best_lr}"
            )

            # 8 — final train with best config + full patience
            result, _ = train_lstm_fold(
                train_sequences=train_seqs,
                val_sequences=val_seqs,
                test_sequences=test_seqs,
                fold=fold_split.fold,
                hidden_size=best_hidden,
                num_layers=best_layers,
                dropout=best_drop,
                learning_rate=best_lr,
                weight_decay=best_wd,
                batch_size=config.get("batch_size", 16),
                max_epochs=config.get("max_epochs", 100),
                patience=config.get("patience", 15),
                class_weights=class_weights_fold,
                seed=seed,
            )

            print(
                f"  Fold {fold_split.fold}: "
                f"macro_f1={result.metrics.macro_f1:.4f} | "
                f"mcc={result.metrics.mcc:.4f} | "
                f"epochs={result.n_epochs_trained}"
            )

            _log_fold_metrics(result.metrics)
            mlflow.log_metric(f"fold_{fold_split.fold}_epochs", result.n_epochs_trained)

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

        # --- Save JSON ---
        output = {
            "model": "lstm",
            "feature_set": "full_set_D_anchored",
            "seed": seed,
            "fold_results": fold_results_raw,
            "aggregated": asdict(aggregated),
        }
        report_path = OUTPUT_DIR / "lstm_results.json"
        with open(report_path, "w") as f:
            json.dump(output, f, indent=2)

        mlflow.log_artifact(str(report_path))
        print(f"\n  Report → {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM baseline (T3.4)")
    parser.add_argument("--fast", action="store_true",
                        help="Smoke test mode — reduced bootstrap and grid.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print top-50 features by bootstrap stability per fold.")
    args = parser.parse_args()
    main(fast=args.fast, verbose=args.verbose)
