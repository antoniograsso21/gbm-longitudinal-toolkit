"""
src/training/run_lgbm_baseline.py
===================================
Entry point for the LightGBM baseline (Step 3 — T3.3).

Runs four ablations (A/B/C/D) inside StratifiedGroupKFold CV with
feature selection (mRMR + Stability Selection) inside each fold.

This is the only script that writes configs/selected_features.yaml —
produced from ablation D fold results via aggregate_fold_selections().
Rationale: Full set D, most stable on small n, SHAP validates selection.

Decision rules evaluated and logged after all ablations complete:
    If macro_F1(B) ≈ macro_F1(C): weak radiomic signal → flag in paper
    If macro_F1(B) > 0.38:        temporal leakage → flag in paper
    (0.38 > trivial macro_F1≈0.29 on 76/11/13 class distribution)
    If interval_weeks SHAP rank ≤ 5: temporal leakage → flag in paper

Side effects are confined to main(). All computation lives in
gbm_baseline.py, feature_selector.py, cross_validation.py,
and metrics.py (pure functions).

Output artifacts
----------------
data/processed/baselines/lgbm_{A|B|C|D}_results.json
configs/selected_features.yaml
data/processed/baselines/fold_stability.json
data/processed/interpretability/shap_top20.csv
data/processed/interpretability/shap_beeswarm.png

MLflow experiment: baselines/lgbm (single run, all ablations)

Usage
-----
    uv run python src/training/run_lgbm_baseline.py
    uv run python src/training/run_lgbm_baseline.py --fast
    uv run python src/training/run_lgbm_baseline.py --ablation D
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from src.models.lgbm_baseline import (
    AblationType,
    LGBMFoldResult,
    SHAPResult,
    build_ablation_feature_set,
    compute_shap,
    train_lgbm_fold,
)
from src.training.cross_validation import CVSplits, build_cv_splits
from src.training.feature_selector import (
    AnchoredFoldSelectionResult,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_REPLICATES_FAST,
    FoldSelectionResult,
    MRMR_N_SELECT,
    MRMR_N_SELECT_FAST,
    STABILITY_THRESHOLD,
    STABILITY_THRESHOLD_FAST,
    aggregate_fold_selections,
)
from src.training.training_utils import select_features_fold_anchored_cached
from src.training.metrics import AggregatedMetrics, FoldMetrics, aggregate_cv_results
from src.training.training_utils import (
    build_run_info,
    fit_transform_fold,
    load_random_config,
    split_train_val,
)
from src.utils.lumiere_io import build_full_feature_set, print_section

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PARQUET_PATH: Path = Path("data/processed/preprocessing/dataset_engineered.parquet")
OUTPUT_DIR: Path = Path("data/processed/baselines")
INTERP_DIR: Path = Path("data/processed/interpretability")
SELECTED_FEATURES_PATH: Path = Path("configs/selected_features.yaml")
FOLD_STABILITY_PATH: Path = Path("data/processed/baselines/fold_stability.json")
GBM_CONFIG_PATH: str = "configs/lgbm_baseline.yaml"
RANDOM_STATE_PATH: str = "configs/random_state.yaml"

ABLATIONS: list[AblationType] = ["A", "B", "C", "D"]
VAL_FRACTION: float = 0.1


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def _load_param_grid(config_path: str) -> tuple[dict, int]:
    """Load LightGBM hyperparameter grid and n_iter from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    param_grid = {
        "n_estimators":  cfg["n_estimators"],
        "max_depth":     cfg["max_depth"],
        "learning_rate": cfg["learning_rate"],
    }
    return param_grid, int(cfg.get("n_iter", 30))



# ---------------------------------------------------------------------------
# Result serialiser (excludes non-serialisable model field)
# ---------------------------------------------------------------------------

def _fold_result_to_dict(r: LGBMFoldResult) -> dict:
    """Serialise LGBMFoldResult to dict, excluding the fitted model object."""
    d = asdict(r)
    d.pop("model", None)
    return d


# ---------------------------------------------------------------------------
# MLflow logging helpers
# ---------------------------------------------------------------------------

def _log_fold_metrics(fold_metrics: FoldMetrics, ablation: AblationType) -> None:
    k = fold_metrics.fold
    prefix = f"{ablation}_fold_{k}"
    mlflow.log_metric(f"{prefix}_macro_f1",          fold_metrics.macro_f1)
    mlflow.log_metric(f"{prefix}_mcc",               fold_metrics.mcc)
    mlflow.log_metric(f"{prefix}_auroc_progressive", fold_metrics.auroc_progressive)
    mlflow.log_metric(f"{prefix}_auroc_stable",      fold_metrics.auroc_stable)
    mlflow.log_metric(f"{prefix}_auroc_response",    fold_metrics.auroc_response)
    mlflow.log_metric(f"{prefix}_prauc_progressive", fold_metrics.prauc_progressive)
    mlflow.log_metric(f"{prefix}_prauc_stable",      fold_metrics.prauc_stable)
    mlflow.log_metric(f"{prefix}_prauc_response",    fold_metrics.prauc_response)


def _log_aggregated_metrics(agg: AggregatedMetrics, ablation: AblationType) -> None:
    p = f"{ablation}_"
    mlflow.log_metric(f"{p}macro_f1_mean",          agg.macro_f1_mean)
    mlflow.log_metric(f"{p}macro_f1_std",           agg.macro_f1_std)
    mlflow.log_metric(f"{p}mcc_mean",               agg.mcc_mean)
    mlflow.log_metric(f"{p}mcc_std",                agg.mcc_std)
    mlflow.log_metric(f"{p}auroc_progressive_mean", agg.auroc_progressive_mean)
    mlflow.log_metric(f"{p}auroc_stable_mean",      agg.auroc_stable_mean)
    mlflow.log_metric(f"{p}auroc_response_mean",    agg.auroc_response_mean)
    mlflow.log_metric(f"{p}prauc_progressive_mean", agg.prauc_progressive_mean)
    mlflow.log_metric(f"{p}prauc_stable_mean",      agg.prauc_stable_mean)
    mlflow.log_metric(f"{p}prauc_response_mean",    agg.prauc_response_mean)


# ---------------------------------------------------------------------------
# SHAP I/O
# ---------------------------------------------------------------------------

def _save_shap_artifacts(shap_result: SHAPResult, output_dir: Path) -> tuple[Path, Path]:
    """Save SHAP top-20 CSV and bar chart PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paired = sorted(
        zip(shap_result.feature_names, shap_result.mean_abs_shap),
        key=lambda x: x[1],
        reverse=True,
    )[:20]

    top20_df = pd.DataFrame(paired, columns=["feature", "mean_abs_shap"])
    csv_path = output_dir / "shap_top20.csv"
    top20_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([p[0] for p in reversed(paired)], [p[1] for p in reversed(paired)], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-20 features by mean |SHAP| — LightGBM D (fold {shap_result.fold})")
    plt.tight_layout()
    png_path = output_dir / "shap_beeswarm.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    return csv_path, png_path


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def _evaluate_decision_rules(
    agg_results: dict[AblationType, AggregatedMetrics],
) -> dict[str, str]:
    """Evaluate radiomic signal and temporal leakage decision rules."""
    f1_B = agg_results["B"].macro_f1_mean
    f1_C = agg_results["C"].macro_f1_mean
    diff_BC = abs(f1_B - f1_C)

    verdicts: dict[str, str] = {}
    verdicts["radiomic_signal"] = (
        f"WEAK — F1(B)={f1_B:.4f} ≈ F1(C)={f1_C:.4f} (diff={diff_BC:.4f}). Declare in paper."
        if diff_BC < 0.02
        else f"PRESENT — F1(C)={f1_C:.4f} > F1(B)={f1_B:.4f} (diff={diff_BC:.4f})."
    )
    verdicts["temporal_leakage"] = (
        f"CONFIRMED — F1(B)={f1_B:.4f} > 0.38. Declare in paper."
        if f1_B > 0.38
        else f"NOT CONFIRMED — F1(B)={f1_B:.4f} ≤ 0.38."
    )
    return verdicts


# ---------------------------------------------------------------------------
# Single ablation CV runner
# ---------------------------------------------------------------------------

def _run_ablation_cv(
    ablation: AblationType,
    df: pd.DataFrame,
    all_feature_cols: list[str],
    cv_splits: CVSplits,
    param_grid: dict,
    n_iter: int,
    seed: int,
    n_jobs: int,
    fast: bool,
    verbose: bool = False,
) -> tuple[list[LGBMFoldResult], AggregatedMetrics, list[FoldSelectionResult]]:
    """
    Run full CV for a single ablation.

    Returns:
        fold_results:           per-fold LGBMFoldResult (includes fitted model)
        aggregated:             mean±std metrics across folds
        fold_selection_results: per-fold FoldSelectionResult (D only, else empty)
    """
    y = df["target_encoded"].values
    fold_results: list[LGBMFoldResult] = []
    fold_metrics_list: list[FoldMetrics] = []
    fold_selection_results: list[FoldSelectionResult] = []

    for fold_split in cv_splits.folds:
        X_train_df = df.iloc[fold_split.train_idx]
        X_test_df  = df.iloc[fold_split.test_idx]
        y_train    = y[fold_split.train_idx]
        y_test     = y[fold_split.test_idx]

        # 1 — normalise: fit on train only
        X_train_scaled, X_test_scaled = fit_transform_fold(
            X_train_df, X_test_df, all_feature_cols
        )

        # 2 — anchored feature selection on normalised train fold
        # Skip mRMR if ablation B is the only one requested.
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=all_feature_cols)
        if ablation != "B":
            selection: AnchoredFoldSelectionResult = select_features_fold_anchored_cached(
                X_train=X_train_scaled_df,
                y_train=y_train,
                fold=fold_split.fold,
                seed=seed,
                fast=fast,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            if ablation == "D":
                # Store radiomic-only FoldSelectionResult for YAML aggregation
                fold_selection_results.append(FoldSelectionResult(
                    fold=fold_split.fold,
                    selected_features=selection.selected_radiomic,
                    bootstrap_stability=selection.bootstrap_stability,
                    n_candidates=selection.n_radiomic_candidates,
                    n_selected=selection.n_radiomic_selected,
                    fast_mode=selection.fast_mode,
                ))
        else:
            # Ablation B: no feature selection needed — sentinel
            selection = AnchoredFoldSelectionResult(
                fold=fold_split.fold,
                selected_radiomic=[],
                anchored_delta=[],
                temporal_cols=[],
                full_feature_set=[],
                bootstrap_stability={},
                n_radiomic_candidates=0,
                n_radiomic_selected=0,
                n_delta_anchored=0,
                fast_mode=fast,
            )

        # 3 — build ablation-specific feature set from anchored selection
        feature_cols = build_ablation_feature_set(
            selection=selection,
            ablation=ablation,
        )

        # 4 — index by name (robust to column order changes)
        X_train_feat = X_train_scaled_df[feature_cols]
        X_test_feat  = pd.DataFrame(X_test_scaled, columns=all_feature_cols)[feature_cols]

        # 5 — split train into train + val for early stopping
        X_tr, y_tr, X_val, y_val = split_train_val(X_train_feat.values, y_train, VAL_FRACTION, seed)

        # 6 — train
        result = train_lgbm_fold(
            X_train=pd.DataFrame(X_tr, columns=feature_cols),
            y_train=y_tr,
            X_val=pd.DataFrame(X_val, columns=feature_cols),
            y_val=y_val,
            X_test=X_test_feat,
            y_test=y_test,
            fold=fold_split.fold,
            ablation=ablation,
            param_grid=param_grid,
            n_iter=n_iter,
            seed=seed,
        )

        fold_results.append(result)
        fold_metrics_list.append(result.metrics)

    aggregated = aggregate_cv_results(fold_metrics_list)
    return fold_results, aggregated, fold_selection_results


# ---------------------------------------------------------------------------
# SHAP runner (no re-fitting — uses already-fitted model from fold_results)
# ---------------------------------------------------------------------------

def _run_shap(
    fold_results_D: list[LGBMFoldResult],
    df: pd.DataFrame,
    all_feature_cols: list[str],
    cv_splits: CVSplits,
) -> None:
    """
    Compute and save SHAP for the best fold of ablation D.

    Uses the already-fitted model stored in fold_results_D — no re-fitting,
    no re-running feature selection. Best fold = highest macro_f1.
    """
    best_result = max(fold_results_D, key=lambda r: r.metrics.macro_f1)
    best_fold_idx = best_result.fold

    # Reconstruct test set for the best fold (scaler re-fit is deterministic)
    fold_split = cv_splits.folds[best_fold_idx]
    X_train_df = df.iloc[fold_split.train_idx]
    X_test_df  = df.iloc[fold_split.test_idx]

    # Scaler re-fit on train fold is deterministic — same result as during CV
    _, X_test_scaled = fit_transform_fold(
        X_train_df, X_test_df, all_feature_cols
    )

    # Feature names stored in the booster — no need to re-run selection
    feature_cols: list[str] = best_result.feature_cols
    X_te = pd.DataFrame(X_test_scaled, columns=all_feature_cols)[feature_cols].values

    print(f"feature_cols in best_result: {best_result.feature_cols}")
    shap_result = compute_shap(
        model=best_result.model,
        X_test=X_te,
        feature_names=feature_cols,
        fold=best_fold_idx,
    )

    csv_path, png_path = _save_shap_artifacts(shap_result, INTERP_DIR)
    mlflow.log_artifact(str(csv_path))
    mlflow.log_artifact(str(png_path))

    if shap_result.interval_weeks_rank is not None:
        mlflow.log_metric("D_interval_weeks_shap_rank", shap_result.interval_weeks_rank)
        if shap_result.interval_weeks_rank <= 5:
            print(
                f"  ⚠️  interval_weeks SHAP rank = {shap_result.interval_weeks_rank} ≤ 5 "
                "— temporal leakage must be declared in paper."
            )
    print(f"  SHAP saved → {csv_path}, {png_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False, ablations: list[AblationType] | None = None, verbose: bool = False) -> None:
    print_section("T3.3 — LightGBM Baseline (Ablations A/B/C/D)")
    if fast:
        print(
            f"  ⚠️  FAST MODE — "
            f"B={BOOTSTRAP_REPLICATES_FAST} | n_select={MRMR_N_SELECT_FAST} | "
            f"tau={STABILITY_THRESHOLD_FAST}. Smoke test only, not production."
        )

    if ablations is None:
        ablations = ABLATIONS

    seed, n_jobs = load_random_config(RANDOM_STATE_PATH)
    param_grid, n_iter = _load_param_grid(GBM_CONFIG_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERP_DIR.mkdir(parents=True, exist_ok=True)

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

    # --- Minimal run provenance for JSON (full params live in MLflow) ---
    run_info = build_run_info(
        seed=seed,
        parquet_path=str(PARQUET_PATH.as_posix()),
        n_rows=int(df.shape[0]),
        n_patients=int(groups.nunique()),
        script_path=str(Path(__file__).as_posix()),
    )

    cv_splits = build_cv_splits(
        X=df[all_feature_cols],
        y=pd.Series(y),
        groups=groups,
        n_splits=5,
        seed=seed,
    )

    mlflow.set_experiment("baselines/lgbm")

    agg_results: dict[AblationType, AggregatedMetrics] = {}
    # Store fold results per ablation — D results needed for SHAP and YAML
    all_fold_results: dict[AblationType, list[LGBMFoldResult]] = {}
    d_fold_selection_results: list[FoldSelectionResult] = []

    with mlflow.start_run(run_name="lgbm_ablations"):
        mlflow.log_param("seed", seed)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_param("ablations", str(ablations))
        mlflow.log_param("fast_mode", fast)

        for ablation in ablations:
            print_section(f"Ablation {ablation}")

            fold_results, aggregated, fold_sel = _run_ablation_cv(
                ablation=ablation,
                df=df,
                all_feature_cols=all_feature_cols,
                cv_splits=cv_splits,
                param_grid=param_grid,
                n_iter=n_iter,
                seed=seed,
                n_jobs=n_jobs,
                fast=fast,
                verbose=verbose,
            )

            all_fold_results[ablation] = fold_results
            if ablation == "D":
                d_fold_selection_results = fold_sel

            # Log fold-level metrics and print progress
            for r in fold_results:
                _log_fold_metrics(r.metrics, ablation)
                print(
                    f"  [{ablation}] Fold {r.fold}: "
                    f"{r.n_features} features | "
                    f"macro_f1={r.metrics.macro_f1:.4f} | "
                    f"mcc={r.metrics.mcc:.4f}"
                )

            agg_results[ablation] = aggregated
            _log_aggregated_metrics(aggregated, ablation)

            print(
                f"  [{ablation}] macro_f1: "
                f"{aggregated.macro_f1_mean:.4f} ± {aggregated.macro_f1_std:.4f}"
            )

            # Save per-ablation JSON (model field excluded — not serialisable)
            report = {
                "schema_version": "baselines.v1",
                "model": "lgbm",
                "ablation": ablation,
                "seed": seed,
                "run_info": run_info,
                "fold_results": [_fold_result_to_dict(r) for r in fold_results],
                "aggregated": asdict(aggregated),
            }
            report_path = OUTPUT_DIR / f"lgbm_{ablation}_results.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(str(report_path))

        # --- Decision rules (requires both B and C) ---
        if "B" in ablations and "C" in ablations:
            print_section("Decision Rules")
            verdicts = _evaluate_decision_rules(agg_results)
            for rule, verdict in verdicts.items():
                print(f"  {rule}: {verdict}")
                mlflow.log_param(f"decision_{rule}", verdict)

        # --- selected_features.yaml from ablation D (production only) ---
        if "D" in ablations and not fast:
            if not d_fold_selection_results:
                raise RuntimeError(
                    "Ablation D ran but produced no fold selection results. "
                    "This is a bug — check _run_ablation_cv."
                )
            aggregated_selection = aggregate_fold_selections(d_fold_selection_results)

            SELECTED_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "selected_features": aggregated_selection.selected_features,
                "n_selected": len(aggregated_selection.selected_features),
                "majority_vote_threshold": 3,
                "n_folds": len(d_fold_selection_results),
            }
            with open(SELECTED_FEATURES_PATH, "w") as f:
                yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
            with open(FOLD_STABILITY_PATH, "w") as f:
                json.dump(aggregated_selection.fold_stability, f, indent=2)

            mlflow.log_artifact(str(SELECTED_FEATURES_PATH))
            mlflow.log_artifact(str(FOLD_STABILITY_PATH))
            mlflow.log_metric("D_n_selected_final", len(aggregated_selection.selected_features))
            print(f"\n  selected_features.yaml → {len(aggregated_selection.selected_features)} features")

        # --- SHAP on best fold of ablation D ---
        if "D" in ablations and "D" in all_fold_results:
            print_section("SHAP — Ablation D")
            _run_shap(
                fold_results_D=all_fold_results["D"],
                df=df,
                all_feature_cols=all_feature_cols,
                cv_splits=cv_splits,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM baseline (T3.3)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print top-50 features by bootstrap stability per fold (tau calibration).",
    )
    parser.add_argument("--fast", action="store_true",
                        help=(
                            f"Smoke test mode: B={BOOTSTRAP_REPLICATES_FAST}, "
                            f"n_select={MRMR_N_SELECT_FAST}, tau={STABILITY_THRESHOLD_FAST}. "
                            "Never use for production."
                        ))
    parser.add_argument("--ablation", choices=["A", "B", "C", "D"], default=None,
                        help="Run a single ablation only (default: all).")
    args = parser.parse_args()
    main(fast=args.fast, ablations=[args.ablation] if args.ablation else None, verbose=args.verbose)