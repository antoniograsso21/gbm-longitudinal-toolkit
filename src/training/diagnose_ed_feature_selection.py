"""
src/training/diagnose_ed_feature_selection.py
=============================================
Non-production diagnostic for Step 4 ED feature representation.

Runs the MI-univariate selector inside the same CV/scaling pattern used by
Step 3, but compares multiple percentiles without changing
configs/feature_selector.yaml or any production cache. The goal is to decide
whether the weak ED node in Step 4 is an expected limitation or a sign that the
production selector should be revisited.

Usage:
    uv run python -m src.training.diagnose_ed_feature_selection
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.training.cross_validation import build_cv_splits
from src.training.feature_selector_mi import select_features_fold_mi
from src.training.training_utils import fit_transform_fold, load_random_config
from src.utils.lumiere_io import build_full_feature_set, print_section

PARQUET_PATH = Path("data/processed/preprocessing/dataset_engineered.parquet")
OUTPUT_DIR = Path("data/processed/diagnostics")
REPORT_PATH = OUTPUT_DIR / "ed_feature_selection_diagnostic.json"
RANDOM_STATE_PATH = "configs/random_state.yaml"

COMPARTMENTS = ("CE", "NC", "ED")


def _compartment(feature: str) -> str:
    base = feature.removeprefix("delta_")
    return base.split("_", 1)[0]


def _family(feature: str) -> str:
    base = feature.removeprefix("delta_")
    parts = base.split("_")
    try:
        original_idx = parts.index("original")
    except ValueError:
        return "derived"
    if original_idx + 1 >= len(parts):
        return "unknown"
    return parts[original_idx + 1]


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {k: int(counter.get(k, 0)) for k in sorted(counter)}


def _summarise_percentile(
    df: pd.DataFrame,
    all_feature_cols: list[str],
    percentile: float,
    seed: int,
    n_neighbors: int,
) -> dict:
    y = df["target_encoded"].values
    groups = df["Patient"]
    cv_splits = build_cv_splits(
        X=df[all_feature_cols],
        y=pd.Series(y),
        groups=groups,
        n_splits=5,
        seed=seed,
    )

    fold_summaries: list[dict] = []
    feature_counts: Counter[str] = Counter()
    ed_feature_counts: Counter[str] = Counter()
    all_family_counts: Counter[str] = Counter()
    ed_family_counts: Counter[str] = Counter()
    compartment_fold_presence: Counter[str] = Counter()

    for fold_split in cv_splits.folds:
        X_train_df = df.iloc[fold_split.train_idx]
        X_test_df = df.iloc[fold_split.test_idx]
        y_train = y[fold_split.train_idx]

        X_train_scaled, _ = fit_transform_fold(X_train_df, X_test_df, all_feature_cols)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=all_feature_cols)

        selection = select_features_fold_mi(
            X_train=X_train_scaled_df,
            y_train=y_train,
            fold=fold_split.fold,
            percentile=percentile,
            n_neighbors=n_neighbors,
            seed=seed,
            fast=False,
            verbose=False,
            justification="ED diagnostic only; production config unchanged.",
        )

        selected = selection.selected_radiomic
        feature_counts.update(selected)
        comp_counts = Counter(_compartment(f) for f in selected)
        family_counts = Counter(_family(f) for f in selected)
        ed_selected = [f for f in selected if _compartment(f) == "ED"]

        for compartment in COMPARTMENTS:
            if comp_counts.get(compartment, 0) > 0:
                compartment_fold_presence[compartment] += 1

        ed_feature_counts.update(ed_selected)
        all_family_counts.update(_family(f) for f in selected)
        ed_family_counts.update(_family(f) for f in ed_selected)

        fold_summaries.append({
            "fold": int(fold_split.fold),
            "n_selected": int(len(selected)),
            "compartment_counts": {
                c: int(comp_counts.get(c, 0)) for c in COMPARTMENTS
            },
            "family_counts": _counter_dict(family_counts),
            "ed_family_counts": _counter_dict(Counter(_family(f) for f in ed_selected)),
            "ed_features": sorted(ed_selected),
        })

    majority_features = sorted([
        feature for feature, count in feature_counts.items()
        if count >= 3
    ])
    majority_by_compartment = Counter(_compartment(f) for f in majority_features)
    majority_by_family = Counter(_family(f) for f in majority_features)
    ed_majority_features = [
        f for f in majority_features if _compartment(f) == "ED"
    ]

    return {
        "percentile": float(percentile),
        "n_neighbors": int(n_neighbors),
        "folds": fold_summaries,
        "compartment_fold_presence": {
            c: int(compartment_fold_presence.get(c, 0)) for c in COMPARTMENTS
        },
        "selected_feature_frequency": {
            k: int(v) for k, v in sorted(feature_counts.items())
        },
        "ed_feature_frequency": {
            k: int(v) for k, v in sorted(ed_feature_counts.items())
        },
        "family_counts_all_selected": _counter_dict(all_family_counts),
        "family_counts_ed_selected": _counter_dict(ed_family_counts),
        "majority_vote_threshold": 3,
        "majority_features": majority_features,
        "majority_counts_by_compartment": {
            c: int(majority_by_compartment.get(c, 0)) for c in COMPARTMENTS
        },
        "majority_counts_by_family": _counter_dict(majority_by_family),
        "ed_majority_features": ed_majority_features,
        "ed_majority_counts_by_family": _counter_dict(
            Counter(_family(f) for f in ed_majority_features)
        ),
    }


def main(percentiles: list[float], n_neighbors: int = 5) -> None:
    print_section("Step 4 — ED Feature Selection Diagnostic")
    seed, _ = load_random_config(RANDOM_STATE_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    all_feature_cols = build_full_feature_set(df)

    report = {
        "schema_version": "ed_feature_selection_diagnostic.v1",
        "parquet": str(PARQUET_PATH),
        "seed": int(seed),
        "n_rows": int(df.shape[0]),
        "n_patients": int(df["Patient"].nunique()),
        "note": (
            "Diagnostic only: production configs/feature_selector.yaml and "
            "selected_features.yaml are unchanged."
        ),
        "percentiles": {},
    }

    for percentile in percentiles:
        print(f"  Running percentile={percentile:g}%")
        summary = _summarise_percentile(
            df=df,
            all_feature_cols=all_feature_cols,
            percentile=percentile,
            seed=seed,
            n_neighbors=n_neighbors,
        )
        key = f"{percentile:g}"
        report["percentiles"][key] = summary
        majority_ed = summary["majority_counts_by_compartment"]["ED"]
        ed_shape = summary["ed_majority_counts_by_family"].get("shape", 0)
        print(
            f"    ED majority features: {majority_ed} "
            f"(shape={ed_shape})"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"\n  Report -> {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ED feature-selection diagnostic")
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=[5.0, 10.0],
        help="MI top-percentiles to evaluate without changing production config.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=5,
        help="n_neighbors for sklearn mutual_info_classif.",
    )
    args = parser.parse_args()
    main(percentiles=args.percentiles, n_neighbors=args.n_neighbors)
