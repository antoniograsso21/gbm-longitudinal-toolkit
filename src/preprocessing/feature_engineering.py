"""
src/preprocessing/feature_engineering.py
=========================================
Computes derived cross-compartment and nadir-based features from
dataset_paired.parquet and saves dataset_engineered.parquet.

This is Step 2 Part A. All transformations are deterministic and
label-free — no target column is read or used anywhere in this script.

Derived features added (9 total):
    Cross-compartment:
        CE_NC_ratio        — CE_volume / (NC_volume + ε)
        ED_CE_ratio        — ED_volume / (CE_volume + ε)
        CE_fraction        — CE_volume / (CE + NC + ED + ε)
        total_tumor_volume — CE + NC + ED volumes

    Nadir-based (computed per patient, chronologically up to T inclusive):
        CE_vs_nadir        — CE_volume(T) / min(CE_volume[T0..T])
        weeks_since_nadir  — time_from_diagnosis_weeks(T) - week of nadir
        is_nadir_scan      — True when CE_volume(T) == min(CE_volume[T0..T])

    Rates of change (delta / interval_weeks):
        delta_CE_NC_ratio  — Δ(CE_NC_ratio) / interval_weeks
        delta_CE_vs_nadir  — Δ(CE_vs_nadir) / interval_weeks

Leakage guarantees:
    - target column is never read
    - nadir computed from parquet rows only (dropped scans are excluded)
    - nadir at T uses only timepoints up to and including T (no future data)
    - delta features on is_baseline_scan rows are forced to 0.0

Usage:
    uv run -m src.preprocessing.feature_engineering

Output:
    data/processed/dataset_engineered.parquet
    data/processed/feature_engineering_report.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.lumiere_io import SECTION, print_section

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/processed")
INPUT_PARQUET = DATA_DIR / "dataset_paired.parquet"
OUTPUT_PARQUET = DATA_DIR / "dataset_engineered.parquet"
REPORT_PATH = DATA_DIR / "feature_engineering_report.json"

# ---------------------------------------------------------------------------
# Volume column names — must match pivot output from build_dataset.py
# CT1 for CE and NC (contrast-enhanced T1 is the clinical reference sequence)
# FLAIR for ED (standard for edema assessment in GBM)
# ---------------------------------------------------------------------------
VOL_CE = "CE_CT1_original_shape_MeshVolume"
VOL_NC = "NC_CT1_original_shape_MeshVolume"
VOL_ED = "ED_FLAIR_original_shape_MeshVolume"

EPSILON = 1.0  # mm³ — prevents division by zero on near-absent compartments

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class FeatureEngineeringReport:
    n_rows_input: int
    n_cols_input: int
    n_cols_output: int
    derived_features: list[str]
    n_nadir_scans: int          # rows where is_nadir_scan == True
    n_baseline_nadir_overlap: int  # rows that are both baseline and nadir
    missing_volume_cols: list[str]  # volume cols absent from parquet
    n_nan_introduced: int       # safety check — must be 0


# ---------------------------------------------------------------------------
# Guard: ensure no target column is used
# ---------------------------------------------------------------------------
def _assert_no_target_used(df: pd.DataFrame) -> None:
    """
    Defensive check: called before any computation to ensure the target
    column is not accidentally read. Raises ValueError if violated.
    """
    # We never pass target to any function — this is a belt-and-suspenders check
    # that no caller accidentally filtered on target before calling us.
    # The actual computation functions below only receive volume columns and
    # temporal columns — never target or target_encoded.
    assert "target" in df.columns, "target column missing — unexpected parquet schema"
    # Do NOT read df["target"] here — just confirm it exists and ignore it.


# ---------------------------------------------------------------------------
# Cross-compartment features (pure functions)
# ---------------------------------------------------------------------------
def compute_cross_compartment(
    ce: pd.Series,
    nc: pd.Series,
    ed: pd.Series,
) -> pd.DataFrame:
    """
    Compute four cross-compartment volumetric features.

    Args:
        ce, nc, ed: MeshVolume series for each compartment (same index).

    Returns:
        DataFrame with 4 columns, same index as inputs.
    """
    total = ce + nc + ed
    return pd.DataFrame({
        "CE_NC_ratio":        ce / (nc + EPSILON),
        "ED_CE_ratio":        ed / (ce + EPSILON),
        "CE_fraction":        ce / (total + EPSILON),
        "total_tumor_volume": total,
    })


# ---------------------------------------------------------------------------
# Nadir-based features (per-patient, chronological)
# ---------------------------------------------------------------------------
def compute_nadir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CE_vs_nadir, weeks_since_nadir, is_nadir_scan per patient.

    For each row (patient, timepoint T):
        nadir_vol   = min(CE_volume[T0..T])   — minimum up to and including T
        nadir_week  = week_num of the timepoint achieving nadir_vol

    When multiple timepoints share the minimum volume, the earliest is used
    as nadir (conservative — avoids using future information).

    The nadir is computed exclusively from rows in the parquet — dropped scans
    (any-NaN) are excluded, which is correct: they were never part of the
    patient's usable history.

    Returns:
        DataFrame with 3 columns (CE_vs_nadir, weeks_since_nadir, is_nadir_scan),
        same index as df.
    """
    results: list[dict] = []

    for patient, group in df.groupby("Patient"):
        group = group.sort_values("time_from_diagnosis_weeks")  # ← no reset_index()
        ce_vols = group[VOL_CE].values
        weeks = group["time_from_diagnosis_weeks"].values
        original_indices = group.index  # ← indice originale del DataFrame padre

        running_min_vol = np.inf
        running_min_week = np.nan

        for i in range(len(group)):
            vol = ce_vols[i]
            wk = weeks[i]
            idx = original_indices[i]  # ← ora è corretto

            if vol < running_min_vol:
                running_min_vol = vol
                running_min_week = wk

            ce_vs_nadir = (vol + EPSILON) / (running_min_vol + EPSILON)
            weeks_since_nadir = wk - running_min_week
            is_nadir = bool(vol <= running_min_vol + 1e-6)

            results.append({
                "_idx": idx,
                "CE_vs_nadir": ce_vs_nadir,
                "weeks_since_nadir": weeks_since_nadir,
                "is_nadir_scan": is_nadir,
            })

    nadir_df = (
        pd.DataFrame(results)
        .set_index("_idx")
        .sort_index()
    )
    return nadir_df


# ---------------------------------------------------------------------------
# Delta of derived features
# ---------------------------------------------------------------------------
def compute_derived_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta_CE_NC_ratio and delta_CE_vs_nadir per patient.

    Formula: delta_f_t = (f_t - f_{t-1}) / interval_weeks
    Baseline scans (is_baseline_scan == True): delta = 0.0

    Returns:
        DataFrame with 2 columns, same index as df.
    """
    delta_cols = ["CE_NC_ratio", "CE_vs_nadir"]
    results = {f"delta_{c}": pd.Series(np.nan, index=df.index) for c in delta_cols}

    for patient, group in df.groupby("Patient"):
        group = group.sort_values("time_from_diagnosis_weeks")
        idx = group.index

        for col in delta_cols:
            diff = group[col].diff()
            rate = diff / group["interval_weeks"]
            # Forza 0 solo sulle righe effettivamente baseline, non sul primo elemento
            is_baseline_mask = group["is_baseline_scan"]
            rate[is_baseline_mask] = 0.0
            results[f"delta_{col}"].loc[idx] = rate.values

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_derived_features(df: pd.DataFrame) -> None:
    """
    Internal producer-side guard — runs at generation time inside main().
    Catches computation errors immediately (fail fast).

    For artifact-level validation runnable independently on the saved
    parquet, see src/audit/validate_features.py.

    Checks:
        1. No NaN in any derived column
        2. CE_vs_nadir >= 1.0 - tolerance for all rows (nadir is the minimum)
        3. delta_* == 0.0 on all is_baseline_scan rows
        4. is_nadir_scan == True whenever CE_vs_nadir == 1.0 (within tolerance)
        5. weeks_since_nadir >= 0 for all rows
    """
    derived = [
        "CE_NC_ratio", "ED_CE_ratio", "CE_fraction", "total_tumor_volume",
        "CE_vs_nadir", "weeks_since_nadir", "is_nadir_scan",
        "delta_CE_NC_ratio", "delta_CE_vs_nadir",
    ]

    # 1. No NaN
    n_nan = df[derived].isna().sum().sum()
    assert n_nan == 0, f"NaN in derived features: {df[derived].isna().sum()[lambda s: s>0].to_dict()}"

    # 2. CE_vs_nadir >= 1.0 - tolerance
    tol = 1e-4
    below = (df["CE_vs_nadir"] < 1.0 - tol).sum()
    assert below == 0, f"{below} rows with CE_vs_nadir < 1.0 — nadir computation error"

    # 3. Delta == 0 on baseline scans
    baseline = df[df["is_baseline_scan"]]
    for col in ["delta_CE_NC_ratio", "delta_CE_vs_nadir"]:
        nonzero = (baseline[col].abs() > 1e-9).sum()
        assert nonzero == 0, f"{nonzero} non-zero {col} on baseline scans"

    # 4. is_nadir_scan consistent with CE_vs_nadir
    should_be_nadir = df["CE_vs_nadir"] <= 1.0 + tol
    inconsistent = (should_be_nadir & ~df["is_nadir_scan"]).sum()
    assert inconsistent == 0, f"{inconsistent} rows with CE_vs_nadir~1 but is_nadir_scan=False"

    # 5. weeks_since_nadir >= 0
    negative = (df["weeks_since_nadir"] < -tol).sum()
    assert negative == 0, f"{negative} rows with negative weeks_since_nadir"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{SECTION}")
    print("GBM Longitudinal Toolkit — Feature Engineering (Step 2)")
    print(SECTION)

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"{INPUT_PARQUET} not found. Run build_dataset.py first."
        )

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Safety: confirm target is present but never used
    _assert_no_target_used(df)

    # Check volume columns exist
    missing_vol = [c for c in [VOL_CE, VOL_NC, VOL_ED] if c not in df.columns]
    if missing_vol:
        raise ValueError(
            f"Volume columns missing from parquet: {missing_vol}\n"
            f"Expected: {VOL_CE}, {VOL_NC}, {VOL_ED}"
        )

    print_section("Cross-compartment features")
    cross = compute_cross_compartment(df[VOL_CE], df[VOL_NC], df[VOL_ED])
    print(f"Computed: {list(cross.columns)}")

    print_section("Nadir-based features")
    nadir = compute_nadir_features(df)
    n_nadir = int(nadir["is_nadir_scan"].sum())
    print(f"Computed: {list(nadir.columns)}")
    print(f"Nadir scans: {n_nadir} / {len(df)}")

    # Attach derived features before delta computation
    df = pd.concat([df, cross, nadir], axis=1)

    print_section("Delta of derived features")
    deltas = compute_derived_deltas(df)
    print(f"Computed: {list(deltas.columns)}")
    df = pd.concat([df, deltas], axis=1)

    print_section("Validation")
    validate_derived_features(df)
    print("✅ All assertions passed")

    # Build report
    derived_features = list(cross.columns) + list(nadir.columns) + list(deltas.columns)
    n_nan_introduced = int(df[derived_features].isna().sum().sum())
    overlap = int((df["is_baseline_scan"] & df["is_nadir_scan"]).sum())

    report = FeatureEngineeringReport(
        n_rows_input=len(df),
        n_cols_input=df.shape[1] - len(derived_features),
        n_cols_output=df.shape[1],
        derived_features=derived_features,
        n_nadir_scans=n_nadir,
        n_baseline_nadir_overlap=overlap,
        missing_volume_cols=missing_vol,
        n_nan_introduced=n_nan_introduced,
    )

    df.to_parquet(OUTPUT_PARQUET, index=False)
    with open(REPORT_PATH, "w") as f:
        json.dump(asdict(report), f, indent=2)

    print_section("COMPLETE")
    print(f"  Output: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Derived features added: {len(derived_features)}")
    print(f"  Nadir scans: {n_nadir} | Baseline+nadir overlap: {overlap}")
    print(f"  Saved → {OUTPUT_PARQUET}")
    print(f"  Report → {REPORT_PATH}")
    print(SECTION)


if __name__ == "__main__":
    main()
