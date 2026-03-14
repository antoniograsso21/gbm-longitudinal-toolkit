"""
src/audit/features_validator.py
================================
Runs 10 assertions on dataset_engineered.parquet to verify feature
engineering integrity.

Assertions:
    1.  Shape == (231, 2585)
    2.  Column groups: radiomic=1284, delta_radiomic=1284, all derived/temporal/flag cols present
    3.  Radiomic/delta symmetry: every radiomic col has its delta_ counterpart (0 orphans)
    4.  Binary flags: is_baseline_scan and is_nadir_scan are bool, not in radiomic pool;
        is_baseline_scan count == 64
    5.  No NaN or inf in any feature column
    6.  Delta features == 0.0 (not NaN) on all is_baseline_scan == True rows
    7.  Derived semantics: CE_vs_nadir >= 1.0, weeks_since_nadir >= 0,
        CE_fraction in [0, 1], total_tumor_volume >= 0
    8.  is_nadir_scan consistent with CE_vs_nadir
    9.  Label distribution matches expected (Progressive=175, Stable=25, Response=31)
    10. scan_index is 0-based and contiguous per patient

Usage:
    python -m src.audit.features_validator
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.lumiere_io import (
    SECTION,
    ValidationReport,
    print_section,
    save_validation_report,
    validation_result,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("data/processed")
PARQUET_PATH = OUTPUT_DIR / "dataset_engineered.parquet"
REPORT_PATH = OUTPUT_DIR / "features_validator_report.json"

EXPECTED_ROWS = 231
EXPECTED_COLS = 2585
EXPECTED_PATIENTS = 64
EXPECTED_BASELINE_SCANS = 64
EXPECTED_RADIOMIC_COLS = 1284
EXPECTED_DELTA_RADIOMIC_COLS = 1284
EXPECTED_CLASS_DIST = {"Progressive": 175, "Stable": 25, "Response": 31}

NON_FEATURE_COLS = [
    "Patient", "Timepoint",
    "target", "target_encoded",
    "is_baseline_scan",
    "is_nadir_scan",
]

DERIVED_CONTINUOUS = [
    "CE_NC_ratio", "ED_CE_ratio", "CE_fraction", "total_tumor_volume",
    "CE_vs_nadir", "weeks_since_nadir",
]

DELTA_DERIVED = ["delta_CE_NC_ratio", "delta_CE_vs_nadir"]

TEMPORAL_META = ["interval_weeks", "scan_index", "time_from_diagnosis_weeks"]

BINARY_FLAGS = ["is_baseline_scan", "is_nadir_scan"]


# ---------------------------------------------------------------------------
# Column group helpers
# ---------------------------------------------------------------------------
def _radiomic_absolute(df: pd.DataFrame) -> list[str]:
    exclude = set(NON_FEATURE_COLS + DERIVED_CONTINUOUS + DELTA_DERIVED + TEMPORAL_META)
    return [c for c in df.columns if c not in exclude and not c.startswith("delta_")]


def _delta_radiomic(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.startswith("delta_") and c not in DELTA_DERIVED]


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
def check_shape(df: pd.DataFrame) -> str:
    ok = df.shape == (EXPECTED_ROWS, EXPECTED_COLS)
    return validation_result("shape", ok,
                             f"got {df.shape}, expected ({EXPECTED_ROWS}, {EXPECTED_COLS})")


def check_column_groups(df: pd.DataFrame) -> str:
    radiomic = _radiomic_absolute(df)
    delta_rc = _delta_radiomic(df)
    msgs = []
    if len(radiomic) != EXPECTED_RADIOMIC_COLS:
        msgs.append(f"radiomic={len(radiomic)}, expected {EXPECTED_RADIOMIC_COLS}")
    if len(delta_rc) != EXPECTED_DELTA_RADIOMIC_COLS:
        msgs.append(f"delta_radiomic={len(delta_rc)}, expected {EXPECTED_DELTA_RADIOMIC_COLS}")
    for col in DERIVED_CONTINUOUS + DELTA_DERIVED + TEMPORAL_META + BINARY_FLAGS:
        if col not in df.columns:
            msgs.append(f"missing column: {col}")
    return validation_result("column_groups", len(msgs) == 0, "; ".join(msgs))


def check_radiomic_delta_symmetry(df: pd.DataFrame) -> str:
    radiomic = _radiomic_absolute(df)
    delta_rc = _delta_radiomic(df)
    missing = [c for c in radiomic if f"delta_{c}" not in df.columns]
    orphans = [c for c in delta_rc if c.replace("delta_", "", 1) not in radiomic]
    msgs = []
    if missing:
        msgs.append(f"{len(missing)} radiomic without delta: {missing[:3]}")
    if orphans:
        msgs.append(f"{len(orphans)} orphan deltas: {orphans[:3]}")
    return validation_result("radiomic_delta_symmetry", len(msgs) == 0, "; ".join(msgs))


def check_binary_flags(df: pd.DataFrame) -> str:
    msgs = []
    for col in BINARY_FLAGS:
        if df[col].dtype != bool:
            msgs.append(f"{col} dtype={df[col].dtype}, expected bool")
    radiomic = _radiomic_absolute(df)
    for flag in BINARY_FLAGS:
        if flag in radiomic:
            msgs.append(f"{flag} incorrectly in radiomic pool")
    n_baseline = int(df["is_baseline_scan"].sum())
    if n_baseline != EXPECTED_BASELINE_SCANS:
        msgs.append(f"is_baseline_scan count={n_baseline}, expected {EXPECTED_BASELINE_SCANS}")
    return validation_result("binary_flags", len(msgs) == 0, "; ".join(msgs))


def check_no_nan_inf(df: pd.DataFrame) -> str:
    radiomic = _radiomic_absolute(df)
    delta_rc = _delta_radiomic(df)
    all_feat = [c for c in radiomic + delta_rc + DERIVED_CONTINUOUS + DELTA_DERIVED + TEMPORAL_META
                if c in df.columns]
    n_nan = int(df[all_feat].isna().sum().sum())
    n_inf = int(np.isinf(df[all_feat].select_dtypes(include="number").values).sum())
    return validation_result("no_nan_inf", n_nan == 0 and n_inf == 0,
                             f"NaN={n_nan}, inf={n_inf}")


def check_delta_baseline(df: pd.DataFrame) -> str:
    delta_rc = _delta_radiomic(df)
    if not delta_rc:
        return validation_result("delta_baseline", False, "no delta radiomic columns found")
    baseline = df[df["is_baseline_scan"]]
    all_delta = delta_rc + DELTA_DERIVED
    nonzero = (baseline[all_delta] != 0.0).any().any()
    nan_present = baseline[all_delta].isna().any().any()
    msgs = []
    if nonzero:
        msgs.append("non-zero delta values on baseline scans")
    if nan_present:
        msgs.append("NaN delta values on baseline scans (expected 0.0 by design)")
    return validation_result("delta_baseline", len(msgs) == 0, "; ".join(msgs))


def check_derived_semantics(df: pd.DataFrame) -> str:
    tol = 1e-4
    msgs = []
    below = int((df["CE_vs_nadir"] < 1.0 - tol).sum())
    if below:
        msgs.append(f"CE_vs_nadir < 1.0 in {below} rows")
    neg_weeks = int((df["weeks_since_nadir"] < -tol).sum())
    if neg_weeks:
        msgs.append(f"weeks_since_nadir < 0 in {neg_weeks} rows")
    oor = int(((df["CE_fraction"] < 0) | (df["CE_fraction"] > 1 + tol)).sum())
    if oor:
        msgs.append(f"CE_fraction out of [0, 1] in {oor} rows")
    neg_vol = int((df["total_tumor_volume"] < 0).sum())
    if neg_vol:
        msgs.append(f"total_tumor_volume < 0 in {neg_vol} rows")
    return validation_result("derived_semantics", len(msgs) == 0, "; ".join(msgs))


def check_nadir_consistency(df: pd.DataFrame) -> str:
    tol = 1e-4
    inconsistent = int(((df["CE_vs_nadir"] <= 1.0 + tol) & ~df["is_nadir_scan"]).sum())
    return validation_result("nadir_consistency", inconsistent == 0,
                             f"{inconsistent} rows with CE_vs_nadir~1.0 but is_nadir_scan=False")


def check_label_distribution(df: pd.DataFrame) -> str:
    dist = df["target"].value_counts().to_dict()
    ok = all(dist.get(k, 0) == v for k, v in EXPECTED_CLASS_DIST.items())
    return validation_result("label_distribution", ok,
                             f"got {dist}, expected {EXPECTED_CLASS_DIST}")


def check_scan_index_contiguity(df: pd.DataFrame) -> str:
    violations = []
    for pat, grp in df.groupby("Patient"):
        indices = sorted(grp["scan_index"].tolist())
        if indices != list(range(len(indices))):
            violations.append(f"{pat}: {indices}")
    return validation_result("scan_index_contiguity", len(violations) == 0,
                             f"non-contiguous scan_index for: {violations[:3]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_section("LUMIERE Features Validation — Step 2")

    if not PARQUET_PATH.exists():
        print(f"ERROR: {PARQUET_PATH} not found. Run features_builder.py first.")
        return

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Patients: {df['Patient'].nunique()}")

    results: dict[str, str] = {}

    print_section("Running assertions")
    results["1_shape"]                   = check_shape(df)
    results["2_column_groups"]           = check_column_groups(df)
    results["3_radiomic_delta_symmetry"] = check_radiomic_delta_symmetry(df)
    results["4_binary_flags"]            = check_binary_flags(df)
    results["5_no_nan_inf"]              = check_no_nan_inf(df)
    results["6_delta_baseline"]          = check_delta_baseline(df)
    results["7_derived_semantics"]       = check_derived_semantics(df)
    results["8_nadir_consistency"]       = check_nadir_consistency(df)
    results["9_label_distribution"]      = check_label_distribution(df)
    results["10_scan_index_contiguity"]  = check_scan_index_contiguity(df)

    passed   = sum(1 for v in results.values() if v == "PASS")
    failed   = sum(1 for v in results.values() if v.startswith("FAIL"))
    warnings = sum(1 for v in results.values() if v.startswith("WARN"))

    print_section("SUMMARY")
    print(f"  PASS:    {passed}")
    print(f"  FAIL:    {failed}")
    print(f"  WARN:    {warnings}")

    class_dist = df["target"].value_counts().to_dict()
    report = ValidationReport(
        passed=passed,
        failed=failed,
        warnings=warnings,
        results=results,
        metadata={
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "n_patients": int(df["Patient"].nunique()),
            "class_distribution": {k: int(v) for k, v in class_dist.items()},
        },
    )

    if failed > 0:
        print("\n  ❌ VALIDATION FAILED — fix features_builder.py before Step 3")
    else:
        print("\n  ✅ VALIDATION PASSED — ready for Step 3")
    print(SECTION)

    save_validation_report(report, REPORT_PATH)


if __name__ == "__main__":
    main()