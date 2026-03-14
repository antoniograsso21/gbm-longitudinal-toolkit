"""
src/audit/validate_preprocessing.py
=====================================
Runs 11 assertions on dataset_paired.parquet to verify preprocessing integrity.

Assertions:
    1.  n_effective == 231
    2.  No patient's last timepoint in the dataset (scan_index contiguous per patient)
    3.  No NaN or inf in any feature column
    4.  Label distribution matches expected (Progressive=175, Stable=25, Response=31)
    5.  Delta features == 0.0 on all is_baseline_scan == True rows
    6.  Log-transform applied correctly: transformed cols >= 0; excluded cols
        (Skewness, Imc1) retain negative values
    7.  No future information: interval_weeks > 0 for all rows
    8.  Patient-039 does NOT appear in the dataset
    9.  Column 'interval_weeks' exists; no column named 'delta_t_weeks'
    10. No duplicate (Patient, Timepoint) rows
    11. time_from_diagnosis_weeks is strictly increasing per patient
    W1. Survival bias check (informational — WARN not FAIL)

Usage:
    python -m src.audit.validate_preprocessing
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.lumiere_io import (
    SECTION,
    ValidationReport,
    LOG_TRANSFORM_EXCLUDE,
    PATIENTS_EXCLUDED,
    RADIOMIC_PREFIX,
    print_section,
    radiomic_cols,
    feature_suffix,
    save_validation_report,
    validation_result,
    validation_warn,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("data/processed")
PARQUET_PATH = OUTPUT_DIR / "dataset_paired.parquet"
REPORT_PATH = OUTPUT_DIR / "validation_preprocessing_report.json"

EXPECTED_N_EFFECTIVE = 231
EXPECTED_CLASS_DIST = {"Progressive": 175, "Stable": 25, "Response": 31}
PATIENT_LOST = "Patient-039"

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
def check_n_effective(df: pd.DataFrame) -> str:
    n = len(df)
    return validation_result(
        "n_effective", n == EXPECTED_N_EFFECTIVE,
        f"got {n}, expected {EXPECTED_N_EFFECTIVE}",
    )


def check_no_last_timepoint(df: pd.DataFrame) -> str:
    """
    Verify label shift integrity: scan_index must be 0-based and contiguous
    per patient. A gap indicates a wrongly dropped non-last timepoint.
    Also verify target is never NaN.
    """
    violations = []
    for pat, grp in df.groupby("Patient"):
        indices = sorted(grp["scan_index"].tolist())
        if indices != list(range(len(indices))):
            violations.append(f"{pat}: {indices}")
    nan_targets = int(df["target"].isna().sum())
    if nan_targets > 0:
        violations.append(f"{nan_targets} rows with NaN target")
    return validation_result(
        "no_last_timepoint", len(violations) == 0,
        f"scan_index not contiguous for: {violations[:3]}",
    )


def check_no_nan_inf(df: pd.DataFrame) -> str:
    rc = radiomic_cols(df)
    numeric_cols = [
        c for c in rc + ["interval_weeks", "time_from_diagnosis_weeks",
                         "scan_index", "target_encoded"]
        if c in df.columns
    ]
    n_nan = int(df[numeric_cols].isna().sum().sum())
    n_inf = int(np.isinf(df[numeric_cols].select_dtypes(include="number").values).sum())
    return validation_result("no_nan_inf", n_nan == 0 and n_inf == 0,
                             f"NaN={n_nan}, inf={n_inf}")


def check_label_distribution(df: pd.DataFrame) -> str:
    dist = df["target"].value_counts().to_dict()
    ok = all(dist.get(k, 0) == v for k, v in EXPECTED_CLASS_DIST.items())
    return validation_result("label_distribution", ok,
                             f"got {dist}, expected {EXPECTED_CLASS_DIST}")


def check_delta_baseline(df: pd.DataFrame) -> str:
    delta_cols = [c for c in df.columns
                  if c.startswith("delta_") and RADIOMIC_PREFIX in c]
    if not delta_cols:
        return validation_result("delta_baseline", False, "no delta columns found")
    baseline = df[df["is_baseline_scan"]]
    nonzero = (baseline[delta_cols] != 0).any().any()
    return validation_result("delta_baseline", not nonzero,
                             "non-zero delta features on baseline scans")


def check_log_transform(df: pd.DataFrame) -> str:
    rc = radiomic_cols(df)
    excluded_suffixes = LOG_TRANSFORM_EXCLUDE
    transformed_cols = [
        c for c in rc
        if feature_suffix(c) not in excluded_suffixes
        and not c.startswith("delta_")
    ]
    excluded_cols = [c for c in rc if feature_suffix(c) in excluded_suffixes]

    if not transformed_cols:
        return validation_result("log_transform", False, "no transformed columns found")

    # log1p output must be >= 0
    neg_vals = [c for c in transformed_cols if df[c].min() < 0]
    if neg_vals:
        return validation_result("log_transform", False,
                                 f"{len(neg_vals)} transformed cols have negatives: {neg_vals[:3]}")

    # Skewness and Imc1 excluded cols must still contain negatives
    suspicious = [
        c for c in excluded_cols
        if ("Skewness" in feature_suffix(c) or "Imc1" in feature_suffix(c))
        and df[c].min() >= 0
    ]
    return validation_result("log_transform", len(suspicious) == 0,
                             f"excluded cols wrongly transformed (no negatives): {suspicious[:3]}")


def check_no_future_info(df: pd.DataFrame) -> str:
    bad = int((df["interval_weeks"] <= 0).sum())
    return validation_result("no_future_info", bad == 0,
                             f"{bad} rows with interval_weeks <= 0")


def check_patient_039_absent(df: pd.DataFrame) -> str:
    present = PATIENT_LOST in df["Patient"].values
    return validation_result("patient_039_absent", not present,
                             f"{PATIENT_LOST} found in dataset")


def check_column_names(df: pd.DataFrame) -> str:
    msgs = []
    if "interval_weeks" not in df.columns:
        msgs.append("interval_weeks missing")
    if "delta_t_weeks" in df.columns:
        msgs.append("delta_t_weeks present (old name)")
    return validation_result("column_names", len(msgs) == 0, "; ".join(msgs))


def check_no_duplicate_pairs(df: pd.DataFrame) -> str:
    dupes = int(df.duplicated(subset=["Patient", "Timepoint"]).sum())
    return validation_result("no_duplicate_pairs", dupes == 0,
                             f"{dupes} duplicate (Patient, Timepoint) rows")


def check_week_monotonic(df: pd.DataFrame) -> str:
    violations = []
    for pat, grp in df.groupby("Patient"):
        weeks = grp.sort_values("scan_index")["time_from_diagnosis_weeks"].tolist()
        if weeks != sorted(set(weeks)):
            violations.append(pat)
    return validation_result("week_monotonic", len(violations) == 0,
                             f"non-monotonic weeks for: {violations}")


def check_survival_bias(df: pd.DataFrame) -> tuple[str, dict]:
    means = df.groupby("target")["time_from_diagnosis_weeks"].mean()
    spread = float(means.max() - means.min())

    print(f"\n  time_from_diagnosis_weeks by class:")
    for cls in ["Progressive", "Stable", "Response"]:
        if cls in df["target"].values:
            m = df[df["target"] == cls]["time_from_diagnosis_weeks"].mean()
            s = df[df["target"] == cls]["time_from_diagnosis_weeks"].std()
            print(f"    {cls}: mean={m:.1f}w, std={s:.1f}w")

    summary = {k: round(float(v), 1) for k, v in means.items()}
    if spread > 15.0:
        return (validation_warn("survival_bias",
                                f"mean time_from_diagnosis spread={spread:.1f}w > 15w"),
                summary)
    return (validation_result("survival_bias", True,
                              f"spread={spread:.1f}w — no strong evidence"),
            summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_section("LUMIERE Preprocessing Validation — Step 1")

    if not PARQUET_PATH.exists():
        print(f"ERROR: {PARQUET_PATH} not found. Run build_dataset.py first.")
        return

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Patients: {df['Patient'].nunique()}")

    results: dict[str, str] = {}

    print_section("Running assertions")
    results["1_n_effective"]        = check_n_effective(df)
    results["2_no_last_timepoint"]  = check_no_last_timepoint(df)
    results["3_no_nan_inf"]         = check_no_nan_inf(df)
    results["4_label_distribution"] = check_label_distribution(df)
    results["5_delta_baseline"]     = check_delta_baseline(df)
    results["6_log_transform"]      = check_log_transform(df)
    results["7_no_future_info"]     = check_no_future_info(df)
    results["8_patient_039_absent"] = check_patient_039_absent(df)
    results["9_column_names"]       = check_column_names(df)
    results["10_no_duplicate_pairs"]= check_no_duplicate_pairs(df)
    results["11_week_monotonic"]    = check_week_monotonic(df)

    print_section("Survival bias check (informational)")
    results["W1_survival_bias"], survival_summary = check_survival_bias(df)

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
            "survival_bias_summary": survival_summary,
        },
    )

    if failed > 0:
        print("\n  ❌ VALIDATION FAILED — fix build_dataset.py before Step 2")
    else:
        print("\n  ✅ VALIDATION PASSED — ready for Step 2")
    print(SECTION)

    save_validation_report(report, REPORT_PATH)


if __name__ == "__main__":
    main()