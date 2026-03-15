"""
src/utils/lumiere_io.py
=======================
Shared pure utilities for the LUMIERE dataset.

All LUMIERE-specific constants and low-level I/O functions live here.
Scripts that import from this module: lumiere_audit.py, dataset_builder.py,
dataset_validator.py, features_validator.py, graphs_validator.py,
graph_builder.py.

LUMIERE-specific design decisions documented here:
- parse_week: handles week-NNN and week-NNN-M sub-week format
- PATIENTS_EXCLUDED: audit-validated exclusion list with rationale
- LOG_TRANSFORM_EXCLUDE: features excluded from log1p (Hounsfield + bounded domains)
- load_and_clean_rano: single source of truth for RANO cleanup chain

Generalisation note (Phase 5):
    These functions are intentionally LUMIERE-specific in V1.
    Abstraction to a DatasetConfig layer is deferred to Phase 5,
    once the full pipeline is complete and interface requirements are known.
"""

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (override in tests or CLI via module-level assignment)
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path("data/raw/lumiere")

# ---------------------------------------------------------------------------
# RANO label constants
# ---------------------------------------------------------------------------
RANO_EXCLUDE: frozenset[str] = frozenset({
    "Pre-Op",
    "Post-Op",
    "Post-Op ",   # trailing-space variant in source data
    "Post-Op/PD",
})

RANO_MAPPING: dict[str, str] = {
    "CR": "Response",
    "PR": "Response",
    "SD": "Stable",
    "PD": "Progressive",
}

LABEL_ENCODING: dict[str, int] = {
    "Progressive": 0,
    "Stable": 1,
    "Response": 2,
}

# ---------------------------------------------------------------------------
# Patient exclusions (audit-validated)
# ---------------------------------------------------------------------------
# Patient-025: all RANO dates completely misaligned with imaging dates.
# Root cause: temporal reference frame error in source data.
# Paper Methods: "One patient (Patient-025) was excluded due to irreconcilable
# temporal reference frame inconsistency between RANO assessment dates and
# imaging acquisition dates."
PATIENTS_EXCLUDED: frozenset[str] = frozenset({"Patient-025"})

# ---------------------------------------------------------------------------
# Feature transform constants
# ---------------------------------------------------------------------------
# Segmentation label → short column prefix
LABEL_PREFIX: dict[str, str] = {
    "Contrast-enhancing": "CE",
    "Edema": "ED",
    "Necrosis": "NC",
    "Non-enhancing": "NE",   # HD-GLIO-AUTO — used for ablation A6
    "Contrast-enhancing ": "CE",  # trailing-space variant — defensive
}

# Features excluded from log1p transformation.
# These can legitimately take negative values — applying log1p after a global
# shift would change the feature's mathematical meaning.
# Determined empirically from the DeepBraTumIA CSV (audit Phase 0).
# Matched against the feature suffix after stripping the label_sequence_ prefix.
LOG_TRANSFORM_EXCLUDE: frozenset[str] = frozenset({
    "original_firstorder_10Percentile",   # CT Hounsfield — negative for air/background
    "original_firstorder_90Percentile",
    "original_firstorder_Mean",
    "original_firstorder_Median",
    "original_firstorder_Minimum",        # always <= 0 for CT background
    "original_firstorder_Skewness",       # defined on (-inf, +inf)
    "original_glcm_ClusterShade",         # signed by definition
    "original_glcm_Correlation",          # bounded [-1, 1]
    "original_glcm_Imc1",                 # bounded [-1, 0]
    "original_glcm_DifferenceEntropy",    # reaches -epsilon via float arithmetic
    "original_glcm_JointEntropy",
    "original_glcm_SumEntropy",
})

SKEW_THRESHOLD: float = 2.0
RADIOMIC_PREFIX: str = "original_"

# String values treated as NaN in all LUMIERE CSVs
NA_VALUES: list[str] = ["na", "n/a", "NA", "N/A", "N-A", "nan", "NaN", ""]

# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------
CSV_DEEPBRATUMIA: str = "LUMIERE-pyradiomics-deepbratumia-features.csv"
CSV_HDGLIO: str = "LUMIERE-pyradiomics-hdglioauto-features.csv"
CSV_RANO: str = "LUMIERE-ExpertRating-v202211.csv"
CSV_DEMOGRAPHICS: str = "LUMIERE-Demographics_Pathology.csv"
CSV_COMPLETENESS: str = "LUMIERE-datacompleteness.csv"

# ---------------------------------------------------------------------------
# Core pure functions
# ---------------------------------------------------------------------------

def parse_week(date_str: str) -> float:
    """
    Parse a LUMIERE week string into a float ordinal.

    Handles two formats:
        'week-044'   → 44.0   (standard)
        'week-000-1' → 0.1    (first sub-week scan, e.g. pre-op)
        'week-061-1' → 61.1   (sub-week scan at week 61)

    The sub-week suffix is multiplied by 0.1, preserving chronological ordering
    while making all week values unique floats.

    Raises:
        ValueError: if the string does not match the expected LUMIERE format.
    """
    m = re.match(r"week-(\d+)(?:-(\d+))?$", date_str)
    if not m:
        raise ValueError(
            f"Unexpected LUMIERE date format: '{date_str}'. "
            f"Expected 'week-NNN' or 'week-NNN-M'."
        )
    week = float(m.group(1))
    if m.group(2):
        week += float(m.group(2)) * 0.1
    return week


def load_csv(filename: str, data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load a LUMIERE CSV with consistent NA handling.

    All LUMIERE CSVs use a mix of empty strings, "na", "NA", "N-A" for
    missing values. This function normalises all of them to NaN.

    Args:
        filename: CSV filename (relative to data_dir).
        data_dir: override for DATA_DIR module constant (useful in tests).
    """
    path = (data_dir or DATA_DIR) / filename
    return pd.read_csv(
        path,
        na_values=NA_VALUES,
        keep_default_na=True,
        low_memory=False,
    )


def load_and_clean_rano(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load the RANO label file and apply the full LUMIERE cleanup chain.

    Cleanup chain (in order):
        1. Exclude patients in PATIENTS_EXCLUDED (audit-validated)
        2. Exclude Pre/Post-Op entries (RANO_EXCLUDE)
        3. Map raw ratings to grouped labels (RANO_MAPPING)
        4. Drop rows with unmapped ratings (e.g. Rating='None' — data entry artefact)
        5. Resolve duplicate (Patient, Date) entries: keep last occurrence
           (Patient-042 week-010: conflicting SD+PD, PD is authoritative)

    Returns:
        DataFrame with columns: [Patient, Timepoint, Rating_grouped]
        One row per valid (Patient, Timepoint) pair.

    Raises:
        FileNotFoundError: if CSV_RANO is not found in data_dir.
    """
    raw = load_csv(CSV_RANO, data_dir)
    raw.columns = ["Patient", "Date", "LessThan3M", "NonMeasurable", "Rating", "Rationale"]

    df = raw[~raw["Patient"].isin(PATIENTS_EXCLUDED)].copy()
    df = df[~df["Rating"].isin(RANO_EXCLUDE)].copy()
    df["Rating_grouped"] = df["Rating"].map(RANO_MAPPING)
    df = df[df["Rating_grouped"].notna()].copy()

    n_before = len(df)
    df = df.drop_duplicates(subset=["Patient", "Date"], keep="last").copy()
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"  [lumiere_io] Resolved {n_dupes} duplicate (Patient, Date) — kept last rating")

    return df.rename(columns={"Date": "Timepoint"})[["Patient", "Timepoint", "Rating_grouped"]]


def radiomic_cols(df: pd.DataFrame) -> list[str]:
    """
    Return all radiomic feature column names from a DataFrame.

    Works both pre-pivot (columns start with 'original_') and
    post-pivot (columns contain 'original_' after a label_seq_ prefix).
    """
    return [c for c in df.columns if RADIOMIC_PREFIX in c]


def feature_suffix(col: str) -> str:
    """
    Strip the label+sequence prefix from a post-pivot column name.

    Examples:
        'CE_CT1_original_shape_Elongation' → 'original_shape_Elongation'
        'NC_FLAIR_original_glcm_Correlation' → 'original_glcm_Correlation'
        'original_shape_Elongation' → 'original_shape_Elongation'  (pre-pivot, no-op)
    """
    parts = col.split("_", 2)
    return parts[2] if len(parts) == 3 and parts[2].startswith(RADIOMIC_PREFIX) else col

# ---------------------------------------------------------------------------
# Console utilities (shared across audit, preprocessing, validate)
# ---------------------------------------------------------------------------
SECTION: str = "=" * 60


def print_section(title: str) -> None:
    """Print a section header — consistent across all scripts."""
    print(f"\n{SECTION}\n{title}\n{SECTION}")


# ---------------------------------------------------------------------------
# Validation utilities (shared across dataset_validator, features_validator,
# validate_graphs — DRY: never duplicate these helpers in per-step files)
# ---------------------------------------------------------------------------
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ValidationReport:
    """
    Structured result for any pipeline validation script.

    Fields:
        passed:   number of assertions that returned PASS
        failed:   number of assertions that returned FAIL
        warnings: number of assertions that returned WARN
        results:  dict mapping assertion name -> status string
        metadata: step-specific summary values (shape, class dist, etc.)
    """
    passed: int
    failed: int
    warnings: int
    results: dict[str, str]
    metadata: dict[str, Any]


def validation_result(label: str, ok: bool, msg: str = "") -> str:
    """
    Print and return a PASS / FAIL result line.

    Args:
        label: short assertion name shown in output.
        ok:    True -> PASS, False -> FAIL.
        msg:   failure detail appended to "FAIL: " (ignored when ok=True).

    Returns:
        "PASS" or "FAIL: <msg>"
    """
    status = "PASS" if ok else f"FAIL: {msg}"
    icon = "✅" if ok else "❌"
    print(f"  {icon}  {label}: {status}")
    return status


def validation_warn(label: str, msg: str) -> str:
    """
    Print and return a WARN result line.

    Args:
        label: short assertion name shown in output.
        msg:   warning detail.

    Returns:
        "WARN: <msg>"
    """
    print(f"  ⚠️   {label}: WARN: {msg}")
    return f"WARN: {msg}"


def save_validation_report(report: "ValidationReport", path: "Path") -> None:
    """
    Serialise a ValidationReport to JSON and exit with code 1 if any FAIL.

    Calling sys.exit(1) here ensures DVC and CI pipelines detect failures
    without each validate_*.py having to re-implement the exit logic.

    Args:
        report: populated ValidationReport dataclass.
        path:   output JSON path (parent directory must exist).
    """
    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\n  Saved -> {path}")
    if report.failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Week string utilities
# ---------------------------------------------------------------------------

def float_week_to_str(week_num: float) -> str:
    """
    Convert a float week ordinal back to the LUMIERE string format.

    Examples:
        44.0 -> 'week-044'
        0.1  -> 'week-000-1'
    """
    base = int(week_num)
    suffix = round((week_num - base) * 10)
    if suffix > 0:
        return f"week-{base:03d}-{suffix}"
    return f"week-{base:03d}"


def add_week_column(
    df: "pd.DataFrame",
    date_col: str = "Date",
    patient_col: str = "Patient",
) -> "pd.DataFrame":
    """Return a copy of df with a 'week_num' float column parsed from date_col."""
    import pandas as _pd  # local import — lumiere_io has no top-level pandas dep
    df = df.copy()
    df["week_num"] = df[date_col].apply(parse_week)
    return df.sort_values([patient_col, "week_num"])


def compute_consecutive_pairs(
    df: "pd.DataFrame",
    patient_col: str = "Patient",
) -> "pd.DataFrame":
    """
    For each patient build consecutive (t, t+1) pairs from a week-sorted DataFrame.

    Returns a DataFrame with columns:
        patient, week_t, week_t1, delta_weeks, rating_t, label_t1

    Raises:
        ValueError: if any delta_weeks < 0 (ordering inconsistency).
    """
    import pandas as _pd
    records = []
    for patient, group in df.groupby(patient_col):
        rows = group.reset_index(drop=True)
        for i in range(len(rows) - 1):
            delta = rows.iloc[i + 1]["week_num"] - rows.iloc[i]["week_num"]
            if delta < 0:
                raise ValueError(
                    f"Negative delta_t ({delta:.1f}) for {patient} "
                    f"at weeks {rows.iloc[i]['week_num']} -> "
                    f"{rows.iloc[i+1]['week_num']}."
                )
            records.append({
                "patient": patient,
                "week_t": rows.iloc[i]["week_num"],
                "week_t1": rows.iloc[i + 1]["week_num"],
                "delta_weeks": delta,
                "rating_t": rows.iloc[i]["Rating_grouped"],
                "label_t1": rows.iloc[i + 1]["Rating_grouped"],
            })
    return _pd.DataFrame(records)

def build_full_feature_set(df: "pd.DataFrame") -> list[str]:
    """
    Return all ML feature columns from dataset_engineered.parquet (Full set D).

    Excludes identifiers, target columns, and boolean flags that are
    incompatible with continuous MI estimators (Kraskov k-NN).

    Excluded columns:
        Patient, Timepoint          — identifiers
        target, target_encoded      — target
        is_baseline_scan            — boolean flag
        is_nadir_scan               — boolean flag (incompatible with Kraskov MI)

    Used by all run_*.py entry points to build the consistent feature pool
    passed to mRMR + Stability Selection. Centralised here (DRY) to prevent
    divergence across LR, LightGBM, and LSTM entry points.

    Args:
        df: dataset_engineered DataFrame loaded from parquet.

    Returns:
        List of column names for Full set D (2579 columns for LUMIERE v202211).
    """
    exclude = {
        "Patient", "Timepoint",
        "target", "target_encoded",
        "is_baseline_scan",
        "is_nadir_scan",   # boolean flag — incompatible with Kraskov k-NN MI estimator
    }
    # Note: CE_vs_nadir and weeks_since_nadir ARE included — they are continuous
    # features but are excluded from the mRMR radiomic pool inside feature_selector.py
    # (NADIR_COLS constant). They are included in full_feature_set for LightGBM/LSTM/GNN.
    return [c for c in df.columns if c not in exclude]