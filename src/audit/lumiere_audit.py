"""
LUMIERE Dataset Audit — Phase 0
================================
Explores the LUMIERE CSVs following the EDA Guidelines in CONTEXT.md.

FUNDAMENTAL RULE: the unit of analysis is always the PATIENT, not the scan.
Every statistic is computed first per patient, then aggregated.

Usage:
    python -m src.audit.lumiere_audit
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.utils.lumiere_io import (
    CSV_COMPLETENESS,
    CSV_DEEPBRATUMIA,
    CSV_DEMOGRAPHICS,
    CSV_HDGLIO,
    CSV_RANO,
    PATIENTS_EXCLUDED,
    RANO_EXCLUDE,
    RANO_MAPPING,
    load_csv,
    parse_week,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/raw/lumiere")
OUTPUT_DIR = Path("data/processed")

# Columns to drop in preprocessing — confirmed 100% NaN in audit
# Reader: populated in some PyRadiomics versions but empty in LUMIERE (all "N-A")
COLS_TO_DROP_RADIOMIC: list[str] = ["Reader"]

# Threshold for DeepBraTumIA viability as primary segmentation source.
# Derived from: patients with >=3 valid RANO timepoints in the HD-GLIO-AUTO
# baseline audit = 55. Below this number the 3-node graph loses too many
# patients to be credible vs the 2-node baseline.
DEEPBRATUMIA_VIABILITY_THRESHOLD: int = 55

# Column name constants — different CSVs use different names for the same concept
PATIENT_COL = "Patient"
TIMEPOINT_COL_RADIOMIC = "Time point"     # radiomic CSVs
TIMEPOINT_COL_COMPLETENESS = "Timepoint"  # datacompleteness CSV
LABEL_COL = "Label name"
SEQUENCE_COL = "Sequence"

SECTION = "=" * 60


# ---------------------------------------------------------------------------
# Result dataclasses — typed, serialisable, testable
# ---------------------------------------------------------------------------
@dataclass
class RanoStats:
    total_timepoints: int
    valid_timepoints: int           # after Pre/Post-Op exclusion and deduplication
    n_patients: int
    timepoints_per_patient: dict[str, float]
    class_distribution_per_scan: dict[str, int]
    class_distribution_per_patient: dict[str, int]
    dominant_patients: dict[str, int]
    n_duplicate_timepoints: int     # (Patient, Date) conflicts resolved by keeping last
    n_rano_timepoints_unmatched: int  # RANO dates absent from datacompleteness (format mismatch)


@dataclass
class TemporalStats:
    delta_weeks_summary: dict[str, float]
    mean_delta_by_class: dict[str, float]
    n_zero_delta: int


@dataclass
class RadiomicStats:
    """Coverage and quality statistics for one radiomic CSV."""
    source: str
    shape: tuple[int, int]
    n_labels: int
    labels_found: list[str]
    # Coverage chain: completeness -> CSV -> usable
    n_scans_in_completeness: int    # ground truth from datacompleteness.csv
    n_scans_in_csv: int             # unique (Patient, Timepoint) actually in CSV
    n_scans_missing_from_csv: int   # in completeness but absent from CSV (extraction failed)
    n_scans_with_all_nan: int       # in CSV but all original_* features are NaN
    n_scans_with_partial_nan: int   # usable but at least one sequence has NaN features
    n_scans_fully_usable: int       # all required labels present, no all-NaN rows
    n_patients_fully_usable: int    # patients with >=1 fully usable scan
    cols_all_nan: list[str]         # columns that are 100% NaN — drop in preprocessing
    n_high_skew_features: int
    top_skew: dict[str, float]


@dataclass
class PairedStats:
    """n_effective broken down by radiomic data source."""
    source: str
    n_effective_rano_only: int              # RANO consecutive pairs — upper bound
    n_pairs_dropped_missing_features: int   # pairs where t or t+1 had no usable scan
    n_effective: int                        # true ML sample size
    n_patients: int
    class_distribution: dict[str, int]


@dataclass
class AuditResult:
    rano: RanoStats
    temporal: TemporalStats
    radiomic_hdglio: RadiomicStats
    radiomic_deepbratumia: RadiomicStats
    paired_hdglio: PairedStats
    paired_deepbratumia: PairedStats


# ---------------------------------------------------------------------------
# Private utilities — pure functions, no side effects
# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    print(f"\n{SECTION}\n{title}\n{SECTION}")


def _float_week_to_str(week_num: float) -> str:
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


def _add_week_column(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Return a copy of df with a 'week_num' float column parsed from date_col."""
    df = df.copy()
    df["week_num"] = df[date_col].apply(parse_week)
    return df.sort_values([PATIENT_COL, "week_num"])


def _compute_consecutive_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each patient build consecutive (t, t+1) pairs from a week-sorted DataFrame.

    Returns a DataFrame with columns:
        patient, week_t, week_t1, delta_weeks, rating_t, label_t1

    Raises:
        ValueError: if any delta_weeks < 0 (ordering inconsistency).
    """
    records = []
    for patient, group in df.groupby(PATIENT_COL):
        rows = group.reset_index(drop=True)
        for i in range(len(rows) - 1):
            delta = rows.iloc[i + 1]["week_num"] - rows.iloc[i]["week_num"]
            if delta < 0:
                raise ValueError(
                    f"Negative delta_t ({delta:.1f}) for {patient} "
                    f"at weeks {rows.iloc[i]['week_num']} -> {rows.iloc[i+1]['week_num']}."
                )
            records.append({
                "patient": patient,
                "week_t": rows.iloc[i]["week_num"],
                "week_t1": rows.iloc[i + 1]["week_num"],
                "delta_weeks": delta,
                "rating_t": rows.iloc[i]["Rating_grouped"],
                "label_t1": rows.iloc[i + 1]["Rating_grouped"],
            })
    return pd.DataFrame(records)


def _analyse_scan_completeness(
    feat: pd.DataFrame,
    required_labels: list[str],
) -> tuple[set[tuple[str, str]], int, int]:
    """
    Classify every (Patient, Timepoint) scan in the radiomic CSV.

    A scan is FULLY USABLE if every required label has at least one row where
    at least one original_* feature is non-NaN (i.e. extraction succeeded for
    at least one sequence).

    A scan is PARTIALLY NaN if it is fully usable but at least one sequence
    row has all-NaN original_* features (some sequences failed, others did not).

    Returns:
        complete_scans: set of (Patient, Timepoint) that are fully usable
        n_all_nan_scans: scans where ALL rows for the scan are all-NaN
        n_partial_nan_scans: usable scans with at least one all-NaN sequence row
    """
    radiomic_cols = [c for c in feat.columns if c.startswith("original_")]

    feat = feat.copy()
    feat["_row_usable"] = feat[radiomic_cols].notna().any(axis=1)

    # Per (Patient, Timepoint, Label): is at least one sequence row usable?
    label_usability = (
        feat.groupby([PATIENT_COL, TIMEPOINT_COL_RADIOMIC, LABEL_COL])["_row_usable"]
        .any()
        .reset_index()
        .rename(columns={"_row_usable": "label_usable"})
    )

    # Per (Patient, Timepoint): which labels are usable?
    usable_labels_per_scan = (
        label_usability[label_usability["label_usable"]]
        .groupby([PATIENT_COL, TIMEPOINT_COL_RADIOMIC])[LABEL_COL]
        .apply(set)
        .reset_index()
    )
    required_set = set(required_labels)
    complete_mask = usable_labels_per_scan[LABEL_COL].apply(
        lambda s: required_set.issubset(s)
    )
    complete_df = usable_labels_per_scan[complete_mask]
    complete_scans = set(zip(complete_df[PATIENT_COL], complete_df[TIMEPOINT_COL_RADIOMIC]))

    # Scans where every single row has all-NaN features
    row_usability_per_scan = (
        feat.groupby([PATIENT_COL, TIMEPOINT_COL_RADIOMIC])["_row_usable"]
        .any()
        .reset_index()
        .rename(columns={"_row_usable": "scan_has_any_usable_row"})
    )
    n_all_nan_scans = int((~row_usability_per_scan["scan_has_any_usable_row"]).sum())

    # Usable scans that still have at least one all-NaN sequence row
    has_any_nan_row = (
        feat.groupby([PATIENT_COL, TIMEPOINT_COL_RADIOMIC])["_row_usable"]
        .apply(lambda s: (~s).any())
        .reset_index()
        .rename(columns={"_row_usable": "has_nan_row"})
    )
    complete_with_nan = has_any_nan_row[
        has_any_nan_row.apply(
            lambda r: (r[PATIENT_COL], r[TIMEPOINT_COL_RADIOMIC]) in complete_scans
            and r["has_nan_row"],
            axis=1,
        )
    ]
    n_partial_nan_scans = len(complete_with_nan)

    return complete_scans, n_all_nan_scans, n_partial_nan_scans


def _compute_paired_with_radiomics(
    rano_valid: pd.DataFrame,
    complete_scans: set[tuple[str, str]],
    source_name: str,
) -> PairedStats:
    """
    Compute n_effective requiring complete radiomic features at BOTH t and t+1.

    A paired example (t, t+1) is usable only if:
    1. t and t+1 are consecutive RANO-labelled timepoints
    2. Scan at t  has complete features (graph construction at t)
    3. Scan at t+1 has complete features (delta feature computation needs t+1)
    """
    df = _add_week_column(rano_valid)
    all_pairs = _compute_consecutive_pairs(df)
    n_rano_only = len(all_pairs)

    # Orphan check: RANO timepoints whose string key does not appear in complete_scans.
    # Known cause: 'week-000' (plain) in RANO vs 'week-000-1' in radiomic/completeness CSVs
    # for Patient-020, Patient-068, Patient-071. These are genuine source data inconsistencies.
    radiomic_timepoints = {tp for _, tp in complete_scans}
    rano_timepoints = set(rano_valid["Date"].unique())
    orphan_tp = rano_timepoints - radiomic_timepoints
    if orphan_tp:
        orphan_rows = rano_valid[rano_valid["Date"].isin(orphan_tp)][[PATIENT_COL, "Date", "Rating_grouped"]]
        print(
            f"\n  Orphan RANO timepoints (no matching key in {source_name} CSV): {len(orphan_tp)}"
        )
        print(f"  These are excluded from n_effective — investigate before preprocessing:")
        print(f"  {orphan_rows.to_string(index=False)}")

    def _both_have_features(row: pd.Series) -> bool:
        t_key = (row["patient"], _float_week_to_str(row["week_t"]))
        t1_key = (row["patient"], _float_week_to_str(row["week_t1"]))
        return t_key in complete_scans and t1_key in complete_scans

    usable = all_pairs[all_pairs.apply(_both_have_features, axis=1)]
    class_dist = usable["label_t1"].value_counts().to_dict()

    return PairedStats(
        source=source_name,
        n_effective_rano_only=n_rano_only,
        n_pairs_dropped_missing_features=n_rano_only - len(usable),
        n_effective=len(usable),
        n_patients=int(usable["patient"].nunique()),
        class_distribution={k: int(v) for k, v in class_dist.items()},
    )


# ---------------------------------------------------------------------------
# Step 1 — Raw file overview (compact for large radiomic CSVs)
# ---------------------------------------------------------------------------
def audit_raw_files() -> None:
    """
    Print shape, columns, and missing values for every CSV.
    Full row preview is skipped for radiomic CSVs (152 columns — illegible).
    """
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV found in {DATA_DIR}. Copy raw files before running.")
        return

    print(f"Files found: {[f.name for f in csv_files]}")
    large_csvs = {CSV_HDGLIO, CSV_DEEPBRATUMIA}

    for filepath in csv_files:
        _section(f"  {filepath.name}")
        df = load_csv(filepath.name, DATA_DIR)
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"\nColumns:\n{df.columns.tolist()}")

        if filepath.name not in large_csvs:
            print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")

        null_pct = (df.isnull().sum() / len(df) * 100).round(1)
        missing = null_pct[null_pct > 0]
        if missing.empty:
            print("\nNo missing values")
        else:
            print(f"\nMissing values (% per column):\n{missing.to_string()}")


# ---------------------------------------------------------------------------
# Step 2 — RANO labels audit (per-patient)
# ---------------------------------------------------------------------------
def audit_rano() -> tuple[pd.DataFrame, RanoStats]:
    """
    Analyse RANO labels following the per-patient EDA rule.

    (Patient, Date) duplicates are resolved by keeping the last occurrence,
    consistent with the documented decision for Patient-042 week-010 (PD kept,
    SD dropped — see paper Methods).

    Returns:
        rano_valid: filtered, mapped, and deduplicated DataFrame
        stats: RanoStats dataclass
    """
    _section("RANO AUDIT — per patient")

    rano = load_csv(CSV_RANO, DATA_DIR)
    rano.columns = ["Patient", "Date", "LessThan3M", "NonMeasurable", "Rating", "Rationale"]

    print(f"Total timepoints in file: {len(rano)}")
    print(f"Raw rating distribution:\n{rano['Rating'].value_counts().to_string()}")

    rano_valid = rano[~rano["Rating"].isin(RANO_EXCLUDE)].copy()
    rano_valid["Rating_grouped"] = rano_valid["Rating"].map(RANO_MAPPING)

    # Drop rows with unrecognised Rating (e.g. literal string "None") — data entry artefacts.
    unmapped_mask = rano_valid["Rating_grouped"].isna()
    if unmapped_mask.any():
        unmapped = rano_valid[unmapped_mask]
        print(
            f"\n[INFO] {unmapped_mask.sum()} row(s) with unrecognised Rating value "
            f"(not in RANO_MAPPING, not in RANO_EXCLUDE) — dropped:"
        )
        print(unmapped[[PATIENT_COL, "Date", "Rating"]].to_string(index=False))
        rano_valid = rano_valid[~unmapped_mask].copy()

    print(f"\nValid timepoints after Pre/Post-Op exclusion: {len(rano_valid)}")
    print(f"Patients with >=1 valid timepoint: {rano_valid[PATIENT_COL].nunique()}")

    tp_per_patient = rano_valid.groupby(PATIENT_COL)["Date"].count()
    describe = tp_per_patient.describe().round(1)
    print(f"\nValid timepoints per patient:\n{describe.to_string()}")
    for threshold in [2, 3, 4, 5]:
        print(f"  Patients with >={threshold} timepoints: {(tp_per_patient >= threshold).sum()}")

    per_scan = rano_valid["Rating_grouped"].value_counts().to_dict()
    print(f"\nClass distribution (per scan):\n{pd.Series(per_scan).to_string()}")

    per_patient = {
        cls: int(rano_valid[rano_valid["Rating_grouped"] == cls][PATIENT_COL].nunique())
        for cls in ["Progressive", "Stable", "Response"]
    }
    print("\nClass distribution (per patient — >=1 occurrence):")
    for cls, n in per_patient.items():
        print(f"  {cls}: {n} patients")

    top5 = tp_per_patient.sort_values(ascending=False).head(5)
    print(f"\nTop 5 patients by scan count (dominance risk):\n{top5.to_string()}")

    duplicates = rano_valid[rano_valid.duplicated(subset=[PATIENT_COL, "Date"], keep=False)]
    n_dupes = int(len(duplicates) // 2)
    if not duplicates.empty:
        print("\nDuplicate (Patient, Date) entries — keeping last occurrence:")
        print(duplicates[[PATIENT_COL, "Date", "Rating_grouped"]].to_string())
        rano_valid = rano_valid.drop_duplicates(
            subset=[PATIENT_COL, "Date"], keep="last"
        ).copy()
        print(f"  Rows after deduplication: {len(rano_valid)}")
    else:
        print("\nNo duplicate (Patient, Date) entries")

    # Cross-file consistency check: RANO timepoints vs datacompleteness.
    # Three categories of mismatch, each with different severity and action:
    #   1. FULLY DISJOINT patient: ALL RANO dates missing from completeness.
    #      -> Likely temporal reference frame error. Exclude patient entirely.
    #   2. PARTIAL mismatch: some RANO dates missing, some matching.
    #      -> Likely transcription error on specific dates. Investigate manually.
    #   3. Format mismatch: 'week-000' vs 'week-000-1'.
    #      -> Resolvable in preprocessing by mapping where unambiguous.
    completeness = load_csv(CSV_COMPLETENESS, DATA_DIR)
    dc_by_patient: dict[str, set[str]] = (
        completeness.groupby(PATIENT_COL)[TIMEPOINT_COL_COMPLETENESS]
        .apply(set)
        .to_dict()
    )
    rano_by_patient: dict[str, set[str]] = (
        rano_valid.groupby(PATIENT_COL)["Date"]
        .apply(set)
        .to_dict()
    )

    fully_disjoint: list[str] = []
    partial_mismatch: dict[str, set[str]] = {}
    n_rano_unmatched = 0

    for patient, r_dates in rano_by_patient.items():
        d_dates = dc_by_patient.get(patient, set())
        extra = r_dates - d_dates
        if not extra:
            continue
        n_rano_unmatched += len(extra)
        if d_dates and r_dates.isdisjoint(d_dates):
            fully_disjoint.append(patient)
        elif extra:
            partial_mismatch[patient] = extra

    if fully_disjoint:
        print(
            f"\n[CRITICAL] Patients with ALL RANO timepoints absent from completeness "
            f"({len(fully_disjoint)} patients — temporal reference frame error):"
        )
        for p in sorted(fully_disjoint):
            print(f"  {p}: RANO={sorted(rano_by_patient[p])}")
            print(f"       completeness={sorted(dc_by_patient.get(p, set()))}")
        print("  -> Action: EXCLUDE these patients entirely. Document in paper Methods.")

    if partial_mismatch:
        print(
            f"\n[WARNING] Patients with some RANO timepoints absent from completeness "
            f"({len(partial_mismatch)} patients — possible transcription error):"
        )
        for p, extra in sorted(partial_mismatch.items()):
            print(f"  {p}: unmatched RANO dates={sorted(extra)}")
        print("  -> Action: investigate manually before preprocessing.")

    if not fully_disjoint and not partial_mismatch:
        print("\nAll RANO timepoints have a matching entry in datacompleteness.")

    stats = RanoStats(
        total_timepoints=len(rano),
        valid_timepoints=len(rano_valid),
        n_patients=int(rano_valid[PATIENT_COL].nunique()),
        timepoints_per_patient=describe.to_dict(),
        class_distribution_per_scan={k: int(v) for k, v in per_scan.items()},
        class_distribution_per_patient=per_patient,
        dominant_patients=top5.to_dict(),
        n_duplicate_timepoints=n_dupes,
        n_rano_timepoints_unmatched=n_rano_unmatched,
    )
    return rano_valid, stats


# ---------------------------------------------------------------------------
# Step 3 — Temporal intervals (clinical workflow leakage check)
# ---------------------------------------------------------------------------
def audit_temporal_intervals(rano_valid: pd.DataFrame) -> TemporalStats:
    """
    Analyse delta_t between consecutive scans per patient.
    Key check: Progressive with significantly lower mean delta_t signals leakage.
    """
    _section("TEMPORAL INTERVALS — clinical workflow leakage check")

    df = _add_week_column(rano_valid)
    pairs = _compute_consecutive_pairs(df)

    summary = pairs["delta_weeks"].describe().round(1)
    print(f"\ndelta_t distribution (weeks):\n{summary.to_string()}")

    mean_by_class = pairs.groupby("label_t1")["delta_weeks"].mean().round(1)
    print(f"\nMean delta_t by RANO class at T+1:\n{mean_by_class.to_string()}")
    print(
        "\nIf Progressive has a significantly lower delta_t -> "
        "clinical workflow leakage confirmed. Declare in paper."
    )

    n_zero = int((pairs["delta_weeks"] == 0).sum())
    if n_zero > 0:
        print(f"\n{n_zero} pairs with delta_t=0 (same-week scans). Investigate before preprocessing.")

    return TemporalStats(
        delta_weeks_summary=summary.to_dict(),
        mean_delta_by_class=mean_by_class.to_dict(),
        n_zero_delta=n_zero,
    )


# ---------------------------------------------------------------------------
# Step 4 — Radiomic features audit (parametric — used for both sources)
# ---------------------------------------------------------------------------
def audit_radiomic_features(
    csv_name: str,
    source_name: str,
    required_labels: list[str],
) -> tuple[RadiomicStats, set[tuple[str, str]]]:
    """
    Analyse a PyRadiomics CSV and return coverage statistics plus the set of
    fully usable (Patient, Timepoint) scan keys.

    Coverage chain audited:
    1. datacompleteness.csv -> ground truth of all attempted scans
    2. radiomic CSV -> scans where extraction produced any rows
    3. all-NaN rows -> present in CSV but extraction silently failed
    4. partial-NaN scans -> usable but some sequences missing (logged for Methods)
    5. fully usable scans -> all required labels present in at least one sequence

    Args:
        csv_name: filename of the radiomic CSV
        source_name: label used in output and dataclass ("HD-GLIO-AUTO" / "DeepBraTumIA")
        required_labels: labels that must ALL be usable for a scan to be complete
    """
    _section(f"RADIOMIC FEATURES AUDIT — {source_name}")

    feat = load_csv(csv_name, DATA_DIR)
    completeness = load_csv(CSV_COMPLETENESS, DATA_DIR)
    print(f"Shape: {feat.shape}")

    # 1. Coverage vs datacompleteness
    dc_scans = set(zip(completeness[PATIENT_COL], completeness[TIMEPOINT_COL_COMPLETENESS]))
    feat_scans = set(zip(feat[PATIENT_COL], feat[TIMEPOINT_COL_RADIOMIC]))
    n_in_completeness = len(dc_scans)
    n_in_csv = len(feat_scans)
    n_missing_from_csv = len(dc_scans - feat_scans)

    print(f"\nScans in datacompleteness (ground truth): {n_in_completeness}")
    print(f"Scans in {source_name} CSV:               {n_in_csv}")
    if n_missing_from_csv > 0:
        print(
            f"Scans absent from CSV:                    {n_missing_from_csv} "
            f"(label not found or ROI too small — declare in paper Methods)"
        )
    else:
        print(f"All completeness scans present in {source_name} CSV")

    # 2-4. NaN analysis and usable scan classification
    complete_scans, n_all_nan, n_partial_nan = _analyse_scan_completeness(
        feat, required_labels
    )
    n_fully_usable = len(complete_scans)
    n_patients_usable = len({p for p, _ in complete_scans})

    print(f"\nScans with ALL rows all-NaN (silent failure):    {n_all_nan}")
    print(
        f"Usable scans with partial NaN sequences:         {n_partial_nan} "
        f"(included in model — missing sequences handled in preprocessing)"
    )
    print(f"Scans fully usable ({len(required_labels)} labels, >=1 good sequence): {n_fully_usable}")
    print(f"Patients with >=1 fully usable scan:             {n_patients_usable}")

    # Labels and structure
    labels_found = sorted(feat[LABEL_COL].dropna().unique().tolist())
    n_sequences = feat[SEQUENCE_COL].nunique() if SEQUENCE_COL in feat.columns else "?"
    print(f"\nLabels found:  {labels_found}")
    print(f"Sequences: {n_sequences} | Labels: {len(labels_found)}")
    print(f"  -> Each scan generates {n_sequences}x{len(labels_found)} rows")

    # Columns that are 100% NaN — must be dropped in preprocessing
    all_nan_cols = feat.columns[feat.isnull().all()].tolist()
    if all_nan_cols:
        print(f"\nColumns 100% NaN (drop in preprocessing): {all_nan_cols}")
        print(f"  Note: 'Reader' is populated by some PyRadiomics versions but")
        print(f"  always empty in LUMIERE (stored as 'N-A' in the raw CSV).")
    else:
        print("\nNo columns are 100% NaN")

    # Skewness on original_* features only
    radiomic_cols = [c for c in feat.columns if c.startswith("original_")]
    skewness = feat[radiomic_cols].skew().abs()
    high_skew = skewness[skewness > 2].sort_values(ascending=False)
    print(f"\nFeatures with |skewness| > 2 (log-transform candidates): {len(high_skew)}")
    if not high_skew.empty:
        print(high_skew.head(10).to_string())

    # Decision guidance (DeepBraTumIA only)
    if source_name == "DeepBraTumIA":
        print(f"\n{'-' * 60}")
        verdict = (
            "VIABLE as primary source"
            if n_patients_usable >= DEEPBRATUMIA_VIABILITY_THRESHOLD
            else "REVERT TO HD-GLIO-AUTO"
        )
        print(f"DECISION — threshold: >={DEEPBRATUMIA_VIABILITY_THRESHOLD} patients")
        print(f"  Current: {n_patients_usable} patients -> {verdict}")
        print(f"{'-' * 60}")

    stats = RadiomicStats(
        source=source_name,
        shape=feat.shape,
        n_labels=len(labels_found),
        labels_found=labels_found,
        n_scans_in_completeness=n_in_completeness,
        n_scans_in_csv=n_in_csv,
        n_scans_missing_from_csv=n_missing_from_csv,
        n_scans_with_all_nan=n_all_nan,
        n_scans_with_partial_nan=n_partial_nan,
        n_scans_fully_usable=n_fully_usable,
        n_patients_fully_usable=n_patients_usable,
        cols_all_nan=all_nan_cols,
        n_high_skew_features=int(len(high_skew)),
        top_skew=high_skew.head(10).round(2).to_dict(),
    )
    return stats, complete_scans


# ---------------------------------------------------------------------------
# Step 5 — Effective sample size (RANO + radiomics join, per source)
# ---------------------------------------------------------------------------
def compute_n_effective(
    rano_valid: pd.DataFrame,
    complete_scans: set[tuple[str, str]],
    source_name: str,
) -> PairedStats:
    """
    Compute n_effective: paired examples where BOTH t and t+1 have usable scans.

    Requirements for a valid paired example:
    1. Consecutive RANO-labelled timepoints (t, t+1)
    2. Complete radiomic features at t   (graph construction)
    3. Complete radiomic features at t+1 (delta feature computation requires t+1)

    Args:
        rano_valid: filtered and deduplicated RANO DataFrame
        complete_scans: (Patient, Timepoint) keys with fully usable features
        source_name: "HD-GLIO-AUTO" or "DeepBraTumIA"
    """
    _section(f"N_EFFECTIVE — {source_name} (label shift + radiomics join)")

    stats = _compute_paired_with_radiomics(rano_valid, complete_scans, source_name)

    print(f"\nRANO-only pairs (upper bound):                     {stats.n_effective_rano_only}")
    print(f"Pairs dropped (t or t+1 missing usable scan):     {stats.n_pairs_dropped_missing_features}")
    print(f"n_effective (true ML sample size):                 {stats.n_effective}")
    print(f"Patients represented:                              {stats.n_patients}")
    print(f"\nClass distribution (label_t+1):\n{pd.Series(stats.class_distribution).to_string()}")

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("GBM Longitudinal Toolkit — LUMIERE Audit")
    print(SECTION)

    audit_raw_files()

    rano_valid, rano_stats = audit_rano()
    temporal_stats = audit_temporal_intervals(rano_valid)

    radiomic_hd, complete_hd = audit_radiomic_features(
        csv_name=CSV_HDGLIO,
        source_name="HD-GLIO-AUTO",
        required_labels=["Non-enhancing", "Contrast-enhancing"],
    )
    radiomic_db, complete_db = audit_radiomic_features(
        csv_name=CSV_DEEPBRATUMIA,
        source_name="DeepBraTumIA",
        required_labels=["Necrosis", "Contrast-enhancing", "Edema"],
    )

    paired_hd = compute_n_effective(rano_valid, complete_hd, "HD-GLIO-AUTO")
    paired_db = compute_n_effective(rano_valid, complete_db, "DeepBraTumIA")

    result = AuditResult(
        rano=rano_stats,
        temporal=temporal_stats,
        radiomic_hdglio=radiomic_hd,
        radiomic_deepbratumia=radiomic_db,
        paired_hdglio=paired_hd,
        paired_deepbratumia=paired_db,
    )

    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\n{SECTION}")
    print("AUDIT COMPLETE")
    print(
        f"  HD-GLIO-AUTO  n_effective = {result.paired_hdglio.n_effective} "
        f"({result.paired_hdglio.n_patients} patients)"
    )
    print(
        f"  DeepBraTumIA  n_effective = {result.paired_deepbratumia.n_effective} "
        f"({result.paired_deepbratumia.n_patients} patients)"
    )
    print(f"  Saved -> {stats_path}")
    print(SECTION)


if __name__ == "__main__":
    main()