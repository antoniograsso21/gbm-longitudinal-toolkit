"""
LUMIERE Preprocessing — Step 1
================================
Transforms raw LUMIERE CSVs into dataset_paired.parquet:
one row per (patient, timepoint) paired example ready for ML.

Pipeline sub-steps (execution order):
    1. Pivot radiomic CSV: long → wide (1 row per patient/timepoint)
    2. Merge with RANO labels (inner join on Patient + Timepoint)
    3. Label shift: target = RANO(t+1), drop last timepoint per patient
    4. Drop scans with segmentation failures + log-transform high-skew features
       (must precede temporal features — dropped rows would bias scan_index)
    5. Add temporal features: interval_weeks, time_from_diagnosis_weeks, scan_index
    6. Compute delta features: Δf = (f_t - f_{t-1}) / interval_weeks

Normalization is NOT performed here.
It lives inside the cross-validator in Step 3 (StandardScaler fit on train fold only).

Usage:
    python -m src.preprocessing.build_dataset

Output:
    data/processed/dataset_paired.parquet
    data/processed/preprocessing_report.json
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.lumiere_io import (
    SECTION,
    print_section,
    CSV_DEEPBRATUMIA,
    CSV_RANO,
    LABEL_ENCODING,
    LABEL_PREFIX,
    LOG_TRANSFORM_EXCLUDE,
    PATIENTS_EXCLUDED,
    RADIOMIC_PREFIX,
    RANO_EXCLUDE,
    RANO_MAPPING,
    SKEW_THRESHOLD,
    feature_suffix,
    load_and_clean_rano,
    load_csv,
    parse_week,
    radiomic_cols,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/raw/lumiere")
OUTPUT_DIR = Path("data/processed")

# Columns to drop from radiomic CSVs before processing
COLS_TO_DROP: list[str] = ["Reader", "Image", "Mask", "Label"]

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class PivotStats:
    n_rows_raw: int
    n_patients: int
    n_timepoints: int
    n_feature_columns: int
    labels_found: list[str]
    sequences_found: list[str]


@dataclass
class MergeStats:
    n_radiomic_timepoints: int
    n_rano_timepoints: int
    n_merged: int
    n_patients_merged: int
    n_radiomic_only: int    # timepoints with features but no RANO label
    n_rano_only: int        # timepoints with RANO but no features


@dataclass
class MissingValueStats:
    n_scans_before: int
    n_scans_dropped: int
    n_patients_affected: int
    n_patients_lost_all: int
    segmentation_failures: dict[str, int]  # label prefix -> count of all-NaN rows


@dataclass
class LabelShiftStats:
    n_before: int
    n_after: int
    n_patients: int
    class_distribution: dict[str, int]
    n_short_interval_pairs: int   # pairs with delta_t < 2w — flagged, not dropped


@dataclass
class DeltaStats:
    n_feature_columns: int
    n_delta_columns: int
    n_baseline_scans: int


@dataclass
class PreprocessingReport:
    source: str
    patients_excluded: list[str]
    pivot: PivotStats
    merge: MergeStats
    missing: MissingValueStats
    label_shift: LabelShiftStats
    n_high_skew_features: int
    n_log_excluded_features: int
    delta: DeltaStats
    output_shape: tuple[int, int]
    output_path: str


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sub-step 1 — Pivot radiomic CSV
# ---------------------------------------------------------------------------
def pivot_radiomic(csv_name: str) -> tuple[pd.DataFrame, PivotStats]:
    """
    Pivot the long-format radiomic CSV to 1 row per (Patient, Timepoint).

    Input structure:  1 row per (Patient x Timepoint x Sequence x Label)
    Output structure: 1 row per (Patient x Timepoint)
    Column naming:    {LABEL_PREFIX}_{Sequence}_{feature_name}
                      e.g. CE_CT1_original_shape_Elongation

    Missing combinations (e.g. Necrosis absent for one scan) produce NaN
    in the corresponding columns. These are handled in sub-step 5.
    """
    print_section(f"SUB-STEP 1 — PIVOT: {csv_name}")

    raw = load_csv(csv_name, DATA_DIR)
    print(f"Raw shape: {raw.shape}")

    raw = raw.rename(columns={"Time point": "Timepoint"})

    cols_to_drop = [c for c in COLS_TO_DROP if c in raw.columns]
    cols_to_drop += [c for c in raw.columns if c.startswith("diagnostics_")]
    raw = raw.drop(columns=cols_to_drop)
    raw = raw[~raw["Patient"].isin(PATIENTS_EXCLUDED)].copy()

    labels_found = sorted(raw["Label name"].dropna().unique())
    seqs_found = sorted(raw["Sequence"].dropna().unique())
    unmapped = [l for l in labels_found if l not in LABEL_PREFIX]
    if unmapped:
        raise ValueError(f"Unknown label names (add to LABEL_PREFIX): {unmapped}")

    print(f"Labels: {labels_found}")
    print(f"Sequences: {seqs_found}")

    raw_rc = [c for c in raw.columns if c.startswith(RADIOMIC_PREFIX)]
    print(f"Radiomic feature columns (pre-pivot): {len(raw_rc)}")

    raw["col_prefix"] = (
        raw["Label name"].map(LABEL_PREFIX)
        + "_"
        + raw["Sequence"].fillna("UNK")
    )

    # Efficient pivot: melt -> direct pivot (no intermediate groupby needed
    # because duplicates are absent — verified in audit)
    id_cols = ["Patient", "Timepoint", "col_prefix"]
    long = raw[id_cols + raw_rc].melt(
        id_vars=id_cols,
        value_vars=raw_rc,
        var_name="feature",
        value_name="value",
    )
    long["col_name"] = long["col_prefix"] + "_" + long["feature"]
    long = long.drop(columns=["col_prefix", "feature"])

    pivoted = long.pivot(
        index=["Patient", "Timepoint"],
        columns="col_name",
        values="value",
    ).reset_index()
    pivoted.columns.name = None

    n_patients = pivoted["Patient"].nunique()
    n_timepoints = len(pivoted)
    n_feat_cols = len(radiomic_cols(pivoted))

    expected = len(labels_found) * len(seqs_found) * len(raw_rc)
    if n_feat_cols != expected:
        print(f"  ⚠️  Feature col count: expected {expected}, got {n_feat_cols} "
              f"(difference = missing label/seq combinations -> NaN, handled in step 5)")

    print(f"Pivoted shape: {pivoted.shape}")
    print(f"Patients: {n_patients} | Timepoints: {n_timepoints} | Feature columns: {n_feat_cols}")

    stats = PivotStats(
        n_rows_raw=len(raw),
        n_patients=n_patients,
        n_timepoints=n_timepoints,
        n_feature_columns=n_feat_cols,
        labels_found=labels_found,
        sequences_found=seqs_found,
    )
    return pivoted, stats


# ---------------------------------------------------------------------------
# Sub-step 2 — Merge with RANO labels
# ---------------------------------------------------------------------------
def merge_rano(pivoted: pd.DataFrame) -> tuple[pd.DataFrame, MergeStats]:
    """
    Inner join pivoted radiomic features with valid RANO labels.
    Only timepoints with BOTH features AND a valid RANO label are retained.
    """
    print_section("SUB-STEP 2 — MERGE WITH RANO LABELS")

    rano = load_and_clean_rano(DATA_DIR)

    n_radiomic = len(pivoted)
    n_rano = len(rano)

    merged = pivoted.merge(rano, on=["Patient", "Timepoint"], how="inner")

    n_radiomic_only = n_radiomic - len(merged)
    n_rano_only = n_rano - len(merged)

    print(f"Radiomic timepoints:     {n_radiomic}")
    print(f"RANO timepoints:         {n_rano}")
    print(f"Merged (inner join):     {len(merged)}")
    print(f"Radiomic-only (no RANO): {n_radiomic_only}  — expected: not all scans have RANO")
    print(f"RANO-only (no features): {n_rano_only}  — orphan RANO entries (see audit)")
    print(f"Patients after merge:    {merged['Patient'].nunique()}")

    stats = MergeStats(
        n_radiomic_timepoints=n_radiomic,
        n_rano_timepoints=n_rano,
        n_merged=len(merged),
        n_patients_merged=int(merged["Patient"].nunique()),
        n_radiomic_only=n_radiomic_only,
        n_rano_only=n_rano_only,
    )
    return merged, stats


# ---------------------------------------------------------------------------
# Sub-step 3 — Label shift
# ---------------------------------------------------------------------------
def apply_label_shift(merged: pd.DataFrame) -> tuple[pd.DataFrame, LabelShiftStats]:
    """
    Assign target = RANO label of the NEXT timepoint per patient.
    The last timepoint of every patient is dropped (no future label available).

    This is the most critical transformation: the model must predict future
    state, not current state (which would be clinically useless).
    """
    print_section("SUB-STEP 3 — LABEL SHIFT")

    merged = merged.copy()
    merged["week_num"] = merged["Timepoint"].apply(parse_week)
    merged = merged.sort_values(["Patient", "week_num"]).reset_index(drop=True)

    n_before = len(merged)
    merged["target"] = merged.groupby("Patient")["Rating_grouped"].shift(-1)
    merged["interval_weeks"] = merged.groupby("Patient")["week_num"].diff().shift(-1)

    paired = merged[merged["target"].notna()].copy()
    paired["target_encoded"] = paired["target"].map(LABEL_ENCODING)

    n_after = len(paired)
    n_patients = int(paired["Patient"].nunique())
    class_dist = paired["target"].value_counts().to_dict()

    short = paired[paired["interval_weeks"] < 2]
    if len(short) > 0:
        print(f"\n⚠️  Pairs with delta_t < 2w ({len(short)}) — flagged, not dropped:")
        for _, row in short.iterrows():
            print(f"   {row['Patient']}  {row['Timepoint']} "
                  f"-> dt={row['interval_weeks']:.1f}w  target={row['target']}")

    print(f"\nBefore label shift: {n_before} timepoints")
    print(f"After label shift:  {n_after} pairs  ({n_before - n_after} last-timepoints dropped)")
    print(f"Patients: {n_patients}")
    print(f"Class distribution (target):\n{pd.Series(class_dist).to_string()}")

    # Integrity check: no patient's last timepoint in output
    last_week = merged.groupby("Patient")["week_num"].max()
    for pat, lw in last_week.items():
        if ((paired["Patient"] == pat) & (paired["week_num"] == lw)).any():
            raise ValueError(
                f"[CRITICAL] Last timepoint of {pat} found in paired dataset — "
                f"label shift failed!"
            )
    print("\n✅ Verified: no patient's last timepoint in output")

    stats = LabelShiftStats(
        n_before=n_before,
        n_after=n_after,
        n_patients=n_patients,
        class_distribution={k: int(v) for k, v in class_dist.items()},
        n_short_interval_pairs=len(short),
    )
    return paired, stats


# ---------------------------------------------------------------------------
# Sub-step 4 — Add temporal features
# ---------------------------------------------------------------------------
def add_temporal_features(paired: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal context features.

    - time_from_diagnosis_weeks: week_num of scan T (proxy for disease stage)
    - scan_index: 0-based ordinal position in the patient's scan sequence
    - interval_weeks: already computed in label shift step

    All three are both model features and leakage monitoring variables.
    Their importance in the final model MUST be reported in the paper.
    """
    print_section("SUB-STEP 4 — TEMPORAL FEATURES")

    paired = paired.copy()
    paired["time_from_diagnosis_weeks"] = paired["week_num"]
    paired["scan_index"] = paired.groupby("Patient").cumcount()

    print(f"Added: time_from_diagnosis_weeks, scan_index")
    print(f"interval_weeks range: {paired['interval_weeks'].min():.1f}w"
          f" — {paired['interval_weeks'].max():.1f}w")
    print(f"scan_index range: {paired['scan_index'].min()} — {paired['scan_index'].max()}")

    return paired


# ---------------------------------------------------------------------------
# Sub-step 5 — Handle missing values + log-transform
# ---------------------------------------------------------------------------
def handle_missing_and_transform(
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, MissingValueStats, int]:
    """
    Part A — Drop scans with segmentation failures:
        A scan where ALL features for a segmentation label are NaN indicates
        a DeepBraTumIA failure for that label. Imputing an entire missing
        segmentation introduces fabricated signal. Strategy: drop the scan.

    Part B — Log-transform high-skew features:
        Apply log1p to features with |skewness| > SKEW_THRESHOLD.
        Features in LOG_TRANSFORM_EXCLUDE are skipped — they can legitimately
        be negative (CT Hounsfield intensities, bounded correlation features).
        Applied before delta computation so rates are on the log scale.

    Both parts are in one function because the drop must happen before
    the skewness calculation (dropped rows would bias the skew estimate).
    """
    print_section("SUB-STEP 5 — MISSING VALUES + LOG-TRANSFORM")

    paired = paired.copy()
    rc = radiomic_cols(paired)

    # --- Part A: segmentation failure detection ---
    label_prefixes = sorted({col.split("_")[0] for col in rc})
    print(f"Segmentation labels in dataset: {label_prefixes}")

    failed_mask = pd.Series(False, index=paired.index)
    failure_counts: dict[str, int] = {}

    for prefix in label_prefixes:
        prefix_cols = [c for c in rc if c.startswith(f"{prefix}_")]
        if not prefix_cols:
            continue
        # Drop a scan if ANY feature for this label is NaN.
        # Catches both complete segmentation failures (all 107 NaN) and partial
        # PyRadiomics failures on near-absent regions (e.g. 40/107 NaN for
        # near-zero Necrosis). Imputing within a label block would fabricate signal.
        has_any_nan = paired[prefix_cols].isna().any(axis=1)
        n_failed = int(has_any_nan.sum())
        failure_counts[prefix] = n_failed
        if n_failed > 0:
            print(f"  {prefix}: {n_failed} scans with missing features (any NaN)")
        failed_mask |= has_any_nan

    n_before = len(paired)
    patients_before = set(paired["Patient"].unique())

    paired = paired[~failed_mask].reset_index(drop=True)

    n_dropped = n_before - len(paired)
    patients_after = set(paired["Patient"].unique())
    n_patients_lost_all = len(patients_before - patients_after)

    print(f"\nScans dropped (segmentation failure): {n_dropped} / {n_before}")
    if n_patients_lost_all > 0:
        lost = sorted(patients_before - patients_after)
        print(f"⚠️  Patients who lost ALL examples: {lost}")
    else:
        print("✅ No patient lost all examples")

    # --- Part B: log-transform ---
    rc = radiomic_cols(paired)  # recompute after drop
    skewness = paired[rc].skew().abs()
    high_skew_all = skewness[skewness > SKEW_THRESHOLD].index.tolist()

    high_skew = [
        c for c in high_skew_all
        if feature_suffix(c) not in LOG_TRANSFORM_EXCLUDE
    ]
    n_excluded = len(high_skew_all) - len(high_skew)

    print(f"\nHigh-skew features (|skew| > {SKEW_THRESHOLD}): {len(high_skew_all)}")
    print(f"  Excluded (negative-domain): {n_excluded}")
    print(f"  Applying log1p to:          {len(high_skew)}")

    paired[high_skew] = np.log1p(paired[high_skew])

    inf_count = np.isinf(paired[rc].values).sum()
    if inf_count:
        raise ValueError(f"[CRITICAL] {inf_count} inf values after log-transform")
    print("✅ No inf values introduced by log-transform")

    missing_stats = MissingValueStats(
        n_scans_before=n_before,
        n_scans_dropped=n_dropped,
        n_patients_affected=len(patients_before) - len(patients_after),
        n_patients_lost_all=n_patients_lost_all,
        segmentation_failures={k: int(v) for k, v in failure_counts.items()},
    )
    return paired, missing_stats, len(high_skew)


# ---------------------------------------------------------------------------
# Sub-step 6 — Compute delta features
# ---------------------------------------------------------------------------
def compute_delta_features(paired: pd.DataFrame) -> tuple[pd.DataFrame, DeltaStats]:
    """
    Compute rate-of-change features per patient:
        delta_f_t = (f_t - f_{t-1}) / interval_weeks

    For the first scan of each patient (no t-1): delta_f = 0, is_baseline_scan = True.
    All delta columns are built via pd.concat to avoid DataFrame fragmentation.
    """
    print_section("SUB-STEP 6 — DELTA FEATURES")

    # Consolidate memory layout before the wide horizontal concat
    paired = paired.copy().sort_values(["Patient", "week_num"]).reset_index(drop=True)

    rc = radiomic_cols(paired)
    paired["is_baseline_scan"] = ~paired.duplicated(subset="Patient", keep="first")

    delta_series = []
    for col in rc:
        diff = paired.groupby("Patient")[col].diff()
        rate = (diff / paired["interval_weeks"]).where(~paired["is_baseline_scan"], other=0.0)
        rate.name = f"delta_{col}"
        delta_series.append(rate)

    paired = pd.concat([paired, pd.concat(delta_series, axis=1)], axis=1)

    delta_cols = [f"delta_{c}" for c in rc]
    n_baseline = int(paired["is_baseline_scan"].sum())

    print(f"Delta features computed: {len(delta_cols)}")
    print(f"Baseline scans (df=0):   {n_baseline}")

    nonzero_on_baseline = (paired.loc[paired["is_baseline_scan"], delta_cols] != 0).any().any()
    if nonzero_on_baseline:
        raise ValueError("[CRITICAL] Non-zero delta features found on baseline scans!")
    print("✅ Verified: all delta features = 0 on baseline scans")

    inf_count = np.isinf(paired[delta_cols].values).sum()
    if inf_count:
        raise ValueError(
            f"[CRITICAL] {inf_count} inf values in delta features — "
            f"likely interval_weeks == 0 for some pairs"
        )
    print("✅ No inf values in delta features")

    stats = DeltaStats(
        n_feature_columns=len(rc),
        n_delta_columns=len(delta_cols),
        n_baseline_scans=n_baseline,
    )
    return paired, stats


# ---------------------------------------------------------------------------
# Final cleanup
# ---------------------------------------------------------------------------
def _finalize(paired: pd.DataFrame) -> pd.DataFrame:
    """
    Drop intermediate columns and produce the final sorted DataFrame.
    Keeps: Patient, Timepoint, all radiomic cols, all delta cols,
           temporal features, target, target_encoded, is_baseline_scan.
    Drops: week_num (redundant with time_from_diagnosis_weeks), Rating_grouped.
    """
    drop_cols = ["week_num", "Rating_grouped"]
    paired = paired.drop(columns=[c for c in drop_cols if c in paired.columns])
    paired = paired.sort_values(
        ["Patient", "time_from_diagnosis_weeks"]
    ).reset_index(drop=True)
    # Recalculate scan_index after any row drops so it is always 0-based and contiguous
    paired["scan_index"] = paired.groupby("Patient").cumcount()
    return paired


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("GBM Longitudinal Toolkit — Preprocessing (Step 0.2)")
    print(SECTION)
    print(f"Source:            {CSV_DEEPBRATUMIA}")
    print(f"Excluded patients: {sorted(PATIENTS_EXCLUDED)}")

    pivoted, pivot_stats = pivot_radiomic(CSV_DEEPBRATUMIA)
    merged, merge_stats = merge_rano(pivoted)
    paired, label_shift_stats = apply_label_shift(merged)
    paired, missing_stats, n_high_skew = handle_missing_and_transform(paired)
    paired = add_temporal_features(paired)
    paired, delta_stats = compute_delta_features(paired)
    paired = _finalize(paired)

    output_path = OUTPUT_DIR / "dataset_paired.parquet"
    paired.to_parquet(output_path, index=False)

    report = PreprocessingReport(
        source=CSV_DEEPBRATUMIA,
        patients_excluded=sorted(PATIENTS_EXCLUDED),
        pivot=pivot_stats,
        merge=merge_stats,
        missing=missing_stats,
        label_shift=label_shift_stats,
        n_high_skew_features=n_high_skew,
        n_log_excluded_features=len(LOG_TRANSFORM_EXCLUDE),
        delta=delta_stats,
        output_shape=paired.shape,
        output_path=str(output_path),
    )

    report_path = OUTPUT_DIR / "preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    print_section("PREPROCESSING COMPLETE")
    print(f"  Output shape:  {paired.shape[0]} rows x {paired.shape[1]} columns")
    print(f"  Patients:      {paired['Patient'].nunique()}")
    print(f"  Target dist:   {paired['target'].value_counts().to_dict()}")
    print(f"  Saved  -> {output_path}")
    print(f"  Report -> {report_path}")
    print(SECTION)


if __name__ == "__main__":
    main()