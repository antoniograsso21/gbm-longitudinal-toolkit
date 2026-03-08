"""
LUMIERE Dataset Audit — Phase 0
================================
Explores the four LUMIERE CSVs following the EDA Guidelines in CONTEXT.md.

FUNDAMENTAL RULE: the unit of analysis is always the PATIENT, not the scan.
Every statistic is computed first per patient, then aggregated.

Usage:
    python -m src.audit.lumiere_audit
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/raw/lumiere")
OUTPUT_DIR = Path("data/processed")

RANO_EXCLUDE: frozenset[str] = frozenset({"Pre-Op", "Post-Op", "Post-Op ", "Post-Op/PD"})

RANO_MAPPING: dict[str, str] = {
    "CR": "Response",
    "PR": "Response",
    "SD": "Stable",
    "PD": "Progressive",
}

# String values treated as NaN when loading CSVs
NA_VALUES: list[str] = ["na", "n/a", "NA", "N/A", "nan", "NaN", ""]

SECTION = "=" * 60


# ---------------------------------------------------------------------------
# Result dataclasses — typed, serialisable, testable
# ---------------------------------------------------------------------------
@dataclass
class RanoStats:
    total_timepoints: int
    valid_timepoints: int
    n_patients: int
    timepoints_per_patient: dict[str, float]
    class_distribution_per_scan: dict[str, int]
    class_distribution_per_patient: dict[str, int]
    dominant_patients: dict[str, int]
    n_duplicate_timepoints: int  # (Patient, Date) duplicates in raw RANO file


@dataclass
class TemporalStats:
    delta_weeks_summary: dict[str, float]  # describe() output
    mean_delta_by_class: dict[str, float]
    n_zero_delta: int


@dataclass
class RadiomicStats:
    shape: tuple[int, int]
    n_missing_features: int
    n_high_skew_features: int
    top_skew: dict[str, float]


@dataclass
class AuditResult:
    n_effective: int
    n_patients: int
    class_distribution: dict[str, int]
    rano: RanoStats
    temporal: TemporalStats
    radiomic: RadiomicStats


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    print(f"\n{SECTION}\n{title}\n{SECTION}")


def _load_csv(filename: str) -> pd.DataFrame:
    """Load a LUMIERE CSV treating common NA strings as NaN."""
    return pd.read_csv(DATA_DIR / filename, na_values=NA_VALUES, keep_default_na=True)


def parse_week(date_str: str) -> float:
    """
    Parse LUMIERE week strings into a float ordinal.

    Examples:
        'week-044'   → 44.0
        'week-000-1' → 0.1   (first pre-op scan)
        'week-000-2' → 0.2   (second pre-op scan)

    Raises:
        ValueError: if the string does not match the expected format.
    """
    m = re.match(r"week-(\d+)(?:-(\d+))?$", date_str)
    if not m:
        raise ValueError(f"Unexpected date format: '{date_str}'")
    week = float(m.group(1))
    if m.group(2):
        week += float(m.group(2)) * 0.1
    return week


def _add_week_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a 'week_num' column parsed from 'Date'."""
    df = df.copy()
    df["week_num"] = df["Date"].apply(parse_week)
    return df.sort_values(["Patient", "week_num"])


def _compute_consecutive_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each patient build consecutive (t, t+1) pairs.
    Returns a DataFrame with columns:
        patient, week_t, week_t1, delta_weeks, rating_t, rating_t1, label_t1
    Raises:
        ValueError: if any delta_weeks < 0 (ordering inconsistency).
    """
    records = []
    for patient, group in df.groupby("Patient"):
        rows = group.reset_index(drop=True)
        for i in range(len(rows) - 1):
            delta = rows.loc[i + 1, "week_num"] - rows.loc[i, "week_num"]
            if delta < 0:
                raise ValueError(
                    f"Negative Δt ({delta}) for patient {patient} "
                    f"between week {rows.loc[i, 'week_num']} "
                    f"and {rows.loc[i + 1, 'week_num']}."
                )
            records.append({
                "patient": patient,
                "week_t": rows.loc[i, "week_num"],
                "week_t1": rows.loc[i + 1, "week_num"],
                "delta_weeks": delta,
                "rating_t": rows.loc[i, "Rating_grouped"],
                "label_t1": rows.loc[i + 1, "Rating_grouped"],
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 1 — Raw file overview
# ---------------------------------------------------------------------------
def audit_raw_files() -> None:
    """Print shape, columns, head, and missing values for every CSV."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV found in {DATA_DIR}. Copy files before running.")
        return

    print(f"📂 Files found: {[f.name for f in csv_files]}")

    for filepath in csv_files:
        _section(f"📄 {filepath.name}")
        df = _load_csv(filepath.name)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumns:\n{df.columns.tolist()}")
        print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")

        null_pct = (df.isnull().sum() / len(df) * 100).round(1)
        missing = null_pct[null_pct > 0]
        if missing.empty:
            print("\n✅ No missing values")
        else:
            print(f"\n⚠️  Missing values (% per column):\n{missing.to_string()}")


# ---------------------------------------------------------------------------
# Step 2 — RANO labels audit (per-patient)
# ---------------------------------------------------------------------------
def audit_rano() -> tuple[pd.DataFrame, RanoStats]:
    """
    Analyse RANO labels following the per-patient EDA rule.

    Returns:
        rano_valid: filtered and mapped DataFrame
        stats: RanoStats dataclass
    """
    _section("📊 RANO AUDIT — per patient")

    rano = _load_csv("LUMIERE-ExpertRating-v202211.csv")
    rano.columns = ["Patient", "Date", "LessThan3M", "NonMeasurable", "Rating", "Rationale"]

    print(f"Total timepoints in file: {len(rano)}")
    print(f"Raw rating distribution:\n{rano['Rating'].value_counts().to_string()}")

    rano_valid = rano[~rano["Rating"].isin(RANO_EXCLUDE)].copy()
    rano_valid["Rating_grouped"] = rano_valid["Rating"].map(RANO_MAPPING)

    print(f"\n✅ Valid timepoints after Pre/Post-Op exclusion: {len(rano_valid)}")
    print(f"✅ Patients with ≥1 valid timepoint: {rano_valid['Patient'].nunique()}")

    tp_per_patient = rano_valid.groupby("Patient")["Date"].count()
    describe = tp_per_patient.describe().round(1)
    print(f"\n📊 Valid timepoints per patient:\n{describe.to_string()}")

    for threshold in [2, 3, 4, 5]:
        print(f"  Patients with ≥{threshold} timepoints: {(tp_per_patient >= threshold).sum()}")

    per_scan = rano_valid["Rating_grouped"].value_counts().to_dict()
    print(f"\n📊 Class distribution (per scan — raw):\n{pd.Series(per_scan).to_string()}")

    per_patient = {
        cls: int(rano_valid[rano_valid["Rating_grouped"] == cls]["Patient"].nunique())
        for cls in ["Progressive", "Stable", "Response"]
    }
    print(f"\n📊 Class distribution (per patient — ≥1 occurrence):")
    for cls, n in per_patient.items():
        print(f"  {cls}: {n} patients")

    top5 = tp_per_patient.sort_values(ascending=False).head(5)
    print(f"\n⚠️  Top 5 patients by scan count (dominance risk):\n{top5.to_string()}")

    # Duplicate (Patient, Date) check — catches raw data anomalies like Patient-042 week-010
    duplicates = rano_valid[rano_valid.duplicated(subset=["Patient", "Date"], keep=False)]
    if not duplicates.empty:
        print(f"\n⚠️  Duplicate (Patient, Date) entries found — must be resolved before preprocessing:")
        print(duplicates[["Patient", "Date", "Rating_grouped"]].to_string())
    else:
        print("\n✅ No duplicate (Patient, Date) entries")

    stats = RanoStats(
        total_timepoints=len(rano),
        valid_timepoints=len(rano_valid),
        n_patients=int(rano_valid["Patient"].nunique()),
        timepoints_per_patient=describe.to_dict(),
        class_distribution_per_scan={k: int(v) for k, v in per_scan.items()},
        class_distribution_per_patient=per_patient,
        dominant_patients=top5.to_dict(),
        n_duplicate_timepoints=int(rano_valid.duplicated(subset=["Patient", "Date"]).sum()),
    )
    return rano_valid, stats


# ---------------------------------------------------------------------------
# Step 3 — Temporal intervals (clinical workflow leakage check)
# ---------------------------------------------------------------------------
def audit_temporal_intervals(rano_valid: pd.DataFrame) -> TemporalStats:
    """
    Analyse Δt between consecutive scans per patient.
    Key check: if Progressive has a much lower mean Δt → leakage risk.
    """
    _section("⏱️  TEMPORAL INTERVALS — clinical workflow leakage check")

    df = _add_week_column(rano_valid)
    pairs = _compute_consecutive_pairs(df)

    summary = pairs["delta_weeks"].describe().round(1)
    print(f"\n📊 Δt distribution (weeks):\n{summary.to_string()}")

    mean_by_class = pairs.groupby("label_t1")["delta_weeks"].mean().round(1)
    print(f"\n📊 Mean Δt by RANO class at T+1 (leakage signal):\n{mean_by_class.to_string()}")
    print(
        "\n⚠️  If Progressive has a significantly lower Δt → "
        "clinical workflow leakage confirmed. Must be declared in the paper."
    )

    n_zero = int((pairs["delta_weeks"] == 0).sum())
    if n_zero > 0:
        print(f"\n⚠️  {n_zero} pairs with Δt=0 (same-week scans). Investigate before preprocessing.")

    return TemporalStats(
        delta_weeks_summary=summary.to_dict(),
        mean_delta_by_class=mean_by_class.to_dict(),
        n_zero_delta=n_zero,
    )


# ---------------------------------------------------------------------------
# Step 4 — Radiomic features audit
# ---------------------------------------------------------------------------
def audit_radiomic_features() -> RadiomicStats:
    """
    Analyse the PyRadiomics CSV: missing values and skewness.
    Skewness is computed only on 'original_*' columns (true radiomic features).
    """
    _section("🔬 RADIOMIC FEATURES AUDIT")

    feat = _load_csv("LUMIERE-pyradiomics-hdglioauto-features.csv")
    print(f"Shape: {feat.shape}")

    # Missing values (all columns)
    null_pct = (feat.isnull().sum() / len(feat) * 100).round(1)
    missing = null_pct[null_pct > 0]
    print(f"\n⚠️  Columns with missing values: {len(missing)}")
    if not missing.empty:
        print(missing.sort_values(ascending=False).head(20).to_string())

    # Skewness — only true radiomic features
    radiomic_cols = [c for c in feat.columns if c.startswith("original_")]
    skewness = feat[radiomic_cols].skew().abs()
    high_skew = skewness[skewness > 2].sort_values(ascending=False)
    print(f"\n⚠️  Radiomic features with skewness > 2 (log-transform candidates): {len(high_skew)}")
    if not high_skew.empty:
        print(high_skew.head(10).to_string())

    # Structure insight
    n_sequences = feat["Sequence"].nunique() if "Sequence" in feat.columns else "?"
    n_regions = feat["Label name"].nunique() if "Label name" in feat.columns else "?"
    print(f"\n📊 Sequences: {n_sequences} | Regions: {n_regions}")
    print(f"   → Each scan generates {n_sequences}×{n_regions} rows in this CSV")
    print(f"   → Effective scans ≈ {len(feat) // (n_sequences * n_regions)}")

    return RadiomicStats(
        shape=feat.shape,
        n_missing_features=int(len(missing)),
        n_high_skew_features=int(len(high_skew)),
        top_skew=high_skew.head(10).round(2).to_dict(),
    )


# ---------------------------------------------------------------------------
# Step 5 — Effective sample size (paired examples after label shift)
# ---------------------------------------------------------------------------
def compute_n_effective(rano_valid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Compute n_effective: paired (features_t, label_t+1) examples after label shift.
    This is the true sample size of the project.

    Returns:
        paired_df: DataFrame with all paired examples
        stats: dict with n_effective, n_patients, class distribution
    """
    _section("🎯 N_EFFECTIVE — paired examples after label shift")

    df = _add_week_column(rano_valid)
    paired_df = _compute_consecutive_pairs(df)

    n_effective = len(paired_df)
    n_patients = paired_df["patient"].nunique()
    class_dist = paired_df["label_t1"].value_counts().to_dict()

    print(f"\n✅ n_effective (total paired examples): {n_effective}")
    print(f"✅ Patients represented: {n_patients}")
    print(f"\n📊 Class distribution (label_t+1):\n{pd.Series(class_dist).to_string()}")

    return paired_df, {"n_effective": n_effective, "n_patients": n_patients, "class_distribution": class_dist}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("🧠 GBM Longitudinal Toolkit — LUMIERE Audit")
    print(SECTION)

    audit_raw_files()
    rano_valid, rano_stats = audit_rano()
    temporal_stats = audit_temporal_intervals(rano_valid)
    radiomic_stats = audit_radiomic_features()
    paired_df, paired_stats = compute_n_effective(rano_valid)

    result = AuditResult(
        n_effective=paired_stats["n_effective"],
        n_patients=paired_stats["n_patients"],
        class_distribution=paired_stats["class_distribution"],
        rano=rano_stats,
        temporal=temporal_stats,
        radiomic=radiomic_stats,
    )

    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\n{SECTION}")
    print("✅ AUDIT COMPLETE")
    print(f"   n_effective = {result.n_effective} paired examples")
    print(f"   Patients    = {result.n_patients}")
    print(f"   Saved stats → {stats_path}")
    print(SECTION)


if __name__ == "__main__":
    main()