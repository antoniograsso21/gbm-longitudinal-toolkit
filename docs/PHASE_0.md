# Phase 0 — Data Foundation

## Objective
Verify that the dataset is structurally sound and build the clean paired
dataset that every subsequent phase will consume.

**Output of this phase**: `data/processed/dataset_paired.parquet`
A single file containing one row per (patient, timepoint) paired example,
with all radiomic features, temporal features, and the target label.

---

## Where Phase 0 Sits in the Project

```
Raw CSVs (data/raw/lumiere/)
        │
        ▼
[Step 0.1] AUDIT          src/audit/lumiere_audit.py         ✅ DONE
        │   "Are the data structurally sound?"
        │   Output: data/processed/dataset_stats.json
        │
        ▼
[Step 0.2] PREPROCESSING  src/preprocessing/build_dataset.py ⏳ NEXT
        │   "How do I build the ML-ready dataset?"
        │   Output: data/processed/dataset_paired.parquet
        │
        ▼
[Step 0.3] VALIDATION     src/audit/validate_dataset.py      ⏳ TODO
            "Is the dataset I built correct?"
            Output: data/processed/validation_report.json
```

The audit asks whether the raw data is usable.
The preprocessing transforms it into a usable form.
The validation verifies the transformation was correct.
All three are mandatory before any model is trained.

---

## Step 0.1 — Audit

**Status**: complete.
**Script**: `src/audit/lumiere_audit.py`
**Run**: `python -m src.audit.lumiere_audit`

### What it checks
- Shape and missing values of all four CSVs
- RANO label distribution per patient (not per scan)
- Temporal intervals (Δt) per RANO class — clinical workflow leakage check
- Radiomic feature completeness and skewness
- n_effective: paired examples after label shift

### Key findings
- n_effective = 318 paired examples, 68 patients
- Class distribution: Progressive=229 (72%), Stable=45 (14%), Response=44 (14%)
- Δt leakage: low risk (Progressive=13.3w, Stable=13.1w — nearly identical)
- Δt=0: exactly 1 pair — likely week-000-1/week-000-2 surviving the RANO filter.
  Must be identified and resolved before preprocessing (scrap or flag).
- Missing values: 14.9% uniform across all 152 columns — entire scans missing
  (HD-GLIO-AUTO failure). diagnostics_* columns (144 affected) will be dropped
  in preprocessing anyway. Relevant for model: 107 radiomic features × 14.9%.
- High-skew radiomic features: 62 with |skewness| > 2 (corrected from earlier
  estimate of 68 — previous count included diagnostics_* columns erroneously).
  Worst offender: original_ngtdm_Contrast (skewness=54). Log-transform candidates
  to be applied in preprocessing before normalisation.
- Demographics CSV: string "na" not recognised as NaN — fixed in _load_csv()
  with explicit na_values list. Missing values now correctly detected:
  IDH (25.3%), MGMT quantitative (29.7%), MGMT qualitative (12.1%).

### Output
`data/processed/dataset_stats.json` — full typed audit report

---

## Step 0.2 — Preprocessing

**Status**: to be implemented.
**Script**: `src/preprocessing/build_dataset.py`
**Run**: `python -m src.preprocessing.build_dataset`

### Sub-steps in order

**Sub-step 1 — Pivot the radiomic CSV**

Input: `LUMIERE-pyradiomics-hdglioauto-features.csv` (4792 rows)
Structure: 1 row per (patient × timepoint × sequence × region)

```
Patient | Date | Sequence | Label name | feature_1 ... feature_107
```

Output: 1 row per (patient × timepoint)
Column naming convention: `{region}_{sequence}_{feature_name}`

```
Patient | Date | ET_CT1_feature_1 ... ET_CT1_feature_107
                 ET_T1_feature_1  ... ET_T1_feature_107
                 ET_T2_feature_1  ...
                 ET_FLAIR_feature_1 ...
                 NC_CT1_feature_1 ...
                 ... (2 regions × 4 sequences × 107 features = 856 columns max)
```

Implementation: `pd.pivot_table` or `groupby + unstack`.
Missing combinations (scan where a sequence is absent): NaN — handled in sub-step 5.

**Sub-step 2 — Merge with RANO labels**

Join the pivoted radiomic DataFrame with `LUMIERE-ExpertRating-v202211.csv`
on (Patient, Date). Before joining:
- Verify that the Date format is identical in both files (both use `week-NNN`)
- Exclude Pre-Op and Post-Op entries from the RANO file
- Apply RANO mapping: CR+PR → Response, SD → Stable, PD → Progressive

Result: one row per (patient, timepoint) with features + RANO label.

**Sub-step 3 — Label shift**

For each patient, sort by week_num and shift the label forward by one step:
- target = RANO label of the NEXT timepoint
- Drop the last timepoint of every patient (no future label available)

This is the most critical transformation in the project.
Verify explicitly: no patient's last timepoint should appear in the output.

**Sub-step 4 — Add temporal features**

For each example add three columns:
- `delta_t_weeks`: weeks between scan T and scan T+1
- `time_from_diagnosis_weeks`: week_num of scan T (proxy for disease timeline)
- `scan_index`: ordinal position of this scan for this patient (0-based)

These serve both as model features and as leakage monitoring variables.
Their importance in the final model must be reported in the paper.

**Sub-step 5 — Handle missing values**

Strategy: drop scans where any sequence is entirely missing.
Rationale: imputation on entire missing sequences introduces fabricated signal.

Document and report:
- Number of scans dropped
- Number of patients affected
- Whether any patient loses all their paired examples after dropping

If a patient loses all examples, flag it explicitly in the validation report.

**Sub-step 6 — Compute delta features**

For each radiomic feature f, for each patient, compute:
```
Δf_t = (f_t - f_{t-1}) / delta_weeks
```

For the first scan of each patient (no t-1 exists):
- Set Δf = 0
- Set flag column `is_baseline_scan = True`

The model can use this flag to ignore delta features for baseline scans.
These columns are named `delta_{region}_{sequence}_{feature_name}`.

Note: normalization is NOT performed here.
It lives inside the cross-validator (see Phase 2).

### Output
`data/processed/dataset_paired.parquet`

Parquet is preferred over CSV because:
- Preserves dtypes (no silent string conversion of floats)
- Faster to load in training loops
- Smaller file size with compression

---

## Step 0.3 — Dataset Validation

**Status**: to be implemented.
**Script**: `src/audit/validate_dataset.py`
**Run**: `python -m src.audit.validate_dataset`

### What it verifies

1. **n_effective**: assert len(dataset) == 318 (or document deviation)
2. **No last-timepoint leakage**: for every patient, the last timepoint in the
   raw RANO file must not appear as a training example in the paired dataset
3. **No patient split contamination**: every row of a given patient has the
   same patient ID — verify groupby(patient).count() is consistent
4. **Label distribution**: assert class counts match audit report
5. **Delta features**: assert Δf == 0 for all rows where is_baseline_scan == True
6. **No future information**: assert that for every row, the target label
   corresponds to the next timepoint's RANO, not the current one
7. **Missing value counts**: assert counts match what was documented in sub-step 5

### Output
`data/processed/validation_report.json`

---

## Testing

Unit tests live in `tests/test_preprocessing.py`.
They use synthetic data only — no real CSVs required for CI.

What to test:
- `parse_week`: known input/output pairs including edge cases
- `_compute_consecutive_pairs`: verify label shift on a 3-patient synthetic dataset
- Delta feature computation: verify Δf=0 on first scan, correct formula on others
- Pivot logic: verify column naming convention on a minimal synthetic CSV

---

## Definition of Done for Phase 0

**Audit (Step 0.1)** ✅
- [x] Audit script committed and passing
- [x] dataset_stats.json saved: n_effective=318, 68 patients, 72%/14%/14%
- [x] Demographics missing values correctly detected (na_values fix)
- [x] Skewness corrected to 62 features on original_* only
- [x] Δt=0 pair investigated: Patient-042, week-010-1 (Stable) and week-010-2
      (Progressive) — two clinical visits in the same week. Decision: drop
      week-010-1 in preprocessing (keep the later assessment). Document in paper
      Methods: "One scan with Δt=0 (Patient-042, week 10) removed prior to
      preprocessing." n_effective may decrease by 1 after this removal.

**Preprocessing (Step 0.2)**
- [ ] Δt=0 pair resolved before build_dataset.py is written
- [ ] `dataset_paired.parquet` generated and DVC tracked
- [ ] 62 high-skew features log-transformed before normalisation
- [ ] Missing scan count documented (how many scans dropped, which patients affected)

**Validation (Step 0.3)**
- [ ] Validation script passing all assertions
- [ ] `validation_report.json` committed

**Cross-cutting**
- [ ] Unit tests passing in CI
- [ ] Δt-only ablation baseline recorded (logistic regression on Δt alone)
- [ ] Missing value count and dropped scan count in paper draft Methods section