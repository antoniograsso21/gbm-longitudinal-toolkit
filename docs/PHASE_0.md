# Phase 0 — Data Foundation

## Objective
Verify that the dataset is structurally sound and build the clean paired
dataset that every subsequent phase will consume.

**Primary source**: DeepBraTumIA (3 labels: Necrosis, Contrast-enhancing, Edema)
**Output of this phase**: `data/processed/dataset_paired.parquet`
One row per (patient, timepoint) paired example with all radiomic features,
temporal features, and the target label.

---

## Where Phase 0 Sits in the Project

```
Raw CSVs (data/raw/lumiere/)
        │
        ▼
[Step 0.1] AUDIT          src/audit/lumiere_audit.py         ✅ DONE
        │   Output: data/processed/dataset_stats.json
        │
        ▼
[Step 0.2] PREPROCESSING  src/preprocessing/build_dataset.py ⏳ NEXT
        │   Output: data/processed/dataset_paired.parquet
        │
        ▼
[Step 0.3] VALIDATION     src/audit/validate_dataset.py      ⏳ TODO
            Output: data/processed/validation_report.json
```

---

## Step 0.1 — Audit ✅

**Script**: `src/audit/lumiere_audit.py`
**Run**: `python -m src.audit.lumiere_audit`

### Key findings

**RANO:**
- 398 valid timepoints after Pre/Post-Op exclusion and Patient-042 deduplication
- 81 patients with ≥1 label; 68 with ≥2; 55 with ≥3

**DeepBraTumIA (primary):**
- 599 scans in CSV; 39 absent (extraction failed — label not found or ROI too small)
- 70 scans with all-NaN features (segmentation silent failure)
- 529 fully usable scans, 91 patients
- Partial-NaN scans tracked separately — included in model, handled in preprocessing

**HD-GLIO-AUTO (reference):**
- 175 scans with all-NaN features (vs 70 in DeepBraTumIA — markedly worse)
- 424 fully usable scans, 89 patients

**n_effective (both t AND t+1 must have usable scans):**
- DeepBraTumIA: **212 paired examples, 57 patients**
  - Progressive=163 (77%), Stable=23 (11%), Response=26 (12%)
- HD-GLIO-AUTO: 158 paired examples, 54 patients (reference only)

**Temporal leakage:** low (Progressive=13.3w, Stable=13.3w, Response=16.0w)
**Duplicate:** Patient-042 week-010 — SD and PD in raw file; PD kept (last), documented in Methods
**High-skew features:** 67 with |skewness|>2 in DeepBraTumIA

---

## Step 0.2 — Preprocessing ⏳

**Script**: `src/preprocessing/build_dataset.py`
**Run**: `python -m src.preprocessing.build_dataset`

### Sub-steps in order

**Sub-step 1 — Pivot the DeepBraTumIA CSV**

Input: `LUMIERE-pyradiomics-deepbratumia-features.csv` (7188 rows)
Structure: 1 row per (patient × timepoint × sequence × label)

Output: 1 row per (patient × timepoint)
Column naming: `{label}_{sequence}_{feature_name}`

```
Patient | Timepoint | Necrosis_CT1_feature_1 ... Edema_FLAIR_feature_107
                      (3 labels × 4 sequences × 107 features = 1284 columns max)
```

Implementation: `pd.pivot_table` or `groupby + unstack`.
Missing combinations (sequence absent for a label): NaN — handled in sub-step 5.

**Sub-step 2 — Merge with RANO labels**

Join on (Patient, Timepoint). Before joining:
- Exclude Pre-Op and Post-Op from RANO
- Apply RANO mapping: CR+PR → Response, SD → Stable, PD → Progressive
- Deduplicate Patient-042 week-010: keep last (PD)

**Sub-step 3 — Label shift**

For each patient, sort by week_num and shift label forward:
- target = RANO label of the NEXT timepoint
- Drop the last timepoint of every patient (no future label)
- Drop pairs where t+1 has no usable radiomic features

This is the most critical transformation. Verify: no patient's last timepoint
appears in the output. n_effective should be 212.

**Sub-step 4 — Add temporal features**

Three columns per example:
- `delta_t_weeks`: weeks between scan T and scan T+1
- `time_from_diagnosis_weeks`: week_num of scan T
- `scan_index`: ordinal position for this patient (0-based)

**Sub-step 5 — Handle missing values**

Strategy: scans with all-NaN for ALL labels are already excluded by the
feature join in sub-step 3. Scans with partial-NaN (some sequences missing)
are retained. For these scans, missing sequence features are imputed with
the per-feature mean computed on the training fold only (inside CV).

Document and report:
- Number of partial-NaN scans retained
- Which labels/sequences are most commonly missing

**Sub-step 6 — Log-transform high-skew features**

Apply log1p transform to the 67 features with |skewness|>2.
Apply before normalization, after pivot. Column names unchanged.
Document the feature list in `configs/log_transform_features.yaml`.

**Sub-step 7 — Compute delta features**

For each radiomic feature f, for each patient:
```
Δf_t = (f_t - f_{t-1}) / delta_t_weeks
```
- First scan per patient: Δf = 0, `is_baseline_scan = True`
- Delta columns named: `delta_{label}_{sequence}_{feature_name}`

Normalization is NOT performed here — it lives inside cross-validation.

### Output
`data/processed/dataset_paired.parquet`

---

## Step 0.3 — Dataset Validation ⏳

**Script**: `src/audit/validate_dataset.py`

### What it verifies
1. `n_effective == 212` (or document deviation with explanation)
2. No patient's last timepoint appears as a training example
3. No patient split contamination
4. Label distribution matches audit report
5. Delta features == 0 for all `is_baseline_scan == True` rows
6. Log-transform applied: verify skewness < 2 for the 67 transformed features
7. No future information: target = label of NEXT timepoint, not current

---

## Testing

Unit tests in `tests/test_preprocessing.py`. Synthetic data only — no real CSVs in CI.

Tests to cover:
- `parse_week`: known input/output pairs including edge cases
- `_compute_consecutive_pairs`: label shift on a 3-patient synthetic dataset
- `_float_week_to_str`: round-trip with `parse_week`
- Delta feature computation: Δf=0 on first scan, correct formula otherwise
- Pivot logic: column naming on a minimal synthetic DeepBraTumIA CSV
- Log-transform: skewness reduced below 2 on synthetic skewed data

---

## Definition of Done for Phase 0

**Audit (Step 0.1)** ✅
- [x] Audit script committed and passing
- [x] dataset_stats.json saved with both HD-GLIO-AUTO and DeepBraTumIA stats
- [x] DeepBraTumIA chosen as primary source (91 patients, 529 usable scans)
- [x] n_effective=212 (DeepBraTumIA), 158 (HD-GLIO-AUTO reference)
- [x] Patient-042 duplicate documented and resolution strategy defined
- [x] Partial-NaN scans tracked separately from all-NaN scans
- [x] Temporal leakage assessed as low risk

**Preprocessing (Step 0.2)**
- [ ] `dataset_paired.parquet` generated and DVC tracked
- [ ] 67 high-skew features log-transformed before normalization
- [ ] Partial-NaN scan count documented
- [ ] n_effective verified == 212 after join

**Validation (Step 0.3)**
- [ ] Validation script passing all assertions
- [ ] `validation_report.json` committed

**Cross-cutting**
- [ ] Unit tests passing in CI
- [ ] Log-transform feature list in `configs/log_transform_features.yaml`
- [ ] Missing sequence imputation strategy documented
