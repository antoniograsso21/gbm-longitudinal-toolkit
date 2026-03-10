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
[Step 0.2] PREPROCESSING  src/preprocessing/build_dataset.py ✅ DONE
        │   Output: data/processed/dataset_paired.parquet
        │
        ▼
[Step 0.3] VALIDATION     src/audit/validate_dataset.py      ✅ DONE
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
- Partial-NaN scans tracked in audit for completeness; dropped in preprocessing (see Step 0.2 sub-step 5 — any-NaN strategy)

**HD-GLIO-AUTO (reference):**
- 4 scans with all-NaN features (vs 11 in DeepBraTumIA — markedly worse)
- 424 fully usable scans, 89 patients

**n_effective (both t AND t+1 must have usable scans):**
- DeepBraTumIA: **212 paired examples, 57 patients** (audit — before any-NaN drop)
  - Progressive=163 (77%), Stable=23 (11%), Response=26 (12%)
  - After preprocessing any-NaN drop: **231 paired examples, 64 patients** (see Step 0.2)
- HD-GLIO-AUTO: 158 paired examples, 54 patients (reference only)

**Temporal leakage:** low (Progressive=13.3w, Stable=13.3w, Response=16.0w)
**Patient anomalies:**
- Patient-025: ALL RANO timepoints misaligned with imaging dates — excluded entirely
  (temporal reference frame error). Paper Methods: *"One patient (Patient-025) was excluded
  due to irreconcilable temporal reference frame inconsistency."*
- Patient-026, Patient-083: Rating='None' — auto-handled, zero valid timepoints retained
- Patient-042 week-010: duplicate SD+PD — PD kept (last occurrence)
**High-skew features:** 67 with |skewness|>2 in DeepBraTumIA (audit estimate on raw CSV)

---

## Step 0.2 — Preprocessing ✅

**Script**: `src/preprocessing/build_dataset.py`
**Run**: `python -m src.preprocessing.build_dataset`

### Sub-steps in order

**Sub-step 1 — Pivot the DeepBraTumIA CSV**

Input: `LUMIERE-pyradiomics-deepbratumia-features.csv` (7188 rows raw, 7092 after Patient-025 exclusion)
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
This is the most critical transformation. Verify: no patient's last timepoint
appears in the output. After label shift: 294 pairs across 65 patients (before
segmentation failure drop in sub-step 5). Final n_effective=231 after sub-step 5.

**Sub-step 4 — Handle missing values + log-transform**

Moved before temporal features — see sub-step 5 below.

**Sub-step 4b — Add temporal features**

Three columns per example:
- `interval_weeks`: weeks between scan T and scan T+1
  (named `interval_weeks`, NOT `delta_t_weeks` — avoids collision with the
  `delta_` prefix used by radiomic delta features)
- `time_from_diagnosis_weeks`: week_num of scan T
- `scan_index`: 0-based ordinal position for this patient

**Sub-step 5 — Handle missing values + log-transform**

*(Note: in the actual pipeline, sub-step 5 runs BEFORE sub-step 4 so that `scan_index` is assigned after the drop and remains 0-based and contiguous.)*

*Part A — Drop scans with segmentation failures:*
A scan is dropped if ANY feature for a segmentation label is NaN.
This catches both complete segmentation failures (all 107 NaN) and partial
PyRadiomics failures on near-absent regions (e.g. Patient-032 week-027:
40/107 NC features NaN). Using `all-NaN` would be too permissive.
Result: 63 scans dropped (NC: 62, CE: 17, ED: 11 — NC is the dominant failure).
Patient-039 loses ALL examples — declared in paper.

*Part B — Log-transform high-skew features:*
Apply `log1p` to features with |skewness| > 2.0, EXCLUDING features that
can legitimately take negative values (`LOG_TRANSFORM_EXCLUDE` constant
in `lumiere_io.py`): CT Hounsfield intensities and bounded features
(glcm_Correlation, glcm_Imc1 etc. — applying log1p would change their meaning).
Result: 514 features log-transformed, 30 excluded.
Applied BEFORE delta computation so rates are on the log scale.

**Sub-step 6 — Compute delta features**

For each radiomic feature f, for each patient:
```
delta_f_t = (f_t - f_{t-1}) / interval_weeks
```
- First scan per patient: delta_f = 0, `is_baseline_scan = True`
- `is_baseline_scan` is included as a node feature in the GNN (via `GraphConfig.scalar_node_features`) so the model can distinguish true zero-delta from biological zero-change
- Delta columns named: `delta_{CE|ED|NC}_{sequence}_{feature_name}`
- Built with a single `pd.concat` to avoid DataFrame fragmentation

Normalization is NOT performed here — it lives inside cross-validation.

### Output
`data/processed/dataset_paired.parquet`

### Output schema

```
dataset_paired.parquet

Patient, Timepoint                     — identifiers
target, target_encoded                 — RANO(t+1), integer-encoded

temporal_features (3):
    interval_weeks                     — weeks T → T+1
    time_from_diagnosis_weeks          — week_num of scan T
    scan_index                         — 0-based ordinal per patient

radiomic_features (1284):
    {NC|CE|ED}_{CT1|T1|T2|FLAIR}_{feature_name}

delta_features (1284):
    delta_{NC|CE|ED}_{CT1|T1|T2|FLAIR}_{feature_name}
    (= Δlog(f) / interval_weeks — log-scale growth rate)

flags (1):
    is_baseline_scan                   — True for first scan per patient
```


---

## Step 0.3 — Dataset Validation ⏳

**Script**: `src/audit/validate_dataset.py`

### What it verifies
1. `n_effective == 231` (or document deviation with explanation)
2. `scan_index` is 0-based and contiguous per patient (label shift integrity)
3. No NaN or inf in numeric columns
4. Label distribution matches expected (Progressive=175, Stable=25, Response=31)
5. Delta features == 0 for all `is_baseline_scan == True` rows
6. Log-transform applied correctly: transformed cols have no negative values; excluded cols not transformed
7. No future information: `interval_weeks > 0` for all rows
8. Patient-039 does NOT appear in the dataset
9. `interval_weeks` column exists; no column named `delta_t_weeks`
10. No duplicate (Patient, Timepoint) pairs
11. `time_from_diagnosis_weeks` strictly increasing per patient
12. Survival bias check: `time_from_diagnosis_weeks`, RANO class, and `scans_per_patient` compared across dropped vs retained (informational)

---

## Testing

Unit tests in `tests/test_preprocessing.py`. Synthetic data only — no real CSVs in CI.

Tests to cover:
- `parse_week` (in lumiere_io.py): known input/output pairs including edge cases
- `load_and_clean_rano`: Patient-025 excluded, Patient-042 deduped, unmapped filtered
- `_compute_consecutive_pairs`: label shift on a 3-patient synthetic dataset
- Delta features: delta_f=0 on first scan, correct formula otherwise, no inf
- Pivot logic: column naming CE/ED/NC on minimal synthetic CSV
- Log-transform: LOG_TRANSFORM_EXCLUDE cols NOT transformed; others are
- Missing value drop: any-NaN in label block → drop; other-label NaN → keep

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

**Preprocessing (Step 0.2)** ✅
- [x] `dataset_paired.parquet` generated: 231 rows, zero NaN, zero inf
- [x] `preprocessing_report.json` saved
- [x] any-NaN strategy applied: 63 scans dropped, Patient-039 loss documented
- [x] LOG_TRANSFORM_EXCLUDE applied: 514 features transformed, 30 excluded
- [x] interval_weeks naming (not delta_t_weeks)
- [x] All integrity checks pass (label shift, delta baseline, no inf)

**Validation (Step 0.3)**
- [x] `src/utils/lumiere_io.py` written (shared utilities)
- [x] Validation script passing all assertions (10 hard + 1 warn)
- [x] `validation_report.json` committed

**Cross-cutting**
- [ ] Unit tests passing in CI
- [ ] interval_weeks-only ablation baseline recorded (Phase 2 prerequisite)
- [ ] Patient-025 and Patient-039 exclusions documented in paper Methods draft