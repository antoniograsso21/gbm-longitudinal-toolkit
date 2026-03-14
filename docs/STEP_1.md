# Step 1 — Preprocessing + Validation ✅

## Objective
Transform raw LUMIERE CSVs into `dataset_paired.parquet` and verify its integrity.
This parquet is the single input consumed by all subsequent steps.

**Scripts**:
- `src/preprocessing/dataset_builder.py`
- `src/audit/dataset_validator.py`

**Run**:
```bash
uv run -m src.preprocessing.dataset_builder
uv run -m src.audit.dataset_validator
```

**Outputs**:
- `data/processed/dataset_paired.parquet`
- `data/processed/dataset_builder_report.json`
- `data/processed/validation_dataset_builder_report.json`

---

## Pipeline Sub-steps (actual execution order)

### 1. Pivot
Input: DeepBraTumIA CSV (7188 rows → 7092 after Patient-025 exclusion)
Output: 1 row per (Patient, Timepoint)
Column naming: `{NC|CE|ED}_{CT1|T1|T2|FLAIR}_{feature_name}`
Missing label/sequence combinations → NaN (handled in step 4)

### 2. Merge with RANO
Inner join on (Patient, Timepoint).
RANO cleanup: exclude Pre/Post-Op, map CR+PR→Response / SD→Stable / PD→Progressive,
deduplicate Patient-042 week-010 (keep PD).
Result: 372 merged rows.

### 3. Label shift
Sort by week_num per patient. Assign target = RANO(t+1). Drop last timepoint per patient.
Result: 294 pairs across 65 patients.
Integrity check: no patient's last timepoint in output.

### 4. Drop segmentation failures + log-transform
**Part A — any-NaN drop**: drop a scan if ANY feature for a segmentation label is NaN.
Catches both complete failures (all 107 NaN) and partial failures (e.g. 40/107 NaN).
Result: 63 scans dropped (NC: 62, CE: 17, ED: 11). Patient-039 loses all examples.

**Part B — log1p transform**: apply to features with |skewness| > 2.0.
Exclude `LOG_TRANSFORM_EXCLUDE` (Hounsfield intensities + bounded features).
Result: 514 features transformed, 30 excluded.
Applied BEFORE delta computation → delta becomes Δlog(f) / interval_weeks (log growth rate).

### 5. Add temporal features
After the drop, so scan_index is always 0-based and contiguous.
- `interval_weeks`: weeks T → T+1
- `time_from_diagnosis_weeks`: week_num of scan T
- `scan_index`: 0-based ordinal per patient

### 6. Compute delta features
`delta_f_t = (f_t - f_{t-1}) / interval_weeks`
First scan per patient: delta_f = 0, `is_baseline_scan = True`.
`is_baseline_scan` is included as a model feature so the GNN can distinguish
true zero-delta from biological zero-change.
Result: 1284 delta columns.

### 7. Finalize
Drop intermediate columns (week_num, Rating_grouped).
Recalculate scan_index after sort to ensure contiguity.

Normalization is NOT performed here — lives inside CV in Step 3.

---

## Output Schema

```
dataset_paired.parquet  —  231 rows × 2576 columns

Patient, Timepoint                         — identifiers
target, target_encoded                     — RANO(t+1), integer-encoded

temporal (3):
    interval_weeks                         — weeks T → T+1
    time_from_diagnosis_weeks              — week_num of scan T
    scan_index                             — 0-based ordinal per patient

radiomic (1284):
    {NC|CE|ED}_{CT1|T1|T2|FLAIR}_{feat}

delta (1284):
    delta_{NC|CE|ED}_{CT1|T1|T2|FLAIR}_{feat}
    = Δlog(f) / interval_weeks

flags (1):
    is_baseline_scan
```

---

## Validation Assertions (dataset_validator.py)

| # | Assertion | Type |
|---|-----------|------|
| 1 | n_effective == 231 (LUMIERE v202211) | FAIL |
| 2 | scan_index contiguous [0..n-1] per patient | FAIL |
| 3 | No NaN or inf in numeric columns | FAIL |
| 4 | Class dist: Progressive=175, Stable=25, Response=31 | FAIL |
| 5 | Delta features == 0 on is_baseline_scan rows | FAIL |
| 6 | Transformed cols have no negative values | FAIL |
| 7 | interval_weeks > 0 for all rows | FAIL |
| 8 | Patient-039 absent | FAIL |
| 9 | interval_weeks exists, delta_t_weeks absent | FAIL |
| 10 | No duplicate (Patient, Timepoint) | FAIL |
| 11 | time_from_diagnosis_weeks monotonic per patient | FAIL |
| 12 | Survival bias check (scans_per_patient, time_from_dx by class) | WARN |

---

## Definition of Done ✅

- [x] `dataset_paired.parquet`: 231 rows, 2576 columns, zero NaN, zero inf
- [x] `dataset_builder_report.json` saved
- [x] `validation_dataset_builder_report.json` saved — all 11 assertions PASS, no FAIL
- [x] any-NaN strategy: 63 scans dropped, Patient-039 loss documented
- [x] 514 features log-transformed, 30 excluded
- [x] scan_index assigned after drop (contiguous)
- [x] DVC tracking: dataset_stats.json, dataset_paired.parquet, validation_dataset_builder_report.json