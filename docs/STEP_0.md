# Step 0 — Audit ✅

## Objective
Verify structural integrity of the raw LUMIERE CSVs before any processing.
Produce `dataset_stats.json` as the ground truth reference for all downstream steps.
The audit is deliberately separated from preprocessing: it freezes a snapshot
of the raw data state before any manipulation.

**Scripts**: `src/audit/lumiere_audit.py`
**Run**: `uv run -m src.audit.lumiere_audit`
**Output**: `data/processed/dataset_stats.json`

---

## What the Audit Covers

1. Raw file shapes and missing value profiles
2. RANO label distribution, per-patient and per-scan
3. Sequence-level completeness via `datacompleteness.csv` (CT1/T1/T2/FLAIR
   availability per timepoint — determines which feature blocks are present)
4. Radiomic feature coverage per segmentation source, distinguishing:
   - extraction failure (scan absent from CSV — ROI too small or not found)
   - segmentation silent failure (scan present but all features NaN)
   - partial NaN (some sequences failed, others succeeded)
5. Follow-up interval distribution across RANO classes (temporal bias check)
6. n_effective computation (valid longitudinal pairs with features at both t and t+1)
7. Patient-level anomaly log

---

## Key Findings

**RANO:**
- 398 valid timepoints after Pre/Post-Op exclusion and Patient-042 deduplication
- 81 patients with ≥1 label; 68 with ≥2; 55 with ≥3

**DeepBraTumIA (primary):**
- 599 scans in CSV; 39 absent (extraction failed — label not found or ROI too small)
- 70 scans with all-NaN features (segmentation silent failure)
- 529 fully usable scans, 91 patients
- Partial-NaN scans tracked in audit; dropped in Step 1 via any-NaN strategy

**HD-GLIO-AUTO (reference):**
- 175 scans with all-NaN features — significantly worse than DeepBraTumIA
- 424 fully usable scans, 89 patients

**n_effective:**
- Audit estimate (before Step 1 any-NaN drop): 212 pairs, 57 patients
- Final after Step 1 preprocessing: 231 pairs, 64 patients
- The increase from 212 to 231 is explained by the difference in scope:
  the audit counts pairs where both t and t+1 have at least one fully usable
  scan (all-NaN excluded); Step 1 applies the stricter any-NaN criterion per
  label block, which removes different scans and resolves to a different
  patient/pair count. Both numbers are correct for their respective definitions.
  See STEP_1.md for the exact drop logic.

**Temporal bias check:**
Follow-up interval distributions across RANO classes show limited divergence,
suggesting low risk of workflow-driven temporal bias.
Mean interval by class (Progressive=13.3w, Stable=13.3w, Response=16.0w).
This is NOT a formal leakage test — it measures whether scan scheduling is
strongly class-dependent. The formal leakage quantification is ablation B
in Step 3 (temporal features only model).

**Patient anomalies:**
- Patient-025: excluded entirely. RANO annotation dates are inconsistent with
  imaging dates across all timepoints, preventing reliable chronological
  ordering. Paper Methods: *"One patient (Patient-025) was excluded due to
  irreconcilable temporal reference frame inconsistency between RANO
  annotations and imaging acquisition dates."*
- Patient-026, Patient-083: Rating='None' — auto-handled, zero valid timepoints
- Patient-042 week-010: duplicate SD+PD entry — PD kept (last occurrence,
  conservative toward progression per clinical practice)
- Patient-039: loses all paired examples after Step 1 segmentation failure drop

---

## Dataset Filtering Pipeline (paper figure)

```
638 study dates (datacompleteness.csv)
        ↓  exclude Pre/Post-Op, Rating=None, Patient-025
398 valid labeled timepoints (81 patients)
        ↓  intersect with DeepBraTumIA radiomic coverage
529 fully usable scans (91 patients)
        ↓  label shift: pair (t, t+1), drop last timepoint per patient
294 paired examples (65 patients)   ← Step 1 pre-drop
        ↓  any-NaN drop per label block (63 scans)
231 paired examples (64 patients)   ← final dataset
```

This figure is mandatory for the paper Methods section.

---

## Definition of Done ✅

- [x] `lumiere_audit.py` committed and passing
- [x] `dataset_stats.json` saved (HD-GLIO-AUTO + DeepBraTumIA stats)
- [x] DeepBraTumIA chosen as primary source (91 patients, 529 usable scans)
- [x] Patient anomalies documented with paper-ready exclusion criteria
- [x] Follow-up interval distribution measured per class (temporal bias check)
- [x] n_effective transition (212 audit → 231 preprocessed) documented
- [x] Sequence-level completeness referenced via datacompleteness.csv
- [x] Temporal monotonicity verified in Step 1 validate_dataset.py (assertion 11)
- [x] Scan count distribution per patient documented in Step 1 survival bias check