# Step 0 — Audit ✅

## Objective
Verify structural integrity of the raw LUMIERE CSVs before any processing.
Produce `dataset_stats.json` as the ground truth reference for all downstream steps.

**Scripts**: `src/audit/lumiere_audit.py`
**Run**: `uv run -m src.audit.lumiere_audit`
**Output**: `data/processed/dataset_stats.json`

---

## What the Audit Covers

1. Raw file shapes and missing value profiles
2. RANO label distribution, per-patient and per-scan
3. Temporal intervals between consecutive scans (leakage check)
4. Radiomic feature coverage per segmentation source
5. n_effective computation (paired examples with features at both t and t+1)

---

## Key Findings

**RANO:**
- 398 valid timepoints after Pre/Post-Op exclusion and Patient-042 deduplication
- 81 patients with ≥1 label; 68 with ≥2; 55 with ≥3

**DeepBraTumIA (primary):**
- 599 scans in CSV; 39 absent (extraction failed)
- 70 scans with all-NaN features (segmentation silent failure)
- 529 fully usable scans, 91 patients

**HD-GLIO-AUTO (reference):**
- 175 scans with all-NaN features — significantly worse than DeepBraTumIA
- 424 fully usable scans, 89 patients

**n_effective (audit estimate, before any-NaN drop):**
- DeepBraTumIA: 212 paired examples, 57 patients
- HD-GLIO-AUTO: 158 paired examples, 54 patients

**Temporal leakage:** low risk
- Progressive: 13.3w mean interval
- Stable: 13.3w
- Response: 16.0w

**Patient anomalies:**
- Patient-025: ALL RANO dates misaligned with imaging — excluded entirely
- Patient-026, Patient-083: Rating='None' — auto-handled
- Patient-042 week-010: duplicate SD+PD — PD kept (last occurrence)
- Patient-039: loses all examples after segmentation failure drop in Step 1

---

## Definition of Done ✅

- [x] `lumiere_audit.py` committed and passing
- [x] `dataset_stats.json` saved (HD-GLIO-AUTO + DeepBraTumIA stats)
- [x] DeepBraTumIA chosen as primary source
- [x] Patient anomalies documented
- [x] Temporal leakage assessed as low risk
