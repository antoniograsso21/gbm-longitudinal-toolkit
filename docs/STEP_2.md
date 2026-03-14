# Step 2 — Feature Engineering ✅

## Objective
Two things happen here, in strict order:

1. **Derived feature computation** — deterministic transformations without using
   the label. Produces `dataset_engineered.parquet`.
2. **Exploratory analysis** — visualise the feature space to inform model design.
   Purely descriptive. Does NOT guide feature selection (that lives inside CV in Step 3).

**Input**: `data/processed/dataset_paired.parquet`
**Outputs**:
- `data/processed/dataset_engineered.parquet` — adds 9 derived features
- `data/processed/features_builder_report.json`
- `data/processed/features_validator_report.json`
- `notebooks/step2_feature_engineering.ipynb`
- `configs/feature_engineering.yaml`

**Scripts**:
- `src/preprocessing/features_builder.py`
- `src/audit/features_validator.py`

**Run**:
```bash
uv run -m src.preprocessing.features_builder
uv run -m src.audit.features_validator
```

---

## What Belongs Here vs Step 3

| Task | Step 2 | Step 3 |
|------|--------|--------|
| Cross-compartment derived features | ✅ | ❌ |
| Correlation / redundancy analysis | ✅ | ❌ |
| UMAP / t-SNE visualization | ✅ (exploratory only) | ❌ |
| mRMR feature selection | ❌ | ✅ (inside CV) |
| Stability Selection | ❌ | ✅ (inside CV) |
| StandardScaler normalization | ❌ | ✅ (inside CV) |

**Hard rule**: any operation that uses the label `target` must live inside CV in Step 3.
EDA plots that colour by class (2.4, 2.5) are descriptive only — they must not
drive feature inclusion decisions. Declare this explicitly in the paper Methods.

---

## Part A — Derived Feature Computation

### Rationale

PyRadiomics computes features per compartment in isolation. But GBM biology is
defined by the *relationship* between compartments, not by any single one.
The RANO criteria themselves are relational: progression is declared when CE
volume increases ≥25% relative to its nadir — not in absolute terms.

Six cross-compartment features and three nadir-based features are added.
All use only information available at time T (no leakage toward T+1).

### Cross-compartment volumetric ratios

Source feature: `{NC|CE|ED}_original_shape_MeshVolume`
Use CT1 sequence for CE and NC; FLAIR for ED.

```python
ε = 1.0  # mm³ — avoids division by zero on near-absent compartments

CE_NC_ratio        = CE_MeshVolume / (NC_MeshVolume + ε)
ED_CE_ratio        = ED_MeshVolume / (CE_MeshVolume + ε)
CE_fraction        = CE_MeshVolume / (CE + NC + ED + ε)
total_tumor_volume = CE_MeshVolume + NC_MeshVolume + ED_MeshVolume
```

**Biological interpretation**:
- `CE_NC_ratio`: high = active enhancing tumor dominates over necrosis (early
  proliferative phase). Low = necrosis dominant (aggressive, hypoxic tumor).
- `ED_CE_ratio`: high = edema disproportionate to enhancement (infiltrative
  pattern, often associated with poor response to therapy).
- `CE_fraction`: percentage of total tumor burden that is actively enhancing.
  Decreases with response, increases with progression.
- `total_tumor_volume`: overall tumor burden. Strongest single prognostic
  predictor in many GBM studies.

### Nadir-based features

The nadir is the minimum CE volume the patient has achieved up to and including T.
This directly encodes the clinical logic of RANO: progression is defined relative
to the nadir, not the previous scan.

**Critical implementation constraint**: compute nadir using only timepoints
present in the parquet (complete features). Using a dropped timepoint as nadir
reference would create an incoherent feature.

```python
# Computed per patient, chronologically up to T inclusive
CE_vs_nadir       = (CE_MeshVolume(T) + ε) / (min(CE_MeshVolume[T0..T]) + ε)
weeks_since_nadir = time_from_diagnosis_weeks(T) - argmin_week(CE_MeshVolume[T0..T])
is_nadir_scan     = (CE_MeshVolume(T) == min(CE_MeshVolume[T0..T]))
```

Note: ε is added to both numerator and denominator to preserve the `>= 1.0`
property under floating-point arithmetic.

**Biological interpretation**:
- `CE_vs_nadir`: > 1 means tumor has grown beyond its best response. The closest
  radiomic proxy to the clinical RANO measurement.
- `weeks_since_nadir`: two patients with identical `CE_vs_nadir` can be in
  different phases — one just reached nadir, the other has been stable for months.
- `is_nadir_scan`: flag analogous to `is_baseline_scan`. Tells the model that
  delta = 0 at nadir reflects best response, not absent history.

**Instability note**: with mean sequence length ~3.6, many patients will have
`CE_vs_nadir = 1.0` at T=0 and T=1 (nadir = current scan by definition).
This is correct behaviour. The model learns from `is_nadir_scan`.

### Delta of derived features (2 only)

```
delta_CE_NC_ratio  = Δ(CE_NC_ratio) / interval_weeks
delta_CE_vs_nadir  = Δ(CE_vs_nadir) / interval_weeks
```

Only these two have clear biological interpretation as rates of change.
Deltas of the other four are omitted to control p >> n.

### Output columns added (9 total)

```
CE_NC_ratio, ED_CE_ratio, CE_fraction, total_tumor_volume
CE_vs_nadir, weeks_since_nadir, is_nadir_scan
delta_CE_NC_ratio, delta_CE_vs_nadir
```

**File**: `src/preprocessing/features_builder.py`
**Run**: `uv run -m src.preprocessing.features_builder`
**Output**: `data/processed/dataset_engineered.parquet`

---

## Part B — Exploratory Analysis

All plots are descriptive. They inform the paper but do not drive feature
selection decisions.

### 2.2 — Feature family redundancy

Inter-feature Pearson correlation within each label block (NC, CE, ED),
grouped by PyRadiomics family.

**Result**: redundancy is surprisingly low across all families — no family
exceeds 9% of pairs with |r| > 0.9. Shape is the most redundant (mean |r| ~0.55),
as expected from geometric covariance. This justifies keeping all families
in the mRMR pool in Step 3 rather than excluding any a priori.

Produce: heatmap per label block + family-level redundancy summary table.

### 2.3 — Shape feature cross-sequence consistency check

Shape features (MeshVolume, Sphericity, Maximum3DDiameter) depend only on the
mask, not image intensity — they should be identical across CT1/T1/T2/FLAIR
for the same compartment. Verify this as a data quality check.

**Result**: all shape features identical across sequences (max_abs_diff = 0.0
for all compartment/feature/sequence pairs). Segmentation is consistent. ✅

### 2.4 — Delta feature distributions by class (descriptive only)

Box plots of key delta features per class (Progressive / Stable / Response).
Goal: visual evidence that rate-of-change features carry signal.

⚠️ Uses label for colouring — descriptive only, must not drive selection.

### 2.5 — Temporal feature distributions by class (descriptive only)

Distribution of interval_weeks, scan_index, time_from_diagnosis_weeks,
weeks_since_nadir by class. Pre-modelling version of ablation B from Step 3.

⚠️ Same warning as 2.4.

### 2.6 — Sequence length distribution

Histogram of n_timepoints per patient by class.
Documents the ~3.6 mean sequence length limitation for paper Limitations.

### 2.7 — UMAP visualization (exploratory only)

UMAP on original radiomic + 9 derived features, coloured by RANO class.
Visual sanity check. Paper figures only — NOT model input.

### 2.8 — Temporal autocorrelation (t vs t+1)

For each radiomic feature, compute Pearson correlation between consecutive
timepoints via `groupby + shift` applied dynamically on the sorted parquet.
No `_prev` columns are stored — `dataset_builder.py` saves delta rates, not raw
previous values. The shift is computed at analysis time only.

```python
df_sorted = df.sort_values(["Patient", "time_from_diagnosis_weeks"])
for col in radiomic_cols:
    prev = df_sorted.groupby("Patient")[col].shift(1)
    mask = prev.notna()  # excludes baseline scans
    r = df_sorted.loc[mask, col].corr(prev[mask])
```

**Result**: all families show low autocorrelation (mean r between 0.43 and 0.54).
The system is highly dynamic — absolute values and delta features both carry
predictive signal. This justifies the delta-graph architecture (Step 4).

| Family | mean r | median r |
|--------|--------|----------|
| shape | 0.540 | 0.524 |
| firstorder | 0.466 | 0.470 |
| glszm | 0.452 | 0.467 |
| ngtdm | 0.440 | 0.449 |
| glcm | 0.439 | 0.450 |
| glrlm | 0.439 | 0.462 |
| gldm | 0.428 | 0.436 |

Interpret:
- `corr > 0.9` → low dynamics, delta features carry most signal
- `corr < 0.5` → high dynamics, absolute values also informative

⚠️ Descriptive only — does not drive feature selection.

---

## PyRadiomics Feature Families — Biological Reference

| Family | N | Biological meaning | Reliability on auto-segmentation |
|--------|---|--------------------|----------------------------------|
| shape | 14 | Volume, diameter, sphericity, surface | High — mask-only, image-independent |
| firstorder | 18 | Intensity statistics (mean, entropy, percentiles) | Medium |
| glcm | 24 | Local texture — co-occurrence of intensity pairs | Medium-low |
| glrlm | 16 | Run-length patterns | Low |
| glszm | 16 | Zone-size patterns | Low |
| gldm | 14 | Dependency patterns | Low |
| ngtdm | 5 | Neighborhood tone difference | Medium-low |

**Most clinically interpretable features per compartment**:

| Compartment | Sequence | Features | Clinical rationale |
|-------------|----------|----------|--------------------|
| CE | CT1 | MeshVolume, Maximum3DDiameter, Sphericity | RANO measurement basis |
| CE | CT1 | SurfaceVolumeRatio, JointEntropy | Infiltrativity + heterogeneity |
| CE | CT1 | firstorder_Mean, firstorder_90Percentile | Contrast uptake intensity |
| NC | T1 | MeshVolume, firstorder_Entropy | Necrotic burden + heterogeneity |
| ED | FLAIR | MeshVolume, firstorder_Mean | Vasogenic edema load |

---

## Outputs

```
data/processed/dataset_engineered.parquet
notebooks/step2_feature_engineering.ipynb
configs/feature_engineering.yaml:
    epsilon: 1.0
    volume_features:
        CE: CE_CT1_original_shape_MeshVolume
        NC: NC_CT1_original_shape_MeshVolume
        ED: ED_FLAIR_original_shape_MeshVolume
    derived_features:
        - CE_NC_ratio
        - ED_CE_ratio
        - CE_fraction
        - total_tumor_volume
        - CE_vs_nadir
        - weeks_since_nadir
        - is_nadir_scan
        - delta_CE_NC_ratio
        - delta_CE_vs_nadir
```

---

## Output Schema — dataset_engineered.parquet

**Shape**: 231 rows × 2585 columns

| Group | Count | Naming convention | dtype |
|---|---|---|---|
| Radiomic (absolute) | 1284 | `{LABEL}_{SEQ}_original_{family}_{feature}` | float64 |
| Delta radiomic | 1284 | `delta_{LABEL}_{SEQ}_original_{family}_{feature}` | float64 |
| Cross-compartment derived | 4 | `CE_NC_ratio`, `ED_CE_ratio`, `CE_fraction`, `total_tumor_volume` | float64 |
| Nadir-based (continuous) | 2 | `CE_vs_nadir`, `weeks_since_nadir` | float64 |
| Delta derived | 2 | `delta_CE_NC_ratio`, `delta_CE_vs_nadir` | float64 |
| Temporal / metadata | 3 | `interval_weeks`, `scan_index`, `time_from_diagnosis_weeks` | float64/int |
| Binary flags | 2 | `is_baseline_scan`, `is_nadir_scan` | bool |
| ID + target | 4 | `Patient`, `Timepoint`, `target`, `target_encoded` | str/int |

**Total**: 1284 + 1284 + 4 + 2 + 2 + 3 + 2 + 4 = 2585 ✅

**Class distribution (target)**:
- Progressive: 175 (75.8%)
- Response: 31 (13.4%)
- Stable: 25 (10.8%)

### Naming convention
- `LABEL` ∈ `{CE, NC, ED}`
- `SEQ` ∈ `{CT1, T1, T2, FLAIR}` — non tutte le combinazioni esistono
  (CE/NC usano CT1 per shape; ED usa FLAIR per volume)
- Simmetria verificata: ogni feature radiomic ha il suo `delta_` corrispondente (0 orfani)

### NaN policy
- **Radiomic assoluti**: 0 NaN — scans con any-NaN droppati in Step 1
- **Delta radiomic**: 0 NaN — le baseline scan hanno delta forzato a `0.0`
  by design in `dataset_builder.py` (`.where(~is_baseline, other=0.0)`), non NaN.
  Step 3 riceve un DataFrame completo; mRMR non richiede imputation.
- **Derived + delta derived**: 0 NaN — verificato post `features_builder.py`

### Columns excluded from mRMR pool (Step 3)
```python
NON_FEATURE_COLS = [
    "Patient", "Timepoint",       # identifiers
    "target", "target_encoded",   # target — mai dentro CV
    "is_baseline_scan",           # binary flag — creato in dataset_builder.py
    "is_nadir_scan",              # binary flag — non feature continua
]
```

`is_baseline_scan` è creata in `dataset_builder.py` (sub-step 6) e persiste nel
parquet. Non ha prefisso standard — va intercettata per nome esplicito.

### Handoff to Step 3
`dataset_engineered.parquet` è il solo input di Step 3.
Feature selection (mRMR + Stability Selection) opera su tutte le colonne
non in `NON_FEATURE_COLS`, esclusivamente dentro il CV loop.

---

## Definition of Done

- [x] `features_builder.py` implemented and passing unit tests
- [x] `dataset_engineered.parquet`: 231 rows × 2585 columns, zero NaN
- [x] `features_builder_report.json` saved
- [x] `features_validator_report.json` saved — all 10 assertions PASS, no FAIL
- [x] `is_nadir_scan` verified: is_nadir_scan == True only when CE_vs_nadir == 1.0
- [x] Nadir computed from parquet timepoints only (not raw RANO)
- [x] Shape feature cross-sequence consistency check done
- [x] Correlation heatmaps saved to `notebooks/figures/`
- [x] Sequence length histogram saved
- [x] `feature_engineering.yaml` committed
- [x] EDA plots documented as descriptive-only in notebook markdown
- [x] No label-dependent operations performed outside CV
- [x] Temporal autocorrelation analysis saved to `notebooks/figures/`