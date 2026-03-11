# Step 2 — Feature Engineering ⏳

## Objective
Exploratory analysis of the feature space to inform model design.
No label-dependent transformations here — those belong inside CV in Step 3.

**Input**: `data/processed/dataset_paired.parquet`
**Output**: `notebooks/step2_feature_engineering.ipynb` + `configs/feature_engineering.yaml`

---

## What Belongs Here vs Step 3

| Task | Step 2 | Step 3 |
|------|--------|--------|
| Correlation analysis | ✅ | ❌ |
| Redundancy exploration (feature families) | ✅ | ❌ |
| mRMR feature selection | ❌ | ✅ (inside CV) |
| Stability Selection | ❌ | ✅ (inside CV) |
| StandardScaler normalization | ❌ | ✅ (inside CV) |
| UMAP / t-SNE visualization | ✅ (exploratory only) | ❌ |

The hard rule: any operation that uses the label `target` must live inside CV in Step 3.

---

## Analysis Tasks

### 2.1 — Feature redundancy
Inter-feature Pearson correlation within each label block (NC, CE, ED).
Goal: identify highly correlated feature families (shape, first-order, GLCM etc.)
to inform the mRMR budget in Step 3.

Produce: heatmap per label block, list of feature families with >0.9 internal correlation.

### 2.2 — Delta feature analysis
Distribution of delta features per class (Progressive vs Stable vs Response).
Goal: visual evidence that rate-of-change features carry signal.
Flag features where delta distribution is indistinguishable across classes.

### 2.3 — Temporal feature analysis
Distribution of interval_weeks, scan_index, time_from_diagnosis_weeks by class.
Goal: quantify clinical workflow leakage signal strength before running models.
This is the pre-modelling version of ablation B from Step 3.

### 2.4 — Sequence length distribution
Histogram of n_timepoints per patient, broken down by class.
Goal: document the ~3.6 mean sequence length limitation for the paper.

### 2.5 — UMAP visualization (exploratory only)
UMAP on the 1284 radiomic features, colored by RANO class.
Goal: visual sanity check — do classes form separable clusters?
NOT used as model input. Figures only.

---

## Outputs

```
notebooks/step2_feature_engineering.ipynb   — full analysis
configs/feature_engineering.yaml           — constants derived here:
    high_correlation_threshold: 0.9
    delta_signal_features: [...]            — features with clear class separation
    expected_feature_families: [shape, firstorder, glcm, glrlm, glszm, gldm, ngtdm]
```

---

## Definition of Done

- [ ] Correlation heatmaps saved to `notebooks/figures/`
- [ ] Delta distribution plots saved per class
- [ ] Temporal feature distributions documented
- [ ] Sequence length histogram saved
- [ ] `feature_engineering.yaml` committed
- [ ] No label-dependent operations performed outside CV
