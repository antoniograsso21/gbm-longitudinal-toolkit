# Phase 4 — Uncertainty Quantification

## Objective
Equip the GNN predictions with calibrated, distribution-free confidence
intervals using Conformal Prediction.

**Input**: trained GNN from Phase 3, calibration split from CV
**Output**: prediction sets with guaranteed coverage at user-defined confidence level

---

## Why Conformal Prediction

In a clinical context, a single predicted class is not enough.
A radiologist needs to know: "how confident is this prediction?"

Conformal Prediction (CP) is the correct tool here for three reasons:

1. **Distribution-free**: no assumptions about the underlying data distribution.
   It works regardless of whether the model is well-calibrated.

2. **Guaranteed coverage**: for a user-specified confidence level α, CP
   guarantees that the true label is in the prediction set at least (1-α)
   fraction of the time — provably, not approximately.

3. **Small n compatible**: works on n=64 patients without architecture changes.

Alternatives considered and rejected:
- Monte Carlo Dropout: requires architecture changes, no coverage guarantee
- Temperature scaling: calibrates probabilities only, no coverage guarantee
- Bayesian Neural Network: intractable posterior, overkill on n=64

---

## Conformal Prediction Setup

### Calibration split
Use the validation fold from each CV split as calibration set.
In 5-fold CV this means splitting each fold into:
- 60% train / 20% calibration / 20% test

### Nonconformity score
RAPS (Regularised Adaptive Prediction Sets):
```python
score(x, y) = 1 - s_y(x) + lambda * |S(x, y)|
```
where `|S(x, y)|` penalises large prediction sets.

### Threshold calibration
```python
scores = [score(x_i, y_i) for x_i, y_i in calibration_set]
q_hat = np.quantile(scores, np.ceil((n+1)*(1-alpha)) / n)
C(x) = {y : score(x, y) <= q_hat}
```

### Coverage levels
- α=0.10 → 90% marginal coverage
- α=0.05 → 95% marginal coverage
- α=0.20 → 80% marginal coverage (smaller sets, clinical triage use)

---

## Clinical Output Integration (Phase 3 + Phase 4)

The primary clinical deliverable combines uncertainty quantification with
interpretability. At inference time, `src/interpretability/clinical_summary.py`
combines CP prediction set + Integrated Gradients attributions + temporal attention:

```
Patient: Patient-XXX  |  Scan: week-NNN

Predicted class:    Progressive
CP set (90%):       {Progressive}
CP set (95%):       {Progressive, Stable}

Top driving features (Integrated Gradients):
  [+] NC_T1_original_shape_MeshVolume      (+0.43)
  [+] CE_FLAIR_original_firstorder_Mean    (+0.31)
  [-] ED_T2_original_glcm_Correlation     (-0.18)

Most predictive timepoint: week-NNN (weight=0.62 of 5 scans)
```

**Additional check**: do low-confidence predictions (set size > 1) correspond
to diffuse IG attributions (no dominant feature)? This would indicate the model
knows when it doesn't know — a clinically valuable property.

---

## Implementation

**File**: `src/uncertainty/conformal.py`

```python
@dataclass
class ConformalResult:
    alpha: float
    q_hat: float
    coverage: float
    mean_set_size: float
    singleton_rate: float
    empty_set_rate: float      # should be ~0 for valid CP
```

---

## What to Report in the Paper

1. **Empirical coverage**: verify guaranteed coverage holds on test set
2. **Mean prediction set size** per class: Progressive (most common → small sets expected),
   Response (rarest → largest sets expected)
3. **Clinically notable set compositions**: does the model produce
   {Progressive, Stable} or {Progressive, Response} sets?
   The latter would be clinically surprising and worth discussing.
4. **Conditional coverage** per RANO class: marginal coverage is guaranteed;
   conditional is not — but must be checked and reported

---

## Definition of Done for Phase 4

- [ ] Conformal calibration with RAPS score implemented
- [ ] Calibration split correctly separated from train and test in each fold
- [ ] Coverage verified at α ∈ {0.05, 0.10, 0.20}
- [ ] Mean set size and singleton rate reported per class
- [ ] Conditional coverage per RANO class checked and reported
- [ ] Results logged to MLflow experiment `uncertainty/`
- [ ] CP results integrated into comparison table from Phase 2/3
- [ ] Clinical summary output implemented (CP + IG + attention weights)
- [ ] CP/Interpretability alignment check: set size vs IG attribution entropy
