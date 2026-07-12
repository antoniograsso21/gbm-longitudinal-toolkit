# Step 6 — Uncertainty Quantification ⏳

## Objective
Equip GNN predictions with calibrated, distribution-free confidence intervals
using Conformal Prediction (CP).

**Input**: trained GNN from Step 4, calibration split from CV
**Output**: prediction sets with guaranteed marginal coverage

---

## Why Conformal Prediction

1. **Distribution-free**: no assumptions about data distribution
2. **Guaranteed coverage**: for confidence level α, true label is in prediction set
   at least (1-α) of the time — provably, not approximately
3. **Small n compatible**: works on n=64 patients without architecture changes

Alternatives rejected: MC Dropout (no coverage guarantee), Temperature Scaling
(calibrates probabilities only), BNN (intractable posterior on n=64).

---

## Setup

### Nonconformity score — RAPS
```python
score(x, y) = 1 - s_y(x) + lambda * |S(x, y)|
```

### Calibration split
Use validation fold from each CV split as calibration set.
Effective split per fold: 60% train / 20% calibration / 20% test.

### Threshold
```python
q_hat = np.quantile(scores, np.ceil((n+1)*(1-alpha)) / n)
C(x) = {y : score(x, y) <= q_hat}
```

### Coverage levels
- α=0.10 → 90% marginal coverage
- α=0.05 → 95% marginal coverage
- α=0.20 → 80% marginal coverage (smaller sets, clinical triage)

---

## What to Report

1. Empirical coverage vs guaranteed level (verify the guarantee holds)
2. Mean prediction set size per class
3. Clinically notable set compositions: {Progressive, Response} would be surprising
4. Conditional coverage per RANO class (marginal is guaranteed; conditional is not)

---

## Definition of Done

- [ ] RAPS conformal calibration implemented (`src/uncertainty/conformal.py`)
- [ ] Coverage verified at α ∈ {0.05, 0.10, 0.20}
- [ ] Mean set size and singleton rate reported per class
- [ ] Conditional coverage per class checked
- [ ] Results logged to MLflow `uncertainty/`
- [ ] Clinical summary output integrated (Step 5 + Step 6 combined)