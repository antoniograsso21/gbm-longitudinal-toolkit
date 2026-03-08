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
   fraction of the time — not approximately, but provably.

3. **Small n compatible**: unlike Bayesian approaches or Monte Carlo Dropout,
   CP requires no modification to the model architecture and works on n=68.

Alternative approaches considered and rejected:
- Monte Carlo Dropout: requires architecture changes, coverage not guaranteed
- Temperature scaling: only calibrates probabilities, no coverage guarantee
- Bayesian Neural Network: overkill on n=68, intractable posterior

---

## Conformal Prediction Setup

### Calibration split
CP requires a held-out calibration set that the model has never seen.
Use the validation fold from each CV split as the calibration set.

**IMPORTANT**: the calibration set must be separate from both training and test.
In a 5-fold CV this means splitting each fold into:
- 60% train
- 20% calibration
- 20% test

This reduces the effective training size slightly but is necessary for
valid coverage guarantees.

### Nonconformity score
For multi-class classification, use the Regularised Adaptive Prediction Sets
(RAPS) score, which produces smaller and more adaptive prediction sets than
the standard softmax-based score.

```python
# Nonconformity score for class y given softmax output s:
score(x, y) = 1 - s_y(x)

# For RAPS (penalises large prediction sets):
score(x, y) = 1 - s_y(x) + lambda * |S(x, y)|
```

where `|S(x, y)|` is the size of the prediction set and λ is a small penalty.

### Threshold calibration
```python
# On calibration set:
scores = [score(x_i, y_i) for x_i, y_i in calibration_set]
q_hat = np.quantile(scores, np.ceil((n+1)*(1-alpha)) / n)

# Prediction set for new example x:
C(x) = {y : score(x, y) <= q_hat}
```

### Coverage levels
Report results at three confidence levels:
- α=0.1 → 90% marginal coverage
- α=0.05 → 95% marginal coverage
- α=0.2 → 80% marginal coverage (smaller sets, clinically useful triage)

---

## Implementation

**File**: `src/uncertainty/conformal.py`

```python
@dataclass
class ConformalResult:
    alpha: float
    q_hat: float
    coverage: float             # empirical coverage on test set
    mean_set_size: float        # average prediction set size
    singleton_rate: float       # fraction of examples with |C(x)|=1
    empty_set_rate: float       # should be ~0 for valid CP

def calibrate(
    softmax_scores: np.ndarray,     # (n_cal, 3)
    true_labels: np.ndarray,        # (n_cal,)
    alpha: float,
) -> float:                         # returns q_hat
    ...

def predict_set(
    softmax_scores: np.ndarray,     # (n_test, 3)
    q_hat: float,
) -> list[list[int]]:               # prediction sets
    ...
```

---

## What to Report in the Paper

1. **Empirical coverage**: verify that the guaranteed coverage holds on the test set.
   If empirical coverage < (1-α), something is wrong with the calibration split.

2. **Mean prediction set size**: smaller is better. A set size of 1 means the
   model is confident. A set size of 3 means complete uncertainty.

3. **Clinically relevant analysis**:
   - What is the set size for Progressive predictions? (most common, should be small)
   - What is the set size for Response predictions? (rarest, likely larger sets)
   - Does the model produce {Progressive, Stable} sets or {Progressive, Response} sets?
     The latter would be clinically surprising and worth discussing.

4. **Conditional coverage**: does coverage hold separately for each RANO class?
   Marginal coverage (over all examples) is guaranteed; conditional coverage
   (per class) is not — but it should be checked and reported.

---

## Definition of Done for Phase 4

- [ ] Conformal calibration implemented with RAPS score
- [ ] Calibration split correctly separated from train and test in each CV fold
- [ ] Coverage verified at α ∈ {0.1, 0.05, 0.2}
- [ ] Mean set size and singleton rate reported per class
- [ ] Conditional coverage per RANO class checked and reported
- [ ] Results logged to MLflow experiment `uncertainty/`
- [ ] CP results integrated into comparison table from Phase 2/3
