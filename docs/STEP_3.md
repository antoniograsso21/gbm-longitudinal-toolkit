# Step 3 — Baseline Models ⏳

## Objective
Establish rigorous performance baselines and run feature selection inside CV.
These results are the scientific benchmark against which the GNN in Step 4 is evaluated.

**Input**: `data/processed/dataset_engineered.parquet`
**Output**: MLflow experiment `baselines/` + `configs/selected_features.yaml`

---

## Why This Hierarchy is Mandatory

On n=231 with 64 patients, gradient boosting frequently matches or beats deep learning.
If the GNN does not beat LightGBM, that is an honest scientific result, not a failure.

```
Baseline 1: Logistic Regression   — linear boundary, minimum reference
Baseline 2: LightGBM              — strong non-linear, handles small n
Baseline 3: LSTM on flat vectors  — temporal without graph structure
Model:      Temporal GNN (Step 4) — graph + temporal
Ablation:   2-node GNN            — isolates value of edema compartment
```

The LR → LightGBM → LSTM → GNN progression isolates:
non-linearity → temporal modelling → graph structure.

---

## Cross-Validation Setup (mandatory for all models)

```python
from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5)
# groups = patient IDs   → prevents intra-patient leakage
# stratify = RANO class  → preserves class balance per fold
```

**Normalization inside each fold only:**
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## Feature Selection (inside CV, fold-by-fold)

mRMR + Stability Selection executed on the training fold only.

**Algorithm**: mRMR
```
max I(xi; y) - (1/|S|) * sum I(xi; xj ∈ S)
```
MI estimator: Kraskov k-NN (appropriate for continuous variables, small n).
Stability Selection: B=100 bootstrap replicates, keep features with P(xi selected) > τ=0.7.
Stability also measured across CV folds — features stable in both are the primary
biological interpretation basis.

**Output**: `configs/selected_features.yaml` — versioned and logged to MLflow.
This file is consumed by Step 4 `GraphConfig.node_feature_cols`.

---

## Metrics (all models, all folds)

```
macro F1      — primary (imbalanced, all classes equal weight)
MCC           — Matthews Correlation Coefficient
AUROC         — one-vs-rest per class
PR-AUC        — one-vs-rest per class (primary for Response 13%, Stable 11%)
```

**Accuracy is never reported.** With 76% Progressive, a trivial classifier
achieves 76% accuracy.

Report mean ± std across folds for all metrics.

---

## Baseline 1 — Logistic Regression

**File**: `src/models/logistic_baseline.py`
Cross-sectional: flat feature vector at single timepoint T.
Tests Assumption A3 — does temporal dynamics add signal?

```yaml
# configs/logistic_baseline.yaml
C: [0.01, 0.1, 1.0, 10.0]
max_iter: 1000
class_weight: balanced
multi_class: multinomial
solver: lbfgs
```

---

## Baseline 2 — LightGBM + SHAP

**File**: `src/models/gbm_baseline.py`

```yaml
# configs/gbm_baseline.yaml
n_estimators: [100, 300, 500]
max_depth: [3, 5, 7]
learning_rate: [0.01, 0.05, 0.1]
class_weight: balanced
early_stopping_rounds: 20
```

RandomizedSearchCV with n_iter=30 inside each fold.

**Temporal ablations A/B/C/D** (mandatory — leakage quantification):
- A: radiomic features only
- B: temporal features only (interval_weeks, scan_index, time_from_diagnosis_weeks)
- C: radiomics + temporal
- D: radiomics + temporal + delta (full input)

If macro F1(B) ≈ macro F1(C) → weak radiomic signal, declare in paper.
If macro F1(B) > 0.33 → temporal leakage confirmed, declare in paper.

**SHAP** (mandatory, run on best fold model):
- Beeswarm plot + top-20 features by mean |SHAP|, grouped by region (CE/ED/NC)
- interval_weeks SHAP rank — if top 5, leakage must be declared

---

## Baseline 3 — LSTM on Flat Vectors

**File**: `src/models/lstm_baseline.py`
Temporal model without graph structure. Isolates graph topology contribution.

```
Input: (batch, seq_len, n_features)
  → LSTM (hidden=64, layers=2, dropout=0.3)
  → Last hidden state
  → Linear(64, 3) → Softmax
```

Use `pack_padded_sequence` for variable-length sequences.

```yaml
# configs/lstm_baseline.yaml
hidden_size: [32, 64, 128]
num_layers: [1, 2]
dropout: [0.2, 0.3, 0.5]
learning_rate: [1e-3, 5e-4]
batch_size: 16
max_epochs: 100
patience: 15
```

Fixed seeds mandatory:
```python
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
```

---

## Comparison Table (target output)

| Model                     | macro F1 ± std | MCC | AUC-PD | AUC-SD | AUC-Resp | PR-AUC-Resp |
|---------------------------|----------------|-----|--------|--------|----------|-------------|
| Logistic Regression       |                |     |        |        |          |             |
| LightGBM (full D)         |                |     |        |        |          |             |
| Ablation A (radiomics)    |                |     |        |        |          |             |
| Ablation B (temporal)     |                |     |        |        |          |             |
| Ablation C (radio+temp)   |                |     |        |        |          |             |
| LSTM                      |                |     |        |        |          |             |
| GNN 2-node (Step 4)       |                |     |        |        |          |             |
| GNN 3-node (Step 4)       |                |     |        |        |          |             |

---

## Definition of Done

- [ ] LR CV results logged to MLflow
- [ ] LightGBM CV results logged to MLflow
- [ ] Ablations A/B/C/D run and documented
- [ ] LSTM CV results logged to MLflow
- [ ] `selected_features.yaml` committed (used by Step 4)
- [ ] Normalization verified: scaler fit only on train fold
- [ ] All random seeds fixed and logged
- [ ] SHAP beeswarm + top-20 table saved
- [ ] interval_weeks SHAP rank reported
- [ ] PR-AUC reported per class
- [ ] Comparison table populated for all baselines
