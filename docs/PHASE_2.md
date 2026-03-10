# Phase 2 — Baseline Models

## Objective
Establish a rigorous performance baseline before building the GNN.

**Input**: `data/processed/dataset_paired.parquet`
**Output**: MLflow experiment `baselines/` with metrics for all models

---

## Why This Hierarchy is Mandatory

On small datasets (n=231, 64 patients), gradient boosting is frequently
competitive with or superior to deep learning. If the GNN does not beat
XGBoost, that is an honest scientific result, not a failure.

```
Baseline 1: Logistic Regression   — linear boundary, minimum reference
Baseline 2: LightGBM              — strong non-linear, handles small n well
Baseline 3: LSTM on flat vectors  — temporal without graph structure
Model:      Temporal GNN (3-node) — graph + temporal (Phase 3)
Ablation:   Temporal GNN (2-node) — HD-GLIO-AUTO, isolates graph topology value
```

The B1→B2→B3→GNN progression isolates: non-linearity, temporal modelling,
graph structure. The 2-node vs 3-node ablation isolates the value of the
edema compartment.

---

## Cross-Validation Setup

**Mandatory for all models.**

```python
from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5)
# groups = patient IDs
# stratify = RANO class
```

Never use KFold or train_test_split. n=64 patients, 5 folds → ~12-13 patients
per fold. This is a small test set — mean ± std across folds is mandatory.

**Normalization inside each fold only:**
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## Metrics (all models, all folds)

```
macro F1      — primary (imbalanced, all classes equal weight)
MCC           — Matthews Correlation Coefficient (robust to imbalance)
AUROC         — one-vs-rest per class
PR-AUC        — one-vs-rest per class (primary for Response and Stable)
```

**Accuracy is never reported.** With 76% Progressive, a trivial classifier
achieves 76% accuracy.

---

## Baseline 1 — Logistic Regression

**File**: `src/models/logistic_baseline.py`

Cross-sectional: flat feature vector for single timepoint T (no history).
Tests Assumption A3 — does temporal dynamics add signal?

**Config** (`configs/logistic_baseline.yaml`):
```yaml
C: [0.01, 0.1, 1.0, 10.0]
max_iter: 1000
class_weight: balanced
multi_class: multinomial
solver: lbfgs
```

---

## Baseline 2 — LightGBM

**File**: `src/models/gbm_baseline.py`

Same flat features as LR, plus delta features (captures some temporal signal).

**Config** (`configs/gbm_baseline.yaml`):
```yaml
model: lightgbm
n_estimators: [100, 300, 500]
max_depth: [3, 5, 7]
learning_rate: [0.01, 0.05, 0.1]
class_weight: balanced
early_stopping_rounds: 20
```

RandomizedSearchCV with n_iter=30 inside each fold.

**SHAP explanation** (mandatory — run after CV on best fold model):
```python
import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
# Report: beeswarm plot + top-20 features by mean |SHAP|, grouped by region (CE/ED/NC)
# Key check: interval_weeks SHAP rank — if top 5, leakage must be declared in paper
```

**Temporal feature ablations A/B/C/D** (mandatory):
Run LightGBM for each configuration:
- A: radiomic features only (no interval_weeks, scan_index, time_from_diagnosis_weeks)
- B: temporal features only (interval_weeks, scan_index, time_from_diagnosis_weeks)
- C: radiomics + temporal
- D: radiomics + temporal + delta features (full input)
If macro F1(B) ≈ macro F1(C) → weak radiomic signal, declare in paper.
If macro F1(B) > 0.33 → temporal leakage confirmed, declare in paper.
Log all four as experiments `baselines/ablation_{A,B,C,D}`.

---

## Baseline 3 — LSTM on Flat Vectors

**File**: `src/models/lstm_baseline.py`

Temporal model without graph structure. Isolates the graph topology contribution.

**Architecture**:
```
Input: (batch, seq_len, n_features)
  → LSTM (hidden=64, layers=2, dropout=0.3)
  → Last hidden state
  → Linear(64, 3)
  → Softmax
```

Use `pack_padded_sequence` for variable-length sequences.

**Config** (`configs/lstm_baseline.yaml`):
```yaml
hidden_size: [32, 64, 128]
num_layers: [1, 2]
dropout: [0.2, 0.3, 0.5]
learning_rate: [1e-3, 5e-4]
batch_size: 16
max_epochs: 100
patience: 15
```

**Seeds** (mandatory, fix before every run):
```python
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
```

---

## Comparison Table (target output)

| Model                  | macro F1 (mean±std) | MCC | AUC-PD | AUC-SD | AUC-Resp |
|------------------------|---------------------|-----|--------|--------|----------|
| Logistic Regression    |                     |     |        |        |          |
| LightGBM               |                     |     |        |        |          |
| Radiomics only (A)     |                     |     |        |        |          |
| Temporal only (B)      |                     |     |        |        |          |
| Radiomics+temporal (C) |                     |     |        |        |          |
| LSTM                   |                     |     |        |        |          |
| GNN 2-node (HD-GLIO)   |                     |     |        |        |          |
| GNN 3-node (DeepBraTumIA) |                  |     |        |        |          |

---

## Definition of Done for Phase 2

- [ ] Logistic Regression CV results logged to MLflow
- [ ] LightGBM CV results logged to MLflow
- [ ] Temporal ablations A/B/C/D run and documented (leakage + radiomic signal checks)
- [ ] LSTM CV results logged to MLflow
- [ ] Normalization verified: scaler fit only on train fold
- [ ] All random seeds fixed and logged
- [ ] SHAP global explanation for LightGBM saved (beeswarm + top-20 table)
- [ ] interval_weeks SHAP rank reported
- [ ] PR-AUC reported per class alongside AUROC
- [ ] Comparison table populated for B1, B2, B3