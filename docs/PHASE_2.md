# Phase 2 — Baseline Models

## Objective
Establish a rigorous performance baseline before building the GNN.

**Input**: `data/processed/dataset_paired.parquet`
**Output**: MLflow experiment `baselines/` with metrics for all models

---

## Why This Hierarchy is Mandatory

On small datasets (n=212, 57 patients), gradient boosting is frequently
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

Never use KFold or train_test_split. n=57 patients, 5 folds → ~11-12 patients
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
macro F1  — primary (imbalanced, all classes equal weight)
MCC       — Matthews Correlation Coefficient (robust to imbalance)
AUC       — one-vs-rest AUROC per class
```

**Accuracy is never reported.** With 77% Progressive, a trivial classifier
achieves 77% accuracy.

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

**delta_t-only ablation** (mandatory):
Train LightGBM on delta_t alone. If macro F1 > 0.33 (random baseline for
3 balanced classes) → clinical workflow leakage confirmed, declare in paper.
Log as experiment `baselines/delta_t_ablation`.

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
| delta_t-only           |                     |     |        |        |          |
| LSTM                   |                     |     |        |        |          |
| GNN 2-node (HD-GLIO)   |                     |     |        |        |          |
| GNN 3-node (DeepBraTumIA) |                  |     |        |        |          |

---

## Definition of Done for Phase 2

- [ ] Logistic Regression CV results logged to MLflow
- [ ] LightGBM CV results logged to MLflow
- [ ] delta_t-only ablation run and documented
- [ ] LSTM CV results logged to MLflow
- [ ] Normalization verified: scaler fit only on train fold
- [ ] All random seeds fixed and logged
- [ ] delta_t feature importance rank reported for LightGBM
- [ ] Comparison table populated for B1, B2, B3
