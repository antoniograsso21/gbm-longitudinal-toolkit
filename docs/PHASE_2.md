# Phase 2 — Baseline Models

## Objective
Establish a rigorous performance baseline before building the GNN.
The baseline hierarchy serves two purposes: it gives the paper a credible
comparison, and it provides an honest answer to "does the GNN actually help?".

**Input**: `data/processed/dataset_paired.parquet`
**Output**: MLflow experiment `baselines/` with metrics for all three models

---

## Why This Hierarchy is Mandatory

On small datasets (n=68 patients, 318 examples), gradient boosting is
frequently competitive with or superior to deep learning. If the GNN does
not beat XGBoost, that is an honest scientific result, not a failure.
The paper must include this comparison to be credible.

Skipping the baseline hierarchy is the most common methodological weakness
in radiomics papers.

```
Baseline 1: Logistic Regression   — minimum reference, linear boundary
Baseline 2: XGBoost / LightGBM    — strong non-linear, handles small n well
Baseline 3: LSTM on flat vectors  — temporal model without graph structure
Model:      Temporal GNN           — graph + temporal (Phase 3)
```

The progression from B1 to B3 to GNN isolates the contribution of each
component: non-linearity (B1→B2), temporal modelling (B2→B3), graph
structure (B3→GNN).

---

## Cross-Validation Setup

**Mandatory for all models in this phase.**

```python
from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5)
# groups = patient IDs — never split the same patient across folds
# stratify = RANO class — preserve class distribution in each fold
```

Never use KFold or train_test_split. A scan from a patient in the training
set must never appear in the test set. This is non-negotiable.

**Normalization inside each fold**:
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # fit only on train
```

The scaler is never fit on the full dataset.

---

## Metrics (all models, all folds)

```
macro F1       — primary metric (imbalanced classes, all classes equal weight)
MCC            — Matthews Correlation Coefficient (robust to imbalance)
AUC per class  — one-vs-rest AUROC for each of the 3 classes
```

**Accuracy is never reported as a primary metric.**
With 72% Progressive, a trivial classifier achieves 72% accuracy.

All metrics are computed per fold and reported as mean ± std.

---

## Baseline 1 — Logistic Regression

**File**: `src/models/logistic_baseline.py`
**Purpose**: minimum linear reference. If LR is competitive, the problem
may be linearly separable and the GNN is not needed.

**Features**: flat feature vector for a single timepoint T (no history).
This is the cross-sectional model — it tests Assumption A3 (temporal
dynamics add signal beyond a single timepoint).

**Config** (`configs/logistic_baseline.yaml`):
```yaml
C: [0.01, 0.1, 1.0, 10.0]        # regularisation strength (grid search)
max_iter: 1000
class_weight: balanced            # handles class imbalance
multi_class: multinomial
solver: lbfgs
```

**What to log to MLflow**:
- All metrics (macro F1, MCC, AUC per class)
- Best C value per fold
- Feature importances (coefficients)
- Δt feature coefficient — leakage check

---

## Baseline 2 — XGBoost / LightGBM

**File**: `src/models/gbm_baseline.py`
**Purpose**: strong non-linear baseline. This is often the hardest model
to beat on small tabular datasets.

**Features**: same flat feature vector as LR, plus delta features
(captures some temporal signal without explicit sequence modelling).

**Config** (`configs/gbm_baseline.yaml`):
```yaml
model: lightgbm                   # faster than XGBoost on small n
n_estimators: [100, 300, 500]
max_depth: [3, 5, 7]
learning_rate: [0.01, 0.05, 0.1]
class_weight: balanced
early_stopping_rounds: 20
```

Hyperparameter search: RandomizedSearchCV with n_iter=30 inside each fold.

**What to log to MLflow**:
- All metrics
- Feature importances (gain)
- Δt feature importance rank — if top 3, leakage is confirmed
- Best hyperparameters per fold

**Δt-only ablation** (mandatory, belongs here):
Train LightGBM on Δt alone (single feature).
If macro F1 > random baseline (0.33 for 3 balanced classes):
→ clinical workflow leakage confirmed, must be declared in paper.
Log result as experiment `baselines/delta_t_ablation`.

---

## Baseline 3 — LSTM on Flat Vectors

**File**: `src/models/lstm_baseline.py`
**Purpose**: temporal model without graph structure. Isolates the
contribution of the graph topology in Phase 3.

**Input**: sequence of flat feature vectors per patient
```
(n_timepoints × n_features) → RANO class at last T+1
```

Patients have variable-length sequences (1 to 15 timepoints).
Use `pack_padded_sequence` to handle variable length efficiently.

**Architecture**:
```
Input: (batch, seq_len, n_features)
  → LSTM (hidden=64, layers=2, dropout=0.3)
  → Last hidden state
  → Linear(64, 3)
  → Softmax
```

**Config** (`configs/lstm_baseline.yaml`):
```yaml
hidden_size: [32, 64, 128]
num_layers: [1, 2]
dropout: [0.2, 0.3, 0.5]
learning_rate: [1e-3, 5e-4]
batch_size: 16
max_epochs: 100
patience: 15                      # early stopping
```

**Random seeds** (mandatory):
```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

Fix seeds before every training run. Log seed value to MLflow.

**What to log to MLflow**:
- All metrics per fold
- Training curves (loss per epoch)
- Best hyperparameters per fold

---

## Comparison Table (target output for paper)

| Model           | macro F1 (mean±std) | MCC | AUC-PD | AUC-SD | AUC-Resp |
|-----------------|---------------------|-----|--------|--------|----------|
| Logistic Reg.   |                     |     |        |        |          |
| LightGBM        |                     |     |        |        |          |
| Δt-only         |                     |     |        |        |          |
| LSTM            |                     |     |        |        |          |
| Temporal GNN    |                     |     |        |        |          |

This table is generated automatically by `src/training/metrics.py`
reading from MLflow at the end of Phase 3.

---

## Definition of Done for Phase 2

- [ ] Logistic Regression implemented, CV results logged to MLflow
- [ ] LightGBM implemented, CV results logged to MLflow
- [ ] Δt-only ablation run and result documented
- [ ] LSTM implemented, CV results logged to MLflow
- [ ] Normalization verified: scaler fit only on train fold in each split
- [ ] All random seeds fixed and logged
- [ ] Feature importance of Δt reported for LightGBM (leakage check b)
- [ ] Comparison table populated for B1, B2, B3
