# Step 3 — Baseline Models ⏳

## Objective
Establish rigorous performance baselines and run feature selection inside CV.
These results are the scientific benchmark against which the GNN in Step 4 is evaluated.

**Input**: `data/processed/dataset_engineered.parquet`
**Output**: MLflow experiment `baselines/` + `configs/selected_features.yaml` (produced by LightGBM D, T3.3)

---

## Input Schema — dataset_engineered.parquet (231 rows)

```
# Identifiers
Patient                            — patient ID string
Timepoint                          — timepoint string (e.g. "week-004")

# Target
target                             — RANO(t+1) string: "Progressive" | "Stable" | "Response"
target_encoded                     — integer: 0=Progressive, 1=Stable, 2=Response

# Temporal features (3)
interval_weeks                     — weeks T → T+1
time_from_diagnosis_weeks          — week_num of scan T
scan_index                         — 0-based ordinal per patient

# Auxiliary flags
is_baseline_scan                   — True for first scan per patient

# Radiomic features (1284)
{NC|CE|ED}_{CT1|T1|T2|FLAIR}_{feat}      — 107 features × 3 regions × 4 sequences

# Delta radiomic features (1284)
delta_{NC|CE|ED}_{CT1|T1|T2|FLAIR}_{feat} — Δf / interval_weeks

# Derived cross-compartment features (4)
CE_NC_ratio                        — CE / (NC + ε)
ED_CE_ratio                        — ED / (CE + ε)
CE_fraction                        — CE / (CE + NC + ED + ε)
total_tumor_volume                 — CE + NC + ED

# Derived nadir-based features (3)
CE_vs_nadir                        — (CE(T) + ε) / (min(CE[T0..T]) + ε)
weeks_since_nadir                  — weeks since best response
is_nadir_scan                      — True when CE(T) == min(CE[T0..T])

# Delta of derived features (2)
delta_CE_NC_ratio                  — Δ(CE_NC_ratio) / interval_weeks
delta_CE_vs_nadir                  — Δ(CE_vs_nadir) / interval_weeks

# Feature columns: 1284 + 1284 + 3 + 4 + 2 + 2 = 2579  (is_nadir_scan excluded from feature sets)
# Non-feature columns: Patient, Timepoint, target, target_encoded, is_baseline_scan = 5
# Total columns: 2579 + 5 + 1 = 2585  (2579 features + 5 non-feature + is_nadir_scan flag)
```

**Feature set definitions for ablations (referenced throughout this step):**
- **Radiomic set**: all 1284 `{NC|CE|ED}_*` columns + 4 cross-compartment + 2 nadir-based (CE_vs_nadir, weeks_since_nadir) + 2 delta-derived = 1292 columns
  Note: `is_nadir_scan` is a boolean flag — excluded from all feature sets passed to mRMR (Kraskov k-NN requires continuous input).
- **Temporal set**: `interval_weeks`, `scan_index`, `time_from_diagnosis_weeks` (3 columns)
- **Delta set**: all 1284 `delta_{NC|CE|ED}_*` columns (excludes delta_CE_NC_ratio and delta_CE_vs_nadir, which are in Radiomic set)
- **Full set (D)**: Radiomic + Temporal + Delta = 2579 columns (is_nadir_scan excluded — boolean flag)

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

## Implementation Order (mandatory)

```
T3.0  CV infrastructure + metrics
  ↓
T3.1  Feature selection (mRMR + Stability Selection) inside CV
  ↓
T3.2  Baseline 1: Logistic Regression
  ↓
T3.3  Baseline 2: LightGBM + ablations A/B/C/D + SHAP
  ↓
T3.4  Baseline 3: LSTM
  ↓
T3.5  MLflow consolidation + validator
```

Each task depends on the previous. T3.0 is the execution foundation.
T3.1 is a pure library — it has no standalone run, it is called inside every model's CV loop.

---

## T3.0 — CV Infrastructure and Metrics

**Files**: `src/training/cross_validation.py`, `src/training/metrics.py`

```python
from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5)
# groups = patient IDs   → prevents intra-patient leakage
# stratify = RANO class  → preserves class balance per fold
```

`cross_validation.py` exposes:
- `build_cv_splits(X, y, groups)` → list of (train_idx, test_idx) tuples
- Assertion: no patient appears in both train and test within any fold

`metrics.py` exposes:
- `compute_metrics(y_true, y_pred, y_proba)` → dict with macro_f1, MCC, AUROC per class, PR-AUC per class
- `aggregate_cv_results(fold_metrics: list[dict])` → mean ± std for each metric
- **Accuracy is never computed or returned**

Seed centralised in `configs/random_state.yaml` (seed=42), imported by all models.

**Normalization inside each fold only — mandatory pattern:**
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train only
X_test  = scaler.transform(X_test)        # transform only
```

---

## T3.1 — Feature Selection (inside CV, fold-by-fold)

**File**: `src/training/feature_selector.py` — pure library, **no standalone entry point**.

`feature_selector.py` exposes pure functions called inside the CV loop of each model.
There is no `run_feature_selection.py` — running feature selection before the models
would be data leakage (selection would see the full dataset including test folds).

The mandatory call pattern inside every `run_*.py`:
```python
# inside each fold, after normalisation:
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=all_feature_cols)
selection = select_features_fold(X_train=X_train_scaled_df, y_train=y_train, fold=k)
# each model then uses selection.selected_features (or its radiomic/temporal subset)
```

`selected_features.yaml` is produced **only** at the end of `run_lgbm_baseline.py`
(T3.3, ablation D) via `aggregate_fold_selections()`. It is a reporting artefact
consumed by Step 4 `GraphConfig.node_feature_cols` — it is never read by the baselines.

**Algorithm**: mRMR
```
max I(xi; y) - (1/|S|) * sum I(xi; xj ∈ S)
```
MI estimator: Kraskov k-NN via `npeet` library (correct for continuous variables, small n).
`sklearn.feature_selection.mutual_info_classif` is not used — it applies a different
discretisation-based estimator inappropriate for the radiomic feature distribution.

**Stability Selection**:
- B=100 bootstrap replicates on the training fold
- Feature bootstrap stability = fraction of replicates in which feature is selected
- Keep features with bootstrap stability > τ=0.7

**Cross-fold stability** (computed in T3.3 aggregation only):
- Stability score (fold) = fraction of folds in which feature passes bootstrap threshold
- Features stable in both dimensions are the primary biological interpretation basis

**Aggregation into selected_features.yaml** (T3.3 only):
- Majority vote: feature included if bootstrap-stable in ≥ 3/5 folds
- More robust than strict intersection on n=231

**Runtime notes**:
- Total MI calls: ~625k across 5 folds (50 mRMR steps × 25 avg redundancy × 100 bootstrap × 5 folds).
  This is the most expensive part of Step 3. Parallelised via joblib (n_jobs from configs/random_state.yaml).
  Expected runtime: 3–6h on CPU (i7-7700HQ) with joblib + MI cache. Run overnight.
  On laptops set n_jobs=6 in random_state.yaml to reduce thermal load (~2 thread headroom for OS).
- tau=0.7 with B=100 means a feature must appear in ≥70 replicates to be selected.
  On small n this can produce 2–11 features per fold. This is expected behaviour, not a bug.
  Monitor fold_k_n_selected in MLflow after the run. If variance is extreme (e.g. 1 vs 30),
  lower tau to 0.6 and rerun. Document the chosen tau in the paper Methods section.

**Execution scope**:
- mRMR + Stability Selection on **Full set (D)** in all models (LR, LightGBM, LSTM)
- LR uses the radiomic-only subset of the fold's selected features (no delta_*, no temporal)
- Ablation B (temporal only, 3 features) skips mRMR — no selection needed.
  Guard in `_run_ablation_cv`: mRMR is called only if `ablation != 'B'`.
  If multiple ablations are run together (default), mRMR runs once per fold
  and its result is shared across A/C/D — B receives a sentinel empty result.

---

## T3.2 — Baseline 1: Logistic Regression

**File**: `src/models/logistic_baseline.py`

Cross-sectional model: flat feature vector at single timepoint T (Radiomic set only, no delta).
Direct test of Assumption A3 — does temporal dynamics add signal beyond a static snapshot?

LR is the lower bound: if the GNN does not beat LR, the architecture adds no value.

```yaml
# configs/logistic_baseline.yaml
C: [0.01, 0.1, 1.0, 10.0]
max_iter: 1000
class_weight: balanced
solver: lbfgs
# multi_class removed — deprecated in scikit-learn 1.6+,
# lbfgs uses multinomial by default for multiclass problems
```

GridSearchCV inside each fold on the **selected features** (from T3.1, Full set D selection
applied to the Radiomic subset). Normalization applied before GridSearchCV.

---

## T3.3 — Baseline 2: LightGBM + Ablations A/B/C/D + SHAP

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

**Early stopping set**: 10% of the training fold held out as internal validation set
for `early_stopping_rounds`. This set is not used for metric evaluation.

**Temporal ablations A/B/C/D** (mandatory — leakage quantification):

| Ablation | Feature columns used |
|---|---|
| A | Radiomic set (1292 cols) — selected features only |
| B | Temporal set (3 cols) — no mRMR, default params: n_estimators=100, max_depth=3, learning_rate=0.05 |
| C | Radiomic + Temporal (1295 cols) — selected features + temporal |
| D | Full set: Radiomic + Temporal + Delta — mRMR input pool: 2579 cols → model input: selected features only |

Same hyperparameter grid applied to A/C/D for comparability. B uses default params
(3 features — search is not meaningful).

**Decision rules (log result and flag in paper if triggered):**
- If macro F1(B) ≈ macro F1(C): weak radiomic signal → declare in paper
- If macro F1(B) > 0.33: temporal leakage confirmed → declare in paper

**SHAP** (mandatory, run on best-fold model for ablation D):
- Beeswarm plot + top-20 features by mean |SHAP|, grouped by compartment (CE/ED/NC) and type (radiomic/temporal/delta)
- `interval_weeks` SHAP rank: if ≤ 5, leakage must be declared in paper
- Outputs saved to `data/processed/interpretability/shap_top20.csv` and `shap_beeswarm.png`

---

## T3.4 — Baseline 3: LSTM on Flat Vectors

**File**: `src/models/lstm_baseline.py`

Temporal model without graph structure. Isolates the contribution of graph topology
relative to temporal modelling alone. Expected result: LSTM ≈ LightGBM or worse,
given mean sequence length ~3.6 timepoints — this is an honest scientific result.

**Sequence construction**: for each patient, collect timepoints T₀...Tₙ₋₁ (label-shifted,
last timepoint excluded per the paired examples schema). Variable-length sequences
handled via `pack_padded_sequence`.

Input shape: `(batch, seq_len, n_features)` where n_features = features selected inside the LSTM CV loop (Full set D, same mRMR + Stability Selection pattern as LightGBM).

**Architecture (fixed)**:
```
LSTM(hidden=64, layers=2, dropout=0.3)
  → last hidden state
  → Linear(64, 3)
  → Softmax
```

**Hyperparameter search**: manual grid on hidden_size and learning_rate using the
internal validation set (10% of train fold, same as LightGBM). Best config logged per fold.

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

## T3.5 — MLflow Consolidation and Validator

**Files**: `src/training/trainer.py`, `src/validation/baselines_validator.py`

**MLflow experiment structure**:
```
baselines/logistic_regression    — 1 run: lr_cv (fold metrics + aggregated)
baselines/lgbm                   — 1 run: lgbm_ablations
                                   all ablations A/B/C/D in same run
                                   metrics prefixed: A_fold_k_*, B_fold_k_*, ...
                                   ablation D includes SHAP artifacts
baselines/lstm                   — 1 run: lstm_cv (fold metrics + aggregated)
```

All ablations in a single LightGBM run enables direct comparison in the
MLflow dashboard without cross-experiment queries.

**`baselines_validator.py`** post-hoc checks:
- All 5 folds have metrics logged for every model
- `selected_features.yaml` exists and contains ≥ 1 feature
- No metric value is NaN
- Comparison table is fully populated

---

## Metrics (all models, all folds)

```
macro F1      — primary (imbalanced, all classes equal weight)
MCC           — Matthews Correlation Coefficient
AUROC         — one-vs-rest per class (PD, SD, Response)
PR-AUC        — one-vs-rest per class (primary for Response 13%, Stable 11%)
```

**Accuracy is never reported.** With 76% Progressive, a trivial classifier
achieves 76% accuracy — the metric is meaningless here.

Report mean ± std across folds for all metrics.

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

- [ ] T3.0: CV splits verified (no patient leakage), metrics.py tested on synthetic data
- [ ] T3.1: `feature_selector.py` verified (pure functions, no standalone entry point)
- [ ] T3.2: LR CV results logged to MLflow, metrics aggregate computed
- [ ] T3.3: LightGBM ablations A/B/C/D on MLflow; SHAP beeswarm + top-20 table saved; decision rules documented
- [ ] T3.3: `selected_features.yaml` committed (produced by LightGBM D only), stability scores logged, fold-level JSONs saved
- [ ] T3.4: LSTM CV results logged to MLflow; `pack_padded_sequence` tested on variable-length inputs
- [ ] T3.5: validator exits 0; comparison table fully populated
- [ ] Normalization verified: scaler fit only on train fold (never on full dataset)
- [ ] All random seeds fixed and logged (seed=42 from `configs/random_state.yaml`)
- [ ] `interval_weeks` SHAP rank reported; leakage flag set if rank ≤ 5
- [ ] PR-AUC reported per class for all models