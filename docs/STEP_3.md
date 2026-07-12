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
  Note: `is_nadir_scan` is a boolean flag — excluded from all feature sets passed to continuous feature selectors.
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
T3.1  Feature selection (MI univariate, with delta anchoring) inside CV
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
selection = select_features_fold_anchored_cached(
    X_train=X_train_scaled_df, y_train=y_train, fold=k,
    cache_dir="data/processed/feature_selection_cache/"
)
# LR uses: selection.selected_radiomic
# LightGBM/LSTM/GNN use: selection.full_feature_set
```
`select_features_fold_anchored_cached` is defined in `training_utils.py` and wraps
`select_features_fold_anchored` with pickle-based fold-level caching under
`data/processed/feature_selection_cache/`.

**Caching behavior (important)**:
- Cache is **not** keyed on fold index alone.
- Cache filenames encode a fingerprint of the fold’s `(X_train, y_train)` **and** a fingerprint
  of the selection method and method-specific parameters:
  - MI production path: `method`, `fold`, `percentile`, `n_neighbors`, `seed`, `fast`, `variance_threshold`
  - mRMR reference path: `method`, `fold`, `B`, `n_select`, `tau`, `k_mi`, `seed`, `fast`, `variance_threshold`
- Changing any configured selector parameter therefore changes the cache key automatically.
  Delete the cache directory if code changes alter selector behavior without changing the
  configuration, or if you intentionally want to force recomputation despite an apparent cache hit.

`selected_features.yaml` is produced **only** at the end of `run_lgbm_baseline.py`
(T3.3, ablation D) via `aggregate_fold_selections()`. It contains the **radiomic-only**
selected features (majority vote across folds) — consumed by Step 4
`GraphConfig.node_feature_cols`. Never read by the baselines.

**Algorithm (production)**: univariate Mutual Information (MI), rank-based top percentile.

- **Selector**: `sklearn.feature_selection.mutual_info_classif` on the radiomic-only subset.
- **Why not mRMR here**: diagnostic probing on LUMIERE showed low ranking consistency
  (Spearman ρ=0.226 on n≈93 per diagnostic replicate), so mRMR+stability would inject
  selection noise. mRMR remains available as a **reference path** only.

**Cross-fold stability / aggregation** (computed in T3.3 only):
- `selected_features.yaml` is still a majority-vote aggregation across folds (≥3/5)
  for reproducible downstream use (Step 4).

**Aggregation into selected_features.yaml** (T3.3 only):
- Majority vote: feature included if selected by the fold-level selector in ≥ 3/5 folds
  (for MI, present in the radiomic top-percentile selection; for mRMR, stable within fold)
- More robust than strict intersection on n=231

**Runtime notes**:
- MI univariate is substantially cheaper than mRMR+stability. If switching methods,
  delete `data/processed/feature_selection_cache/` to avoid stale cache hits.

**Pre-filtering (variance threshold)**:
- Near-constant features (variance < 1e-6 on normalised train fold) are removed
  before MI scoring. Label-free — uses only X, no target information, safe inside CV.
  Improves robustness on small n and reduces computation.

**Feature selection flow (fold-by-fold, inside CV)**:
```
1. Variance threshold on full set (label-free) → remove near-constant features
2. Univariate MI on radiomic-only subset → selected_radiomic (top percentile, rank-based)
3. **Feature pairing constraint (delta anchoring)**:
   anchored_delta = {delta_f : f ∈ selected_radiomic AND delta_f passes variance}
4. full_feature_set = selected_radiomic + sorted(temporal) + sorted(nadir) + sorted(anchored_delta) + sorted(delta_derived)
```
Biological motivation for anchoring: if a radiomic feature is stable and
informative, its rate of change is biologically plausible. Including all
1284 delta features would introduce noise given mean sequence length ~3.6.
Delta features are NOT independently selected — anchoring enforces pairing and
keeps delta capacity proportional to selected radiomics.

**Execution scope**:
- MI univariate on **radiomic-only subset** in all models (production path)
- LR uses `selected_radiomic` only — pure cross-sectional static baseline
  (no delta, no temporal, no nadir features CE_vs_nadir/weeks_since_nadir)
- LightGBM/LSTM/GNN use `full_feature_set` = selected_radiomic + sorted(temporal) + sorted(nadir) + sorted(anchored_delta) + sorted(delta_derived)
- Ablation B (temporal only, 3 features) skips feature selection — sentinel result used.
  Guard in `_run_ablation_cv`: `select_features_fold_anchored` called only if `ablation != 'B'`.
  When multiple ablations run together, the configured selector runs once per fold and its result
  is shared across A/C/D — B receives an empty AnchoredFoldSelectionResult.

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

**File**: `src/models/lgbm_baseline.py`

`LGBMFoldResult` contains `feature_cols: list[str]` saved at train time alongside
the booster. SHAP uses `best_result.feature_cols` (not `booster_.feature_name_()`)
because `feature_name_()` can return internal names after model serialisation/deserialisation;
using the recorded columns avoids silent misalignment.

For robustness, feature columns are recorded explicitly in `LGBMFoldResult.feature_cols`
and used downstream (including SHAP) to avoid silent misalignment even if arrays are
passed at prediction time. Note: LightGBM may emit a warning if a model is fit with
feature names (DataFrame) and later evaluated with unnamed numpy arrays — this is
not a correctness issue if column ordering is consistent, but should be avoided in
future refactors by keeping DataFrames end-to-end where practical.

```yaml
# configs/lgbm_baseline.yaml
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
| D | Full set: Radiomic + Temporal + Delta — selector input pool: 2579 cols → model input: selected features only |

Same hyperparameter grid applied to A/C/D for comparability. B uses default params
(3 features — search is not meaningful).

**Decision rules (log result and flag in paper if triggered):**
- If macro F1(B) ≈ macro F1(C): weak radiomic signal → declare in paper
- If macro F1(B) > 0.38: temporal leakage confirmed → declare in paper
  (0.38 chosen as conservative threshold above trivial macro_F1≈0.29
  for a 76/11/13 class distribution; 0.33 would be correct only for balanced classes)

**SHAP** (mandatory, run on best-fold model for ablation D):
- Beeswarm plot + top-20 features by mean |SHAP|, grouped by compartment (CE/ED/NC) and type (radiomic/temporal/delta)
- Temporal feature SHAP ranks must be reported. If `interval_weeks` ranks ≤ 5,
  declare strong scheduling leakage; if other temporal features rank highly,
  discuss history-length/disease-time confounding explicitly.
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

Input shape: `(batch, seq_len, n_features)` where n_features = features selected inside the LSTM CV loop (Full set D, same MI-univariate + delta-anchoring pattern as LightGBM).

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

**`feature_selector.py`** exposes a single entry point: `select_features_fold_anchored`.
`select_features_fold` is eliminated — all callers use the anchored variant.
`training_utils.py` exposes `select_features_fold_anchored_cached` wrapping the above.

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
- Baseline JSONs under `data/processed/baselines/` exist; **Current Results** in this doc matches the latest run (or regenerate from those JSONs)

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

## Baseline comparison (Step 3)

The filled numeric table for LR, LightGBM A–D, and LSTM lives in **Current Results** below (single source of truth for Step 3). After Step 4, add GNN 2-node and 3-node rows there, or mirror them in the paper Methods/Results table from the Step 4 runs.

---

## Current Results (DeepBraTumIA, n_effective=231, 5-fold CV)

Run date: 2026-05-02. Reported as mean ± std across folds.

| Model                   | macro F1        | MCC             | AUC-PD          | AUC-SD          | AUC-Resp        | PR-AUC-Resp     | PR-AUC-Stable   |
|:------------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| Logistic Regression     | 0.3619 ± 0.0509 | 0.0361 ± 0.0329 | 0.5092 ± 0.1151 | 0.7619 ± 0.1328 | 0.4463 ± 0.0626 | 0.1849 ± 0.0754 | 0.4358 ± 0.1783 |
| LightGBM A (radiomic)   | 0.4045 ± 0.0459 | 0.1623 ± 0.1133 | 0.5734 ± 0.1532 | 0.7144 ± 0.1328 | 0.4693 ± 0.1979 | 0.2778 ± 0.0828 | 0.2759 ± 0.1235 |
| LightGBM B (temporal)   | 0.3725 ± 0.0364 | 0.1520 ± 0.0538 | 0.6758 ± 0.0931 | 0.4248 ± 0.1997 | 0.6082 ± 0.0910 | 0.3309 ± 0.1560 | 0.1338 ± 0.0422 |
| LightGBM C (radio+temp) | 0.3927 ± 0.0744 | 0.1372 ± 0.1087 | 0.6079 ± 0.1746 | 0.6710 ± 0.1183 | 0.5210 ± 0.2091 | 0.2555 ± 0.0614 | 0.2389 ± 0.1228 |
| LightGBM D (full)       | 0.3844 ± 0.0784 | 0.1305 ± 0.1305 | 0.6048 ± 0.1118 | 0.7748 ± 0.1427 | 0.4609 ± 0.1680 | 0.2223 ± 0.0950 | 0.3778 ± 0.2473 |
| LSTM                    | 0.3347 ± 0.1058 | 0.0581 ± 0.3171 | 0.4963 ± 0.3761 | 0.6429 ± 0.0000 | 0.4787 ± 0.3528 | 0.3365 ± 0.3405 | 0.1667 ± 0.0000 |
| **GNN Full Model**      | 0.3280 ± 0.0722 | 0.0697 ± 0.2454 | 0.5018 ± 0.3120 | 0.6429 ± 0.0000 | 0.5233 ± 0.3214 | 0.2608 ± 0.1523 | 0.1667 ± 0.0000 |

### Interpretation (what this implies for Step 4)

- **Temporal confounding remains relevant but is not the whole signal**: LightGBM B
  (temporal-only) is close to, but below, the predeclared macro-F1 leakage threshold
  (0.3725 vs 0.38) and has the strongest PR-AUC(Response), so scheduling/history effects
  should still be discussed as an important confounder. However, B is not the strongest
  macro-F1 baseline; LightGBM A (radiomic-only) is higher.
- **SHAP does not show interval-week dominance in the full model**: in LightGBM D,
  `interval_weeks` is absent from the top-20 SHAP table. The only temporal feature in
  `data/processed/interpretability/shap_top20.csv` is `time_from_diagnosis_weeks`
  at rank 12, after 11 radiomic features.
- **Dual evidence argues against strong scheduling leakage in the full model**:
  B is below the predeclared macro-F1 leakage threshold and `interval_weeks` ranks
  outside the SHAP top 20 in D. Scheduling/history confounding should still be declared,
  but not framed as the dominant full-model mechanism in this run.
- **Radiomic signal is present but modest**: LightGBM A is the strongest macro-F1 baseline,
  and the full-model SHAP top ranks are dominated by CE/NC radiomic features. Adding
  temporal and delta features does not improve macro F1 over radiomics alone in these runs.
- **Selected-feature biology is plausible but redundant**: `selected_features.yaml` is
  CE-heavy, consistent with RANO's contrast-enhancing focus; NC contributes mainly shape
  and selected texture features; ED is absent from the majority-vote radiomic set.
  Several CE shape features repeat across CT1/FLAIR/T1/T2 even though shape features are
  identical across sequences by construction, so report both nominal selected features
  and unique compartment/family-level patterns in the paper.
- **Delta anchoring does not materially improve over C** in these runs (D ≈ C).
- **LSTM does not clearly beat LightGBM**, consistent with the expectation under short mean
  sequence length (~3.6 timepoints). Treat as an honest result.

Implication: Step 4 should be framed as **exploratory** and must be compared directly to
LightGBM A (best macro-F1 radiomic baseline), LightGBM B (temporal-only confounder), and
LightGBM D (full selected-feature baseline) — beating LR alone is not sufficient.

---

## Definition of Done

- [x] T3.0: CV infrastructure implemented with `StratifiedGroupKFold`, patient grouping, and no-patient-overlap assertions
- [x] T3.0: Metric aggregation implemented for macro F1, MCC, AUROC per class, and PR-AUC per class; accuracy excluded
- [x] T3.1: Feature selector routed through `feature_selector.py` with MI-univariate production path and mRMR reference path
- [x] T3.1: Feature selection runs inside each CV fold only, after train-fold scaling; cached wrapper uses data + method-specific parameter fingerprints
- [x] T3.1: `selected_features.yaml` aggregation documented as cross-fold majority vote (selected in ≥3/5 folds), not MI bootstrap stability
- [x] T3.2: Logistic Regression CV completed; aggregate results saved to `data/processed/baselines/lr_results.json`
- [x] T3.3: LightGBM ablations A/B/C/D completed; fold/aggregate JSONs saved under `data/processed/baselines/`
- [x] T3.3: LightGBM D produced `configs/selected_features.yaml` and `data/processed/baselines/fold_stability.json`
- [x] T3.3: SHAP artifacts saved to `data/processed/interpretability/shap_top20.csv` and `shap_beeswarm.png`
- [x] T3.3: Decision rules documented: B below macro-F1 leakage threshold (0.3725 vs 0.38), `interval_weeks` outside SHAP top 20, temporal/history confounding still discussed
- [x] T3.4: LSTM CV completed; aggregate results saved to `data/processed/baselines/lstm_results.json`
- [x] T3.5: **Current Results** section updated from the 2026-05-02 baseline artifacts
- [x] Normalization pattern uses train-fold scaler fit only (`StandardScaler.fit_transform` on train, `transform` on test)
- [x] Random seed centralized through `configs/random_state.yaml` (seed=42)
- [x] PR-AUC reported per class for all baseline models in the Step 3 table/artifacts
- [x] Step 4 framing updated: compare GNN against LightGBM A, B, and D; beating LR alone is insufficient
- [ ] Baselines validator script/report is not present yet (`src/validation/baselines_validator.py` still to implement if required)
- [ ] Formal unit-test evidence for metrics/CV/LSTM packing is not recorded in this document
- [ ] Commit Step 3 documentation and artifact updates once reviewed
