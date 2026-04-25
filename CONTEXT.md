# GBM Longitudinal Toolkit — Project Context

## Objective
Open-source framework for longitudinal analysis of glioblastoma (GBM).
Demonstrative case study: automatic RANO treatment response prediction on the LUMIERE dataset.
Target: preprint on bioRxiv + public GitHub repository.

## How to Navigate This Repository
Each step has a dedicated plan file with operational detail:
- `docs/STEP_0.md` — Audit
- `docs/STEP_1.md` — Preprocessing + Validation
- `docs/STEP_2.md` — Feature Engineering
- `docs/STEP_3.md` — Baseline Models
- `docs/STEP_4.md` — Graph Construction + GNN
- `docs/STEP_5.md` — Interpretability
- `docs/STEP_6.md` — Uncertainty Quantification
- `docs/STEP_7.md` — Framework
- `docs/STEP_8.md` — Paper

This file (CONTEXT.md) is the strategic document. Step files are the tactical ones.

**Session continuity**: at the start of each new chat session, ask Claude to read
CONTEXT.md and the relevant STEP_N.md before proceeding.

---

## Dataset — LUMIERE
- 91 GBM patients, Swiss dataset, open access (Figshare, non-commercial license)
- 638 study dates, 2487 MRI images
- Sequences: CT1, T1, T2, FLAIR (skull-stripped, co-registered)
- Two segmentation tools available:
  - **DeepBraTumIA** (primary): 3 labels — Necrosis, Contrast-enhancing, Edema
  - **HD-GLIO-AUTO** (reference): 2 labels — Non-enhancing, Contrast-enhancing
- Extracted features: PyRadiomics v3.0.1 (107 features per label per sequence)
- Expert RANO labels: 399 valid timepoints, 81 patients with at least 1 label
- Class grouping: CR+PR → Response | SD → Stable | PD → Progressive
- Patients with ≥3 valid timepoints: 55

### Step 0 Audit Results (completed)
Primary source: DeepBraTumIA

**RANO:**
- Valid timepoints: 398 (after Pre/Post-Op exclusion and Patient-042 deduplication)
- 81 patients with ≥1 valid timepoint; 68 with ≥2; 55 with ≥3

**DeepBraTumIA coverage:**
- 599 scans in CSV (39 absent from datacompleteness — extraction failed)
- 70 scans with all-NaN features (segmentation silent failure)
- 529 fully usable scans across 91 patients
- Partial-NaN scans: tracked in audit, dropped in preprocessing (any-NaN strategy)

**HD-GLIO-AUTO coverage (reference):**
- 599 scans in CSV (same 39 absent)
- 175 scans with all-NaN features — significantly worse than DeepBraTumIA
- 424 fully usable scans across 89 patients

**n_effective (true ML sample size — both t and t+1 must have usable scans):**
- DeepBraTumIA: **231 paired examples, 64 patients**  ← derived from LUMIERE v202211
  - Progressive=175 (76%), Stable=25 (11%), Response=31 (13%)
- HD-GLIO-AUTO: 158 paired examples, 54 patients (reference/ablation only)

**Patient anomalies resolved:**
- Patient-025: excluded entirely — temporal reference frame error in source data
- Patient-026, Patient-083: Rating='None' — auto-handled, zero valid timepoints
- Patient-042 week-010: duplicate SD+PD — PD kept (last occurrence)
- Patient-039: loses all paired examples after segmentation failure drop

**Temporal leakage:** low risk (Progressive=13.3w, Stable=13.3w, Response=16.0w)
**High-skew features:** 67 radiomic features with |skewness|>2 (log-transform candidates)

---

## ⚠️ Critical Dataset Properties — Read Before Touching the Data

### 1. Label Shift (no temporal leakage)
The task is to predict RANO at the next timepoint. The dataset must be built as
explicit pairs (features_t, label_t+1). Direct consequences:

- A timepoint T is a valid example ONLY IF T+1 exists with an available RANO label
- The last timepoint of each patient is NEVER a training/test example
- n_effective requires complete features at BOTH t AND t+1

```python
# Mandatory schema — never deviate
paired_examples = []
for patient in patients:
    for t in patient.timepoints[:-1]:
        if t has features AND t+1 has features AND t+1 has RANO label:
            paired_examples.append((features_t, label_t+1))
```

### 2. Clinical Workflow Leakage (subtle, often ignored in literature)
LUMIERE is irregularly longitudinal: scan intervals are not fixed but determined
by clinical decisions correlated with the target.

Four mandatory ablations (executed in Step 3, reported in paper):
  A) Radiomics features only (no temporal features)
  B) Temporal features only (interval_weeks, scan_index, time_from_diagnosis_weeks)
  C) Radiomics + temporal features
  D) Radiomics + temporal + delta features (full model input)
  If B ≈ C: weak radiomic signal — must declare in paper.
  Additional: interval_weeks SHAP rank in final LightGBM model.

### 3. Feature Selection must live inside cross-validation
Feature selection is executed fold-by-fold inside the training loop — never on
the full dataset. Doing it outside CV is data leakage. The parquet from Step 1
contains the full radiomic + delta pool; selection happens in Step 3.

### 4. Temporal class imbalance (beyond standard class imbalance)
Patients responding to therapy tend to have fewer scans and more distant follow-ups.
To be declared explicitly in the Limitations section.

---

## Available Files in data/raw/lumiere/
- LUMIERE-pyradiomics-deepbratumia-features.csv (7188 rows, 152 columns) — PRIMARY
- LUMIERE-pyradiomics-hdglioauto-features.csv (4792 rows, 152 columns) — REFERENCE
- LUMIERE-ExpertRating-v202211.csv (616 rows — RANO labels)
- LUMIERE-Demographics_Pathology.csv (91 rows — clinical data)
- LUMIERE-datacompleteness.csv (638 rows — sequence availability per timepoint)

---

## Predictive Task
Formulation: given the scan history t1...tN, predict the next RANO state at tN+1.
Note: many radiomics papers use features(t) → label(t), which is clinically useless.
This project uses features(t1..tN) → label(tN+1), which is the correct formulation.

---

## Pipeline Architecture
```
Raw CSVs
    ↓
Step 0 — Audit (lumiere_audit.py)
    ↓
Step 1 — Preprocessing + Validation
    dataset_paired.parquet: 231 rows, 2576 columns
    dataset_validator.py → validation_dataset_builder_report.json
    ↓
Step 2 — Feature Engineering (exploratory, no label-dependent selection)
    dataset_engineered.parquet: 231 rows, 2585 columns (+9 derived features)
    features_validator.py → features_validator_report.json
    ↓
Step 3 — Baseline Models (LR → LightGBM+SHAP → LSTM)
    feature selection MI univariate (with delta anchoring) inside CV here
    ↓
Step 4 — Graph Construction + Temporal GNN
    GraphConfig uses features selected in Step 3
    3-node triangular graph: NC ↔ CE ↔ ED
    GATv2Conv (1 layer) + Temporal Attention
    ↓
Step 5 — Interpretability
    SHAP global (Step 3 output), Integrated Gradients, attention weights
    ↓
Step 6 — Uncertainty Quantification
    Conformal Prediction (RAPS score)
    ↓
Step 7 — Framework (generalisation + CLI + DVC + tutorial)
    ↓
Step 8 — Paper (bioRxiv preprint)
```

---

## Critical Technical Decisions
1. Z-score normalization ONLY on training data of each fold (never fit on entire dataset)
2. StratifiedGroupKFold — group=patient, stratum=RANO class — never mix scans of same patient
   All cross-validation splits use Patient ID as grouping variable to prevent intra-patient leakage.
3. Primary segmentation source: DeepBraTumIA (3 nodes) — more coverage, biologically richer
   HD-GLIO-AUTO kept as reference for ablation (2-node graph vs 3-node graph)
4. Three-node graph: Necrosis ↔ Contrast-enhancing ↔ Edema (triangular topology)
   Extensible: adding a fourth node requires zero changes to downstream architecture
5. Delta-graph normalized by temporal interval: delta_feature / delta_weeks
6. Metrics: macro F1, MCC, AUROC per class, PR-AUC per class — NEVER accuracy
   PR-AUC is primary for minority classes (Response 13%, Stable 11%) under heavy imbalance
7. n_effective = 231 (DeepBraTumIA, LUMIERE v202211) — both t AND t+1 must have complete features
8. Feature selection: MI univariate (production) on radiomic-only subset
   - Executed inside CV only — never on full dataset
   - Primary selector: `sklearn.feature_selection.mutual_info_classif` (rank-based top percentile)
     chosen after diagnosing mRMR rank instability on LUMIERE (Spearman ρ=0.226 on n≈93
     per diagnostic replicate). mRMR remains as a reference path only.
   - **Feature pairing constraint (delta anchoring)** (label-free, no leakage):
     include `delta_f` **only if** the corresponding base feature `f` is selected.
     This prevents exploding the delta space (1284 deltas) on mean sequence length ~3.6.
     Implemented as: anchored_delta = {delta_f : f ∈ selected_radiomic AND delta_f passes variance}.
   - Feature set per model:
     LR: selected_radiomic only (cross-sectional, excludes nadir features)
     LightGBM/LSTM/GNN: selected_radiomic + temporal (3) + anchored_delta
   - Stability threshold \( \tau \) is calibrated empirically.
     Practical guidance: tau=0.7 (Meinshausen & Bühlmann) typically requires n>>200 per fold;
     LUMIERE has n≈185 per fold.
     **Current implementation default**: `STABILITY_THRESHOLD = 0.4` in `src/training/feature_selector.py`.
     If you want a more conservative selection, run with tau=0.5–0.6 and compare fold-level
     `n_radiomic_selected` and downstream metrics (document the chosen value in Methods).
   - Stability measured across bootstrap replicates (within fold) only.
     Cross-fold aggregation via majority vote (≥3/5 folds) for YAML.
   - selected_features.yaml produced exclusively by LightGBM ablation D:
     contains radiomic-only selected features (majority vote across folds).
     Consumed by Step 4 GraphConfig.node_feature_cols.
9. Discarded techniques: MINE, Direct Total Correlation, t-SNE, PCA, UMAP as model input;
   shift+log1p on bounded features; all-NaN detection (replaced by any-NaN per label block)
10. UMAP allowed ONLY for exploratory visualization in the paper
11. Baseline hierarchy — mandatory for paper credibility:
    Baseline 1: Logistic Regression (cross-sectional)
    Baseline 2: LightGBM + SHAP
    Baseline 3: LSTM on flat vectors
    Ablation A-D: temporal leakage quantification
    Ablation: 2-node (HD-GLIO-AUTO) vs 3-node (DeepBraTumIA)
    Model: Temporal GNN (3-node)
12. History length bias — explicit features:
    - time_from_diagnosis_weeks: absolute time from diagnosis to T
    - scan_index: 0-based ordinal per patient
13. Interpretability — first-class output:
    - Global: SHAP on LightGBM (Step 3), feature stability scores
    - Local: Integrated Gradients on GNN (Step 5), attention weight profiles
    - Clinical: prediction set + top driving features per prediction (Step 6)
14. GNN is an exploratory component, not the central scientific claim.
    The pipeline architecture and reproducibility are the primary contributions.
15. Generalisation deferred to Step 7: pipeline is LUMIERE-specific in V1.

---

## Feature selection cache (operational note)
Feature selection is expensive (especially the reference mRMR+stability path). A cached wrapper exists:
`src/training/training_utils.py::select_features_fold_anchored_cached`.

- Cache location: `data/processed/feature_selection_cache/`
- Cache keying: **content + parameters**, not just fold index.
  The cache filename encodes:
  - a data fingerprint of `(X_train, y_train)` for that fold, and
  - a parameter fingerprint (fold, B, n_select, tau, k_mi, seed, fast, variance_threshold).
- Consequence: changing `tau` (or other parameters) automatically changes the cache key.
  You should delete the cache directory only if you want to force recomputation despite an
  apparent cache hit (e.g., after code changes that alter behavior without changing the key).

---

## Preprocessing Summary (see STEP_1.md for detail)
1. Pivot DeepBraTumIA CSV → 1 row per (patient, timepoint)
2. Merge with RANO labels; deduplicate Patient-042 week-010
3. Label shift: target = RANO(t+1), drop last timepoint per patient
4. Drop scans where ANY feature for a segmentation label is NaN (63 dropped)
5. log1p transform 514 high-skew features (30 excluded: Hounsfield + bounded)
6. Add temporal features: interval_weeks, time_from_diagnosis_weeks, scan_index
7. Compute delta features: Δf = (f_t - f_{t-1}) / interval_weeks
8. Normalization: inside CV only (StandardScaler fit on train fold)

Output schema — dataset_paired.parquet:
```
Patient, Timepoint                         — identifiers
target, target_encoded                     — RANO(t+1), integer-encoded
interval_weeks                             — weeks T → T+1
time_from_diagnosis_weeks                  — week_num of scan T
scan_index                                 — 0-based ordinal per patient
is_baseline_scan                           — True for first scan per patient
{NC|CE|ED}_{CT1|T1|T2|FLAIR}_{feat}       — 1284 radiomic features
delta_{NC|CE|ED}_{CT1|T1|T2|FLAIR}_{feat} — 1284 delta features
```

---

## Feature Engineering Summary (see STEP_2.md for detail)
9 derived features added to dataset_engineered.parquet. All label-free.

Cross-compartment (4):
- CE_NC_ratio        — CE / (NC + ε): active tumor vs necrosis balance
- ED_CE_ratio        — ED / (CE + ε): edema disproportionality
- CE_fraction        — CE / (CE + NC + ED + ε): enhancing fraction of total burden
- total_tumor_volume — CE + NC + ED: overall tumor burden

Nadir-based (3) — computed per patient, chronologically up to T inclusive:
- CE_vs_nadir        — (CE(T) + ε) / (min(CE[T0..T]) + ε): closest radiomic proxy to RANO criterion
- weeks_since_nadir  — weeks elapsed since best response
- is_nadir_scan      — True when CE(T) == min(CE[T0..T])

Delta of derived (2):
- delta_CE_NC_ratio  — Δ(CE_NC_ratio) / interval_weeks
- delta_CE_vs_nadir  — Δ(CE_vs_nadir) / interval_weeks

EDA findings (informative, not prescriptive):
- Shape consistency: all shape features identical across sequences ✅
- Redundancy: low across all families (max 9% pairs with |r|>0.9) — keep all in mRMR pool
- Autocorrelation: mean r 0.43–0.54 across families — system is highly dynamic,
  both absolute values and deltas carry predictive signal

Output schema — dataset_engineered.parquet:
```
[all columns from dataset_paired.parquet]
CE_NC_ratio, ED_CE_ratio, CE_fraction, total_tumor_volume  — cross-compartment
CE_vs_nadir, weeks_since_nadir, is_nadir_scan              — nadir-based
delta_CE_NC_ratio, delta_CE_vs_nadir                       — delta derived
```

---

## Repository Structure
```
gbm-longitudinal-toolkit/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/lumiere/            # original CSVs — never modified, DVC tracked
│   └── processed/
│       ├── audit/              # dataset_stats.json
│       ├── preprocessing/      # dataset_paired.parquet, dataset_engineered.parquet
│       │                       # dataset_builder_report.json, features_builder_report.json
│       ├── validation/         # dataset_validator_report.json, features_validator_report.json
│       │                       # graphs_validator_report.json
│       └── graphs/             # {patient_id}.pt (Step 4)
├── src/
│   ├── utils/                  # lumiere_io.py — shared pure functions
│   ├── audit/                  # lumiere_audit.py
│   ├── validation/             # dataset_validator.py, features_validator.py
│   │                           # graphs_validator.py
│   ├── preprocessing/          # dataset_builder.py, features_builder.py
│   ├── graphs/                 # graphs_builder.py
│   ├── models/                 # logistic_baseline.py, gbm_baseline.py,
│   │                           # lstm_baseline.py, gnn.py, temporal_attention.py
│   ├── training/               # trainer.py, cross_validation.py, metrics.py
│   ├── interpretability/       # shap_baseline.py, integrated_gradients.py,
│   │                           # attention_vis.py, clinical_summary.py
│   └── uncertainty/            # conformal.py
├── tests/
├── experiments/                # MLflow runs
├── notebooks/                  # EDA — not production
├── configs/                    # model parameter YAMLs (features_builder.yaml, ...)
├── docs/                       # STEP_0.md ... STEP_8.md
├── FUTURE.md
├── CONTEXT.md
├── pyproject.toml
└── dvc.yaml
```

---

## Roadmap
- Step 0 — Audit ✅
- Step 1 — Preprocessing + Validation ✅
- Step 2 — Feature Engineering ✅
- Step 3 — Baseline Models
- Step 4 — Graph Construction + GNN
- Step 5 — Interpretability
- Step 6 — Uncertainty Quantification
- Step 7 — Framework
- Step 8 — Paper

---

## Technology Stack
- Python 3.12, uv for dependency management
- PyTorch + PyTorch Geometric (GNN)
- scikit-learn (baseline, cross-validation, feature selection)
- LightGBM (strong baseline)
- SHAP (interpretability for baseline models)
- Captum (Integrated Gradients for GNN)
- MLflow (experiment tracking)
- DVC (data versioning)
- Ruff + Mypy (pre-commit)
- Pytest (testing)

---

## Core Assumptions
A1. Radiomic features contain predictive signal for RANO(t+1).
    Verification: ablation A vs B in Step 3.
A2. Expert RANO labels are reliable as ground truth.
    Known limitation: inter-rater variability not quantified.
A3. Temporal feature dynamics contain additional signal beyond a single timepoint.
    Verification: compare LSTM/GNN vs cross-sectional model.
A4. Three nodes (Necrosis, Contrast-enhancing, Edema) capture the primary
    tumor compartments relevant to RANO assessment.
    Known limitation: single institution, automated segmentation errors not QC'd.

---

## Scientific Claim
Defensible formulation:
"Reproducible open-source pipeline for longitudinal radiomics analysis in GBM
with temporal bias control, calibrated uncertainty, and clinically interpretable
predictions — the GNN is an exploratory modelling component, not the central claim."

Notes:
- Verify with systematic literature review before submission.
- The real value is the replicable longitudinal pipeline, not the specific model.
- Additional methodological contribution: Temporal MI Stability (consistency of ranking).
- 2-node vs 3-node graph ablation is itself a contribution (graph topology sensitivity).
- GNN architecture must remain lightweight (1 GAT layer) to avoid overparameterisation on n=231.
- Mean sequence length ~3.6 timepoints limits sequence models to short-term dynamics (declare in Limitations).

---

## Software Engineering Principles
1. **Single Responsibility** — every function and module does exactly one thing.
2. **DRY** — if logic appears in two places, it belongs in `src/utils/lumiere_io.py`.
3. **Fail Fast and Explicitly** — invalid states raise errors immediately with descriptive messages.
4. **Pure Functions Where Possible** — reserve side effects for main().
5. **Occam's Razor** — prefer the simplest solution.
6. **Explicit Over Implicit** — type annotations on all public functions.
7. **Separation of Layers** — utilities → domain functions → entry point.
8. **Typed Results** — use dataclasses for structured return values.
9. **Centralised I/O** — all CSV loading through load_csv() in lumiere_io.py.
10. **No Premature Optimisation** — correct and readable first. On n=231, readability wins.
11. **Reproducibility** — fix random seeds. DVC for data. YAML for hyperparameters. MLflow for metrics.
12. **Step-level Validation** — every step that produces a dataset artifact must have a
    corresponding `validate_*.py` in `src/audit/`. The validator runs after the producer,
    saves a JSON report to `data/processed/`, and exits with code 1 on any FAIL so DVC
    and CI detect regressions automatically. Step 0 (audit) is exempt — it is itself a
    validation pass. Naming convention: `src/validation/dataset_validator.py`, `features_validator.py`,
    `graphs_validator.py`, etc. — all live in `src/validation/`. — semantic names, not step numbers.

---

## Important Notes
- New ideas → FUTURE.md (Neural ODE, raw MRI volumes, multi-institutional)
- GNN with 3 nodes may not beat baseline — limitation, not failure
- Generalizable framework is the main value, not the model on LUMIERE
- Seek clinical collaboration AFTER V1 for prospective validation
- Always distinguish: theoretically motivated choice vs pragmatically necessary choice