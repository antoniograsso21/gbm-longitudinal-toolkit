# GBM Longitudinal Toolkit — Project Context

## Objective
Open-source framework for longitudinal analysis of glioblastoma (GBM).
Demonstrative case study: automatic RANO treatment response prediction on the LUMIERE dataset.
Target: preprint on bioRxiv + public GitHub repository.

## Dataset — LUMIERE
- 91 GBM patients, Swiss dataset, open access (Figshare, non-commercial license)
- 638 study dates, 2487 MRI images
- Sequences: CT1, T1, T2, FLAIR (skull-stripped, co-registered)
- Segmentation: HD-GLIO-AUTO — 2 regions: ET (Enhancing Tumor), NC (Necrotic Core)
- Extracted features: PyRadiomics (107 features per region per sequence = 428 features per node)
- Expert RANO labels: 399 valid timepoints, 81 patients with at least 1 label
- Class distribution: PD=253 (63%), SD=97 (24%), CR=27+PR=20=Response=47 (12%)
- Class grouping: CR+PR → Response | SD → Stable | PD → Progressive
- Patients with ≥3 valid timepoints: 55 (main subset for temporal model)
- Mean timepoints per patient: 4.9, max: 16
- Clinical data: MGMT methylation, IDH1, survival time, age, sex

### Raw MRI Data (not yet downloaded)
- Full MRI volumes available on Figshare alongside the CSV files
- Sequences: CT1, T1, T2, FLAIR (skull-stripped, co-registered, NIfTI format)
- Currently unused — radiomic features already extracted via PyRadiomics are sufficient for V1
- Raw MRI will be relevant for future work (see FUTURE.md): direct deep learning on images,
  peritumoral edema segmentation, visual QC of HD-GLIO-AUTO segmentations

## ⚠️ Critical Dataset Properties — Read Before Touching the Data

### 1. Label Shift (no temporal leakage)
The task is to predict RANO at the next timepoint. The dataset must be built as
explicit pairs (features_t, label_t+1). Direct consequences:

- A timepoint T is a valid example ONLY IF T+1 exists with an available RANO label
- The last timepoint of each patient is NEVER a training/test example
- n_effective (paired examples) is LOWER than 399 — compute it in Phase 0,
  log it, declare it in the paper. It is the variable that determines feasibility.

```python
# Mandatory schema — never deviate
paired_examples = []
for patient in patients_with_3plus_timepoints:
    for t in patient.timepoints[:-1]:
        if t has features AND t+1 has RANO label:
            paired_examples.append((features_t, label_t+1))
# len(paired_examples) = n_effective
```

### 2. Clinical Workflow Leakage (subtle, often ignored in literature)
LUMIERE is irregularly longitudinal: scan intervals are not fixed but determined
by clinical decisions correlated with the target.

Bias mechanism:
- Suspected progression → more frequent scans
- Stable patient → more distant follow-ups
- Consequence: Δt is implicitly correlated with RANO(t+1)
- The model may learn "short interval → PD" without using radiomic features

Three mandatory controls (to be executed and reported in the paper):
  a) Ablation study: model trained on Δt only, no radiomics.
     If performance > random → leakage confirmed, must be declared.
  b) Feature importance of Δt in the final model — if dominant, suspicious.
  c) Temporal binning as sanity check:
     early (0-8 weeks) / mid (8-20 weeks) / late (>20 weeks)

### 3. Temporal class imbalance (beyond standard class imbalance)
Patients responding to therapy tend to have fewer scans and more distant follow-ups.
The dataset is imbalanced not only in classes (PD=63%) but also in the temporal
distribution per class. To be declared explicitly in the Limitations section.

## Available Files in data/raw/lumiere/
- LUMIERE-pyradiomics-hdglioauto-features.csv (4792 rows, 152 columns)
- LUMIERE-ExpertRating-v202211.csv (616 rows — RANO labels)
- LUMIERE-Demographics_Pathology.csv (91 rows — clinical data)
- LUMIERE-datacompleteness.csv (638 rows — sequence availability per timepoint)

## Predictive Task
Formulation A: given the scan history t1...tN, predict the next RANO state at tN+1.
This is the most clinically useful formulation — it answers the real clinical question.
Note: many radiomics papers use features(t) → label(t), which is clinically useless.
This project uses features(t1..tN) → label(tN+1), which is the correct formulation.

## EDA Guidelines — Mandatory Rules for notebooks/
The unit of analysis in EDA is always the PATIENT, not the scan.

**Why**: LUMIERE has an imbalanced scan distribution per patient
(min=1, max=16, mean=4.9). A patient with 12 scans weights 4-6x more than others.
Per-scan statistics produce misleading results: small p-values and apparently strong
visual separations that actually reflect 3-4 dominant patients, not a population pattern.

Operational rules:
1. Every descriptive statistic (mean, std, distribution) must be computed
   first per patient (e.g., mean(Δf) per patient), then aggregated.
2. Every RANO class separation plot must be verified at two levels:
   - per-scan (to see the raw pattern)
   - per-patient (to verify how many patients actually show it)
3. The signal is real when: many patients show the same trend with varying intensity.
   Not when a few observations are very separated.
4. n_effective in EDA reports = number of patients, not number of scans.
5. Always compute and report: scans per patient (min/max/mean/std)
   and RANO class distribution per patient (not per timepoint).

## Architecture
```
Input: temporal series of graphs per patient
         ↓
Feature Selection: ~20-30 features (see Technical Decisions 7-9)
         ↓
Graph Construction per timepoint T:
  - Nodes: ET, NC (normalized PyRadiomics features)
  - Edges: ET↔NC bidirectional, weight = volumetric ratio
  - Delta-graph: (feature_T - feature_T-1) / delta_weeks
  - Temporal interval as explicit edge feature (monitor importance)
         ↓
Phase 2 — Baseline: full hierarchy (see Technical Decision 11)
         ↓
Phase 3 — Temporal GNN: GNN message passing + Temporal Attention
         ↓
Phase 4 — Uncertainty: Conformal Prediction (distribution-free)
         ↓
Output: RANO class + prediction set with calibrated confidence
```

## Critical Technical Decisions
1. Z-score normalization ONLY on training data of each fold (never fit on entire dataset)
2. StratifiedGroupKFold — group=patient, stratum=RANO class — never mix scans of same patient
3. Two-phase feature selection (see points 7-9) before the GNN
4. Delta-graph normalized by temporal interval: delta_feature / delta_weeks
5. Metrics: macro F1, MCC, AUC per class — NEVER accuracy (imbalanced classes)
6. 2-node graph — limitation to declare explicitly in the paper.
   With 2 nodes and 1 edge the graph is nearly equivalent to a feature vector. The value
   of the GNN lies in the temporal inductive bias, not topological complexity.
   Real architecture: node features → temporal dynamics → prediction
   i.e. temporal GNN ≈ LSTM + graph inductive bias. Fully defensible.
7. Feature selection — Phase 1 (Roadmap Phase 0-1): mRMR + Stability Selection
   - Algorithm: Minimum Redundancy Maximum Relevance
   - Formula: max I(xi; y) - (1/|S|) * sum I(xi; xj∈S)
   - Rationale: sample-efficient feature selection in p>>n regime (p≈428, n≈55).
   - MI estimation: Kraskov estimator (standard for continuous variables, small n)
   - Mandatory Stability Selection: repeat mRMR on bootstrap replicates,
     keep only features with P(xi ∈ S) > τ (τ=0.7 as default).
     Without this, on n=55 different features may be selected at each fold.
8. Feature selection — Phase 2 (Roadmap Phase 3-4): Temporal MI Stability
   - Criterion: consistency of ranking — not absolute stability of I(xi^t; y^t).
     Low Var_t[rank(I(xi^t; y^t))] = informationally robust feature.
   - Biological correction: absolute stability is biologically incorrect.
     Example: texture heterogeneity may become predictive AFTER therapy.
   - Implementation: temporal binning (baseline/early/late treatment) to
     stabilize estimation on small n.
   - Original contribution: I(xi^t; y^t) ≠ I(xi^t'; y^t') and selecting features
     with consistent MI ranking is biologically and mathematically motivated.
9. Discarded techniques — log for paper Methods section:
   - MINE: overkill on n=55, overfitting risk in the estimator itself
   - Direct Total Correlation: unstable at high dimensionality (p≈428, n=55)
   - t-SNE: does not preserve global structure, unsuitable for longitudinal trajectories
   - PCA: not biologically interpretable
   - UMAP as model input: unstable, hyperparameter-dependent
10. UMAP — allowed ONLY for exploratory visualization in the paper.
    Never as model features.
11. Baseline hierarchy — mandatory for paper credibility:
    Baseline 1: Logistic Regression (minimum reference)
    Baseline 2: XGBoost / LightGBM (strong on small datasets, often beats LSTM)
    Baseline 3: LSTM on flat vectors
    Model:      Temporal GNN
    Rationale: on small n gradient boosting is competitive. If GNN does not
    beat XGBoost, it is an honest scientific result, not a failure.
12. History length bias — explicit anti-leakage features:
    Add as node/graph features:
    - absolute time from diagnosis to timepoint T
    - number of previous scans for that patient
    Rationale: history length can be a proxy of the target.
    Making these features explicit allows monitoring their importance.

## Repository Structure
```
gbm-longitudinal-toolkit/
├── .github/workflows/ci.yml
├── data/raw/lumiere/           # original CSVs — never modified
├── data/processed/             # pipeline output — DVC versioned
├── src/
│   ├── audit/                  # dataset exploration
│   ├── preprocessing/          # build_dataset.py, normalizer.py
│   ├── graphs/                 # graph_builder.py, delta_graph.py
│   ├── models/                 # lstm_baseline.py, gnn.py, temporal_attention.py
│   ├── training/               # trainer.py, cross_validation.py, metrics.py
│   └── uncertainty/            # conformal.py
├── tests/
├── experiments/                # MLflow runs
├── notebooks/                  # EDA — not production
├── configs/                    # model parameter YAMLs
├── docs/                       # ADR + new dataset tutorial
├── FUTURE.md
├── CONTEXT.md                  # this file
├── pyproject.toml
└── dvc.yaml
```

## Roadmap
- Phase 0 — Data Foundation (2 weeks): build_clean_dataset.py, normalization, DVC
  ⚠️ PRIORITY 1: compute and log n_effective (paired examples after label shift)
  ⚠️ PRIORITY 2: per-patient EDA (follow EDA Guidelines above)
  ⚠️ PRIORITY 3: Δt-only ablation as workflow leakage sanity check
- Phase 1 — Graph Construction (3 weeks): parameterized GraphBuilder, delta-graph
- Phase 2 — Baseline (2 weeks): full hierarchy LR → XGB → LSTM
- Phase 3 — Temporal GNN (4 weeks): TumorGraphNet + TemporalAttention, ablation study
- Phase 4 — UQ (2 weeks): Conformal Prediction, calibration
- Phase 5 — Framework + Paper (4-6 weeks): parameterization, documentation, preprint

## Technology Stack
- Python 3.12, uv for dependency management
- PyTorch + PyTorch Geometric (GNN)
- scikit-learn (baseline, cross-validation, feature selection)
- XGBoost / LightGBM (strong baseline)
- MLflow (experiment tracking)
- DVC (data versioning)
- FastAPI (optional serving)
- Ruff + Mypy (pre-commit)
- Pytest (testing)

## Core Assumptions
These assumptions belong in the paper Methods section. If one fails, the project changes.
Verify empirically where possible.

A1. Radiomic features contain predictive signal for RANO(t+1).
    Verification: compare against Δt-only baseline (if Δt-only beats radiomics → A1 false).

A2. Expert RANO labels are reliable as ground truth.
    Known limitation: inter-rater variability not quantified in LUMIERE dataset.
    To be declared explicitly in the paper.

A3. Temporal feature dynamics contain additional signal beyond a single timepoint.
    Verification: compare LSTM/GNN vs cross-sectional model (features_t only).

A4. 2 nodes (ET, NC) are sufficient to represent the tumor structure
    relevant for RANO prediction.
    Known limitation: declare explicitly, propose extension in FUTURE.md.

## Scientific Claim
Defensible formulation:
"Open-source pipeline for longitudinal radiomics analysis in GBM with temporal
graph modelling and distribution-free uncertainty quantification"

Notes:
- Verify with systematic literature review before submission.
- The real value is the replicable longitudinal pipeline, not the specific model.
- Additional methodological contribution to validate: Temporal MI Stability
  (consistency of ranking) as a temporally-aware feature selection criterion.

## Important Notes
- New ideas → FUTURE.md (e.g. Neural ODE, temporal point process, 3rd edema node)
- GNN with 2 nodes may not beat baseline — limitation, not failure
- Generalizable framework is the main value, not the model on LUMIERE
- Seek clinical collaboration AFTER V1 for prospective validation
- Always distinguish: theoretically motivated choice vs pragmatically necessary choice