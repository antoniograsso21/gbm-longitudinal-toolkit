# GBM Longitudinal Toolkit — Project Context

## Objective
Open-source framework for longitudinal analysis of glioblastoma (GBM).
Demonstrative case study: automatic RANO treatment response prediction on the LUMIERE dataset.
Target: preprint on bioRxiv + public GitHub repository.

## How to Navigate This Repository
Each phase has a dedicated plan file with operational detail:
- `docs/PHASE_0.md` — Data Foundation (audit + preprocessing + validation)
- `docs/PHASE_1.md` — Graph Construction
- `docs/PHASE_2.md` — Baseline Models
- `docs/PHASE_3.md` — Temporal GNN
- `docs/PHASE_4.md` — Uncertainty Quantification
- `docs/PHASE_5.md` — Framework + Paper

This file (CONTEXT.md) is the strategic document. Phase files are the tactical ones.

---

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
- Raw MRI relevant for future work (see FUTURE.md)

### Phase 0 Audit Results (completed)
- n_effective = 318 paired examples after label shift
- Patients represented = 68
- Class distribution: Progressive=229 (72%), Stable=45 (14%), Response=44 (14%)
- Clinical workflow leakage: Δt Progressive=13.3w, Stable=13.1w, Response=16.0w → low risk
- Missing values: 14.9% uniform across all radiomic features (entire scans missing)
- High-skew features: 68 radiomic features with skewness > 2 → log-transform candidates

---

## ⚠️ Critical Dataset Properties — Read Before Touching the Data

### 1. Label Shift (no temporal leakage)
The task is to predict RANO at the next timepoint. The dataset must be built as
explicit pairs (features_t, label_t+1). Direct consequences:

- A timepoint T is a valid example ONLY IF T+1 exists with an available RANO label
- The last timepoint of each patient is NEVER a training/test example
- n_effective (paired examples) is LOWER than 399 — already computed: 318

```python
# Mandatory schema — never deviate
paired_examples = []
for patient in patients_with_3plus_timepoints:
    for t in patient.timepoints[:-1]:
        if t has features AND t+1 has RANO label:
            paired_examples.append((features_t, label_t+1))
```

### 2. Clinical Workflow Leakage (subtle, often ignored in literature)
LUMIERE is irregularly longitudinal: scan intervals are not fixed but determined
by clinical decisions correlated with the target.

Three mandatory controls (to be executed and reported in the paper):
  a) Ablation study: model trained on Δt only, no radiomics.
  b) Feature importance of Δt in the final model.
  c) Temporal binning: early (0-8w) / mid (8-20w) / late (>20w)

### 3. Temporal class imbalance (beyond standard class imbalance)
Patients responding to therapy tend to have fewer scans and more distant follow-ups.
To be declared explicitly in the Limitations section.

---

## Available Files in data/raw/lumiere/
- LUMIERE-pyradiomics-hdglioauto-features.csv (4792 rows, 152 columns)
  Structure: 1 row per (patient × timepoint × sequence × region)
  → Must be pivoted to 1 row per (patient × timepoint) in preprocessing
- LUMIERE-ExpertRating-v202211.csv (616 rows — RANO labels)
- LUMIERE-Demographics_Pathology.csv (91 rows — clinical data)
- LUMIERE-datacompleteness.csv (638 rows — sequence availability per timepoint)

---

## Predictive Task
Formulation A: given the scan history t1...tN, predict the next RANO state at tN+1.
Note: many radiomics papers use features(t) → label(t), which is clinically useless.
This project uses features(t1..tN) → label(tN+1), which is the correct formulation.

---

## EDA Guidelines — Mandatory Rules for notebooks/
The unit of analysis in EDA is always the PATIENT, not the scan.

Operational rules:
1. Every descriptive statistic must be computed first per patient, then aggregated.
2. Every RANO class separation plot must be verified at two levels: per-scan and per-patient.
3. The signal is real when many patients show the same trend with varying intensity.
4. n_effective in EDA reports = number of patients, not number of scans.
5. Always compute and report: scans per patient (min/max/mean/std).

---

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

---

## Critical Technical Decisions
1. Z-score normalization ONLY on training data of each fold (never fit on entire dataset)
2. StratifiedGroupKFold — group=patient, stratum=RANO class — never mix scans of same patient
3. Two-phase feature selection (see points 7-9) before the GNN
4. Delta-graph normalized by temporal interval: delta_feature / delta_weeks
5. Metrics: macro F1, MCC, AUC per class — NEVER accuracy (imbalanced classes)
6. 2-node graph — limitation to declare explicitly in the paper.
   Real architecture: node features → temporal dynamics → prediction
   i.e. temporal GNN ≈ LSTM + graph inductive bias. Fully defensible.
7. Feature selection — Phase 1 (Roadmap Phase 0-1): mRMR + Stability Selection
   - Formula: max I(xi; y) - (1/|S|) * sum I(xi; xj∈S)
   - MI estimation: Kraskov estimator (standard for continuous variables, small n)
   - Stability Selection: keep only features with P(xi ∈ S) > τ=0.7 on bootstrap
8. Feature selection — Phase 2 (Roadmap Phase 3-4): Temporal MI Stability
   - Criterion: consistency of ranking — Low Var_t[rank(I(xi^t; y^t))]
   - Implementation: temporal binning (baseline/early/late treatment)
9. Discarded techniques — log for paper Methods section:
   - MINE, Direct Total Correlation, t-SNE, PCA, UMAP as model input
10. UMAP — allowed ONLY for exploratory visualization in the paper.
11. Baseline hierarchy — mandatory for paper credibility:
    Baseline 1: Logistic Regression
    Baseline 2: XGBoost / LightGBM
    Baseline 3: LSTM on flat vectors
    Model:      Temporal GNN
12. History length bias — add as explicit features:
    - absolute time from diagnosis to timepoint T
    - number of previous scans for that patient

---

## Preprocessing Pipeline (summary — see docs/PHASE_0.md for detail)
1. Pivot radiomic CSV: 4792 rows → 1 row per (patient, timepoint)
2. Merge with RANO labels on (Patient, Timepoint)
3. Apply label shift: assign label_t+1 as target, drop last timepoint per patient
4. Add temporal features: delta_t_weeks, time_from_diagnosis, scan_index
5. Handle missing values: drop scans with incomplete sequences (document count)
6. Compute delta features: Δf = (f_t - f_{t-1}) / delta_weeks
7. Normalization: inside cross-validation only (StandardScaler fit on train fold)

---

## Repository Structure
```
gbm-longitudinal-toolkit/
├── .github/workflows/ci.yml
├── data/raw/lumiere/           # original CSVs — never modified
├── data/processed/             # pipeline output — DVC versioned
├── src/
│   ├── audit/                  # lumiere_audit.py, validate_dataset.py
│   ├── preprocessing/          # build_dataset.py, normalizer.py
│   ├── graphs/                 # graph_builder.py, delta_graph.py
│   ├── models/                 # lstm_baseline.py, gnn.py, temporal_attention.py
│   ├── training/               # trainer.py, cross_validation.py, metrics.py
│   └── uncertainty/            # conformal.py
├── tests/                      # unit tests — no real CSVs, synthetic data only
├── experiments/                # MLflow runs
├── notebooks/                  # EDA — not production
├── configs/                    # model parameter YAMLs
├── docs/                       # PHASE_0.md ... PHASE_5.md + ADR
├── FUTURE.md
├── CONTEXT.md
├── pyproject.toml
└── dvc.yaml
```

---

## Roadmap
- Phase 0 — Data Foundation: audit ✅ | preprocessing → | validation →
- Phase 1 — Graph Construction (3 weeks): GraphBuilder, delta-graph
- Phase 2 — Baseline (2 weeks): LR → XGB → LSTM
- Phase 3 — Temporal GNN (4 weeks): TumorGraphNet + TemporalAttention
- Phase 4 — UQ (2 weeks): Conformal Prediction
- Phase 5 — Framework + Paper (4-6 weeks): documentation, preprint

---

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

---

## Core Assumptions
A1. Radiomic features contain predictive signal for RANO(t+1).
    Verification: compare against Δt-only baseline.
A2. Expert RANO labels are reliable as ground truth.
    Known limitation: inter-rater variability not quantified.
A3. Temporal feature dynamics contain additional signal beyond a single timepoint.
    Verification: compare LSTM/GNN vs cross-sectional model.
A4. 2 nodes (ET, NC) are sufficient to represent tumor structure for RANO prediction.
    Known limitation: declare explicitly, propose edema extension in FUTURE.md.

---

## Scientific Claim
Defensible formulation:
"Open-source pipeline for longitudinal radiomics analysis in GBM with temporal
graph modelling and distribution-free uncertainty quantification"

Notes:
- Verify with systematic literature review before submission.
- The real value is the replicable longitudinal pipeline, not the specific model.
- Additional methodological contribution: Temporal MI Stability (consistency of ranking).

---

## Software Engineering Principles

These principles apply to every module in this codebase.

1. **Single Responsibility** — every function and module does exactly one thing.
2. **DRY** — if logic appears in two places, it belongs in a shared utility.
3. **Fail Fast and Explicitly** — invalid states raise errors immediately with
   descriptive messages. A silent wrong result is worse than a loud exception.
4. **Pure Functions Where Possible** — reserve side effects for main().
5. **Occam's Razor** — prefer the simplest solution. Do not introduce abstractions
   unless they remove real duplication or manage real complexity.
6. **Explicit Over Implicit** — type annotations on all public functions.
   Named constants instead of magic strings or numbers.
7. **Separation of Layers** — utilities (pure logic) → domain functions
   (orchestrate + print) → entry point (I/O only).
8. **Typed Results** — use dataclasses for structured return values.
9. **Centralised I/O** — all CSV loading through a single _load_csv() function.
10. **No Premature Optimisation** — correct and readable first. On n=318,
    readability always wins.
11. **Reproducibility** — fix random seeds explicitly (Python, NumPy, PyTorch).
    Dataset versioned via DVC. All hyperparameters in YAML configs, never
    hardcoded. MLflow logs config alongside metrics.

---

## Important Notes
- New ideas → FUTURE.md (e.g. Neural ODE, 3rd edema node, raw MRI volumes)
- GNN with 2 nodes may not beat baseline — limitation, not failure
- Generalizable framework is the main value, not the model on LUMIERE
- Seek clinical collaboration AFTER V1 for prospective validation
- Always distinguish: theoretically motivated choice vs pragmatically necessary choice
