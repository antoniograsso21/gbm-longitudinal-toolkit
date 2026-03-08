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
- Two segmentation tools available:
  - **DeepBraTumIA** (primary): 3 labels — Necrosis, Contrast-enhancing, Edema
  - **HD-GLIO-AUTO** (reference): 2 labels — Non-enhancing, Contrast-enhancing
- Extracted features: PyRadiomics v3.0.1 (107 features per label per sequence)
- Expert RANO labels: 399 valid timepoints, 81 patients with at least 1 label
- Class grouping: CR+PR → Response | SD → Stable | PD → Progressive
- Patients with ≥3 valid timepoints: 55

### Phase 0 Audit Results (completed)
Primary source: DeepBraTumIA

**RANO:**
- Valid timepoints: 398 (after Pre/Post-Op exclusion and Patient-042 deduplication)
- 81 patients with ≥1 valid timepoint; 68 with ≥2; 55 with ≥3

**DeepBraTumIA coverage:**
- 599 scans in CSV (39 absent from datacompleteness — extraction failed)
- 70 scans with all-NaN features (segmentation silent failure)
- 529 fully usable scans across 91 patients
- Partial-NaN scans (usable but some sequences missing): tracked, handled in preprocessing

**HD-GLIO-AUTO coverage (reference):**
- 599 scans in CSV (same 39 absent)
- 175 scans with all-NaN features — significantly worse than DeepBraTumIA
- 424 fully usable scans across 89 patients

**n_effective (true ML sample size — both t and t+1 must have usable scans):**
- DeepBraTumIA: **212 paired examples, 57 patients**
  - Progressive=163 (77%), Stable=23 (11%), Response=26 (12%)
- HD-GLIO-AUTO: 158 paired examples, 54 patients
  - Progressive=134 (85%), Stable=9 (6%), Response=15 (9%)

**Temporal leakage:** low risk (Progressive=13.3w, Stable=13.3w, Response=16.0w)
**High-skew features:** 67 radiomic features with |skewness|>2 in DeepBraTumIA (log-transform candidates)
**Duplicate:** Patient-042 week-010 had SD and PD — PD kept (last occurrence), documented in paper Methods

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

Three mandatory controls (to be executed and reported in the paper):
  a) Ablation study: model trained on delta_t only, no radiomics.
  b) Feature importance of delta_t in the final model.
  c) Temporal binning: early (0-8w) / mid (8-20w) / late (>20w)

### 3. Temporal class imbalance (beyond standard class imbalance)
Patients responding to therapy tend to have fewer scans and more distant follow-ups.
To be declared explicitly in the Limitations section.

---

## Available Files in data/raw/lumiere/
- LUMIERE-pyradiomics-deepbratumia-features.csv (7188 rows, 152 columns) — PRIMARY
  Structure: 1 row per (patient × timepoint × sequence × label)
  Labels: Necrosis, Contrast-enhancing, Edema
  → Must be pivoted to 1 row per (patient × timepoint) in preprocessing
- LUMIERE-pyradiomics-hdglioauto-features.csv (4792 rows, 152 columns) — REFERENCE
  Labels: Non-enhancing, Contrast-enhancing
- LUMIERE-ExpertRating-v202211.csv (616 rows — RANO labels)
- LUMIERE-Demographics_Pathology.csv (91 rows — clinical data)
- LUMIERE-datacompleteness.csv (638 rows — sequence availability per timepoint)

---

## Predictive Task
Formulation: given the scan history t1...tN, predict the next RANO state at tN+1.
Note: many radiomics papers use features(t) → label(t), which is clinically useless.
This project uses features(t1..tN) → label(tN+1), which is the correct formulation.

---

## Architecture
```
Input: temporal series of graphs per patient
         ↓
Feature Selection: ~20-30 features (mRMR + Stability Selection)
         ↓
Graph Construction per timepoint T:
  - Nodes: Necrosis, Contrast-enhancing, Edema (normalised PyRadiomics features)
  - Edges: triangular topology (3 bidirectional edges)
  - Edge features: volumetric ratio + delta_t_weeks
  - Delta-graph: (feature_T - feature_T-1) / delta_weeks
         ↓
Phase 2 — Baseline: LR → LightGBM → LSTM
         ↓
Phase 3 — Temporal GNN: GATv2Conv message passing + Temporal Attention
         ↓
Phase 4 — Uncertainty: Conformal Prediction (distribution-free)
         ↓
Output: RANO class + prediction set with calibrated confidence
```

---

## Critical Technical Decisions
1. Z-score normalization ONLY on training data of each fold (never fit on entire dataset)
2. StratifiedGroupKFold — group=patient, stratum=RANO class — never mix scans of same patient
3. Primary segmentation source: DeepBraTumIA (3 nodes) — more coverage, biologically richer
   HD-GLIO-AUTO kept as reference for ablation (2-node graph vs 3-node graph)
4. Three-node graph: Necrosis ↔ Contrast-enhancing ↔ Edema (triangular topology)
   Extensible: adding a fourth node requires zero changes to downstream architecture
5. Delta-graph normalized by temporal interval: delta_feature / delta_weeks
6. Metrics: macro F1, MCC, AUC per class — NEVER accuracy (imbalanced classes)
7. n_effective = 212 (DeepBraTumIA) — both t AND t+1 must have complete features
8. Feature selection: mRMR + Stability Selection (τ=0.7, B=100 bootstrap)
   - Formula: max I(xi; y) - (1/|S|) * sum I(xi; xj∈S)
   - MI estimation: Kraskov estimator (continuous variables, small n)
9. Discarded techniques: MINE, Direct Total Correlation, t-SNE, PCA, UMAP as model input
10. UMAP allowed ONLY for exploratory visualization in the paper
11. Baseline hierarchy — mandatory for paper credibility:
    Baseline 1: Logistic Regression
    Baseline 2: XGBoost / LightGBM
    Baseline 3: LSTM on flat vectors
    Ablation:   2-node graph (HD-GLIO-AUTO) vs 3-node graph (DeepBraTumIA)
    Model:      Temporal GNN (3-node)
12. History length bias — add as explicit features:
    - absolute time from diagnosis to timepoint T
    - number of previous scans for that patient

---

## Preprocessing Pipeline (summary — see docs/PHASE_0.md for detail)
1. Pivot DeepBraTumIA CSV: 7188 rows → 1 row per (patient, timepoint)
   Column naming: {label}_{sequence}_{feature_name}
2. Merge with RANO labels on (Patient, Timepoint); deduplicate Patient-042 week-010
3. Apply label shift: assign label_t+1 as target, drop last timepoint per patient
4. Add temporal features: delta_t_weeks, time_from_diagnosis, scan_index
5. Handle missing: document partial-NaN scans; drop scans with all-NaN for all labels
6. Compute delta features: Δf = (f_t - f_{t-1}) / delta_weeks; Δf=0 for baseline scan
7. Normalization: inside cross-validation only (StandardScaler fit on train fold)

---

## Repository Structure
```
gbm-longitudinal-toolkit/
├── .github/workflows/ci.yml
├── data/raw/lumiere/           # original CSVs — never modified, DVC tracked
├── data/processed/             # pipeline output — DVC versioned
├── src/
│   ├── audit/                  # lumiere_audit.py, validate_dataset.py
│   ├── preprocessing/          # build_dataset.py, normalizer.py
│   ├── graphs/                 # graph_builder.py, delta_graph.py
│   ├── models/                 # lstm_baseline.py, gnn.py, temporal_attention.py
│   ├── training/               # trainer.py, cross_validation.py, metrics.py
│   └── uncertainty/            # conformal.py
├── tests/
├── experiments/                # MLflow runs
├── notebooks/                  # EDA — not production
├── configs/                    # model parameter YAMLs
├── docs/                       # PHASE_0.md ... PHASE_5.md
├── FUTURE.md
├── CONTEXT.md
├── pyproject.toml
└── dvc.yaml
```

---

## Roadmap
- Phase 0 — Data Foundation: audit ✅ | preprocessing → | validation →
- Phase 1 — Graph Construction (3 weeks): GraphBuilder (3-node), delta-graph
- Phase 2 — Baseline (2 weeks): LR → LightGBM → LSTM
- Phase 3 — Temporal GNN (4 weeks): TumorGraphNet (3-node) + TemporalAttention
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
    Verification: compare against delta_t-only baseline.
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
"Open-source pipeline for longitudinal radiomics analysis in GBM with temporal
graph modelling and distribution-free uncertainty quantification"

Notes:
- Verify with systematic literature review before submission.
- The real value is the replicable longitudinal pipeline, not the specific model.
- Additional methodological contribution: Temporal MI Stability (consistency of ranking).
- 2-node vs 3-node graph ablation is itself a contribution (graph topology sensitivity).

---

## Software Engineering Principles

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
10. **No Premature Optimisation** — correct and readable first. On n=212,
    readability always wins.
11. **Reproducibility** — fix random seeds explicitly (Python, NumPy, PyTorch).
    Dataset versioned via DVC. All hyperparameters in YAML configs, never
    hardcoded. MLflow logs config alongside metrics.

---

## Important Notes
- New ideas → FUTURE.md (e.g. Neural ODE, raw MRI volumes, multi-institutional)
- GNN with 3 nodes may not beat baseline — limitation, not failure
- Generalizable framework is the main value, not the model on LUMIERE
- Seek clinical collaboration AFTER V1 for prospective validation
- Always distinguish: theoretically motivated choice vs pragmatically necessary choice
