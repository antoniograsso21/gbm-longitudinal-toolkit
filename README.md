# gbm-longitudinal-toolkit

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Open-source pipeline for **longitudinal radiomics in glioblastoma (GBM)** with
explicit temporal-bias control, strong baselines, temporal GNNs, and
distribution-free uncertainty quantification.

---

## What this project is

**gbm-longitudinal-toolkit** is a research-grade, reproducible pipeline that:
- **Formulates the task correctly**: predict **RANO at the next visit**  
  (features at time \(t\) → label at time \(t+1\); last timepoint per patient is never used).
- **Controls temporal bias and leakage**:  
  clinical workflow leakage (irregular follow-up intervals) is quantified via
  temporal-only baselines and SHAP analyses.
- **Provides a full modelling ladder**:
  - Logistic Regression and LightGBM baselines,
  - LSTM on flat temporal vectors,
  - 3-node tumor **graph + temporal GNN** (exploratory component, not the core claim).
- **Adds calibrated uncertainty + interpretability**:
  - Conformal Prediction (RAPS) for distribution-free prediction sets,
  - SHAP for gradient boosting, Integrated Gradients + attention for the GNN,
  - a clinical-style summary per prediction (class, set, key features, key timepoint).

The first case study is the open **LUMIERE** GBM dataset; the architecture is
designed to be generalisable to other longitudinal radiomics cohorts in later steps.

---

## Dataset: LUMIERE (current focus)

- **91 patients**, **638** study dates, **2487** MRI scans.
- Sequences: CT1, T1, T2, FLAIR (skull-stripped, co-registered).
- Segmentations:
  - **DeepBraTumIA** (primary, 3 labels: Necrosis, Contrast-enhancing, Edema),
  - **HD-GLIO-AUTO** (reference, 2 labels: Non-enhancing, Contrast-enhancing).
- Radiomics: PyRadiomics v3.0.1, 107 features per label per sequence.
- RANO labels: 399 valid timepoints, 81 patients with ≥1 label.
- Effective supervised sample size after audit + preprocessing  
  (label shift, any-NaN drop, temporal pairing):
  - **231 paired examples**, 64 patients (DeepBraTumIA, primary).

This repository **does not ship LUMIERE**. Download the CSVs from their original
Figshare source under the non-commercial license and place them under
`data/raw/lumiere/` as described in `CONTEXT.md`.

---

## Pipeline at a glance

The end-to-end pipeline is split into documented steps (`docs/STEP_*.md`):

- **Step 0 – Audit**: structural checks on all raw CSVs, dataset_stats.json.
- **Step 1 – Preprocessing + Validation**:  
  build `dataset_paired.parquet` (231×2576, 1 row = (patient, timepoint)),  
  run strict validation (n_effective, leakage checks, NaN/inf, monotonic time).
- **Step 2 – Feature Engineering**:  
  9 label-free derived features added → `dataset_engineered.parquet` (231×2585).  
  Cross-compartment ratios (CE_NC_ratio, ED_CE_ratio, CE_fraction, total_tumor_volume),  
  nadir-based features (CE_vs_nadir, weeks_since_nadir, is_nadir_scan),  
  delta of derived features (delta_CE_NC_ratio, delta_CE_vs_nadir).  
  EDA: correlation maps, shape consistency check, delta distributions,  
  temporal distributions, UMAP (visualisation only), temporal autocorrelation.
- **Step 3 – Baseline models**:  
  StratifiedGroupKFold, **feature selection inside CV** (mRMR + Stability Selection),
  LR, LightGBM+SHAP, LSTM, temporal-feature ablations to quantify leakage.
- **Step 4 – Graphs + Temporal GNN**:  
  3-node tumor graphs (NC ↔ CE ↔ ED), GATv2Conv + temporal attention,  
  2-node vs 3-node ablations (HD-GLIO-AUTO vs DeepBraTumIA).
- **Step 5 – Interpretability**:  
  SHAP, attention weights, Integrated Gradients, clinical summaries.
- **Step 6 – Uncertainty**:  
  Conformal Prediction (RAPS), coverage and set-size analysis.
- **Step 7 – Framework**:  
  configs, CLI entry points, DVC pipeline, new-dataset tutorial.
- **Step 8 – Paper**: bioRxiv preprint and journal submission.

The high-level design and scientific assumptions are captured in `CONTEXT.md`.

---

## Repository structure

```text
gbm-longitudinal-toolkit/
├── data/
│   ├── raw/lumiere/           # original CSVs — never modified, DVC-tracked
│   └── processed/             # audit, parquet, validation reports
├── src/
│   ├── utils/                 # LUMIERE-specific I/O and parsing (lumiere_io.py)
│   ├── audit/                 # lumiere_audit.py, dataset_validator.py
│   ├── preprocessing/         # dataset_builder.py, features_builder.py
│   ├── graphs/                # graphs_builder.py, temporal sequences
│   ├── models/                # logistic_baseline.py, gbm_baseline.py, lstm_baseline.py, gnn.py
│   ├── training/              # CV loops, metrics, MLflow integration
│   ├── interpretability/      # SHAP, attention, Integrated Gradients, clinical_summary.py
│   └── uncertainty/           # conformal prediction utilities
├── configs/                   # YAML configs (models, graphs, conformal, features_builder.yaml)
├── docs/                      # STEP_0.md … STEP_8.md
├── experiments/               # optional MLflow runs
├── notebooks/                 # exploratory analyses (not production)
├── CONTEXT.md                 # project context and constraints
├── FUTURE.md                  # ideas beyond the current scope
├── dvc.yaml                   # data / experiment pipeline
└── pyproject.toml
```

---

## Installation

```bash
git clone https://github.com/antoniograsso21/gbm-longitudinal-toolkit.git
cd gbm-longitudinal-toolkit

# Python 3.12+ recommended
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e .
```

Core runtime dependencies are intentionally minimal in `pyproject.toml`.  
Training stack (PyTorch, PyTorch Geometric, LightGBM, MLflow, etc.) is expected
to be installed via extras or environment files as the implementation matures.

---

## Quick start (current status)

The **audit, preprocessing, and feature engineering pipeline (Steps 0–2)** is
implemented and passing validation.

1. **Prepare the data**
   - Download LUMIERE CSVs and place them under `data/raw/lumiere/`.
   - See `CONTEXT.md` for the exact file list.

2. **Run audit and preprocessing**

   ```bash
   uv run -m src.audit.lumiere_audit
   uv run -m src.preprocessing.dataset_builder
   uv run -m src.validation.dataset_validator
   ```

   This produces:
   - `data/processed/dataset_stats.json`
   - `data/processed/dataset_paired.parquet`
   - `data/processed/dataset_builder_report.json`
   - `data/processed/dataset_validator_report.json`

3. **Run feature engineering**

   ```bash
   uv run -m src.preprocessing.features_builder
   ```

   This produces:
   - `data/processed/dataset_engineered.parquet` (231×2585, +9 derived features)
   - `data/processed/features_builder_report.json`

4. **Explore the design**
   - Read `CONTEXT.md` for the scientific rationale.
   - Read `docs/STEP_3.md`–`STEP_6.md` for the planned modelling, GNN, and UQ pipeline.

As later steps are implemented, the goal is to expose simple CLI entry points
(`run_baselines`, `run_gnn`, `run_conformal`, `run_explanations`) wired into `dvc.yaml`.

---

## Contributing and citing

Contributions are welcome, especially around:
- extending the baseline and GNN implementations,
- improving temporal bias diagnostics,
- adding support for additional longitudinal radiomics datasets,
- strengthening uncertainty quantification and interpretability tooling.

If you build on this work, you can cite:

> Grasso A. *gbm-longitudinal-toolkit: open-source framework for longitudinal radiomics in glioblastoma.* GitHub repository, 2026–.

This project is released under the **MIT License** (see `LICENSE`).