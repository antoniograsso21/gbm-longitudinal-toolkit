# gbm-longitudinal-toolkit

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Longitudinal radiomics framework for glioblastoma (GBM) treatment response prediction,  
with temporal graph neural networks and distribution-free uncertainty quantification.

---

## Project goal

**gbm-longitudinal-toolkit** aims to be a **reusable, open-source pipeline** for longitudinal radiomics in GBM and other cancers, with:
- a **label‑shift aware formulation** (predict RANO at the *next* visit, not the current one),
- a **3‑node tumor graph** (Necrosis, Contrast‑enhancing, Edema) per timepoint,
- strong **non-graph baselines** (Logistic Regression, gradient boosting, LSTM),
- a **temporal GNN** for graph‑structured histories,
- and **Conformal Prediction** for distribution‑free uncertainty estimates.

The first case study is the **LUMIERE** dataset; the design is general and intended to support other longitudinal radiomics cohorts.

---

## Dataset: LUMIERE (current case study)

The toolkit is currently engineered around the open‑access **LUMIERE** GBM dataset:
- **91 patients**, **638** study dates, **2487** MRI scans
- Sequences: CT1, T1, T2, FLAIR (skull‑stripped, co‑registered)
- Segmentations:
  - **DeepBraTumIA** (primary, 3 labels: Necrosis, Contrast‑enhancing, Edema)
  - **HD‑GLIO‑AUTO** (reference, 2 labels: Non‑enhancing, Contrast‑enhancing)
- Radiomics: PyRadiomics v3.0.1, **107 features per label per sequence**
- RANO labels: 399 valid timepoints, 81 patients with ≥1 label
- Effective supervised sample size (after label shift and QC):
  - **212 paired examples**, 57 patients (DeepBraTumIA, primary)

The repository does **not** redistribute LUMIERE; you must download it separately under its original license and place the CSVs under `data/raw/lumiere/` as documented in `CONTEXT.md`.

---

## What this toolkit provides

- **End‑to‑end longitudinal pipeline**
  - From raw radiomics CSVs → clean paired dataset (`features_t`, `label_t+1`)
  - Explicit control of **label shift**, temporal leakage, and small‑n effects

- **Graph construction**
  - 3‑node triangular tumor graphs per timepoint (Necrosis, Contrast‑enhancing, Edema)
  - Edge features: volumetric ratios and Δt (weeks between scans)
  - Optional 2‑node graphs (HD‑GLIO‑AUTO) for topology ablations

- **Baselines and temporal models**
  - Cross‑sectional Logistic Regression
  - Gradient boosting (e.g. LightGBM)
  - LSTM on flat temporal vectors (no graph)
  - Temporal GNN with GATv2‑based message passing + temporal attention

- **Uncertainty quantification**
  - Conformal Prediction (RAPS) on top of the GNN and baselines
  - User‑selectable coverage levels (e.g. 90%, 95%) with guaranteed coverage

- **Reproducibility & engineering**
  - Python ≥ 3.12, type‑annotated code, Ruff + mypy
  - Data versioning with **DVC**
  - Configuration‑driven experiments (YAML configs in `configs/`)
  - CI‑friendly design with synthetic‑data tests

For a strategic overview see `CONTEXT.md`. Detailed implementation plans live in `docs/PHASE_0.md` … `docs/PHASE_5.md`.

---

## Repository structure (target)

The project is organised as follows (some components are still under active development):

```text
gbm-longitudinal-toolkit/
├── data/
│   ├── raw/lumiere/          # original CSVs — never modified, DVC-tracked
│   └── processed/            # pipeline outputs (paired dataset, graphs, reports)
├── src/
│   ├── audit/                # dataset audit & validation (e.g. lumiere_audit.py)
│   ├── preprocessing/        # build_dataset.py (label shift, Δ features, etc.)
│   ├── graphs/               # graph_builder.py, delta_graph.py, temporal_sequence.py
│   ├── models/               # baselines, GNN, temporal attention
│   ├── training/             # CV loops, metrics, MLflow integration
│   └── uncertainty/          # conformal prediction utilities
├── configs/                  # YAML configs for models, graphs, feature selection, CP
├── docs/                     # PHASE_*.md plans, tutorials
├── notebooks/                # exploratory analyses (not production)
├── experiments/              # MLflow runs (optional)
├── CONTEXT.md                # high-level design & constraints
├── FUTURE.md                 # ideas beyond the current scope
├── dvc.yaml                  # pipeline stages
└── pyproject.toml
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/antoniograsso21/gbm-longitudinal-toolkit.git
cd gbm-longitudinal-toolkit

# (Recommended) create and activate a Python 3.12+ virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package (core dependencies only; ML stack will be expanded)
pip install -e .
```

Additional dependencies for training models (PyTorch, PyTorch Geometric, LightGBM, MLflow, etc.) will be specified in dedicated extras / docs as the implementation matures.

---

## Quick start (current status)

At the moment, the most mature component is the **Phase 0 audit** of the LUMIERE dataset.

1. **Prepare data**
   - Download the LUMIERE CSVs under their original license.
   - Place them under `data/raw/lumiere/` following the layout described in `CONTEXT.md`.

2. **Run the dataset audit**

   ```bash
   python -m src.audit.lumiere_audit
   ```

   This produces `data/processed/dataset_stats.json` and `data/processed/dataset_stats.txt`, summarising scan coverage, RANO labels, NaNs, temporal spacing, and the effective sample size.

3. **Follow the phase plans**
   - `docs/PHASE_0.md`: build the clean paired dataset and validation checks.
   - `docs/PHASE_1.md`: graph construction and feature selection.
   - `docs/PHASE_2.md`: baselines (LR, LightGBM, LSTM) and delta‑t ablations.
   - `docs/PHASE_3.md`: temporal GNN and ablation study.
   - `docs/PHASE_4.md`: conformal prediction and uncertainty metrics.
   - `docs/PHASE_5.md`: framework packaging, CLI, tutorial, and paper.

As the implementation progresses, the goal is to expose high‑level CLI entry points such as:

```bash
python -m src.preprocessing.build_dataset
python -m src.graphs.build_graphs
python -m src.training.run_baselines --config configs/gbm_baseline.yaml
python -m src.training.run_gnn --config configs/gnn.yaml
python -m src.uncertainty.run_conformal --config configs/conformal.yaml
```

---

## Roadmap

- **Phase 0 — Data foundation**: audit ✅, preprocessing & validation in progress.
- **Phase 1 — Graph construction**: graph builder, delta graphs, temporal sequences.
- **Phase 2 — Baseline models**: LR → LightGBM → LSTM with rigorous CV.
- **Phase 3 — Temporal GNN**: 3‑node tumor graph + temporal attention + ablations.
- **Phase 4 — Uncertainty**: Conformal Prediction with RAPS, coverage analysis.
- **Phase 5 — Framework + paper**: configs, CLI, tutorial, preprint, GitHub release.

See the `docs/PHASE_*.md` files for precise definitions of done for each phase.

---

## Contributing

Contributions are welcome, especially around:
- improving the preprocessing and validation pipeline for LUMIERE,
- extending support to other longitudinal radiomics datasets,
- strengthening baselines and ablation studies,
- integrating additional uncertainty quantification methods.

Please open an issue to discuss major changes before submitting a pull request.

---

## Citation

An accompanying preprint is planned. Until then, you can cite the repository as:

> Grasso A. *gbm-longitudinal-toolkit: open-source framework for longitudinal radiomics in glioblastoma.* GitHub repository, 2026–.

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.