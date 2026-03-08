# Phase 5 — Framework + Paper

## Objective
Package the pipeline as a reusable open-source framework, write the preprint,
and publish to bioRxiv + GitHub.

---

## Two Parallel Tracks

Phase 5 runs two tracks simultaneously:

```
Track A — Framework          Track B — Paper
──────────────────           ──────────────────
Parameterisation             Methods section
Documentation                Results section
Tutorial notebook            Discussion section
CLI entry point              Abstract + Introduction
GitHub release               bioRxiv submission
```

Both tracks must be complete before submission.

---

## Track A — Framework

### A1. Parameterisation
Every hardcoded value must live in a YAML config by the end of this phase.

Config files (all in `configs/`):
```
graph_config.yaml         — node/edge feature columns, label mapping
feature_selector.yaml     — mRMR parameters, stability threshold τ
logistic_baseline.yaml    — LR hyperparameter grid
gbm_baseline.yaml         — LightGBM hyperparameter grid
lstm_baseline.yaml        — LSTM architecture and training params
gnn.yaml                  — GNN architecture and training params
conformal.yaml            — alpha levels, RAPS lambda
```

### A2. CLI Entry Points
Each phase should be runnable with a single command:

```bash
python -m src.preprocessing.build_dataset
python -m src.graphs.build_graphs
python -m src.training.run_baselines --config configs/gbm_baseline.yaml
python -m src.training.run_gnn --config configs/gnn.yaml
python -m src.uncertainty.run_conformal --config configs/conformal.yaml
```

Each entry point logs to MLflow and saves outputs under `data/processed/`.

### A3. New Dataset Tutorial
`docs/NEW_DATASET_TUTORIAL.md` — how to use this framework on a different
longitudinal radiomics dataset. This is the core reusability claim of the paper.

Minimum sections:
1. Required CSV format (radiomic features, labels, patient IDs)
2. Config files to modify
3. What to expect from the audit and preprocessing steps
4. How to interpret the results

### A4. DVC Pipeline
`dvc.yaml` defines the full reproducible pipeline:

```yaml
stages:
  preprocess:
    cmd: python -m src.preprocessing.build_dataset
    deps: [data/raw/lumiere/]
    outs: [data/processed/dataset_paired.parquet]

  build_graphs:
    cmd: python -m src.graphs.build_graphs
    deps: [data/processed/dataset_paired.parquet]
    outs: [data/processed/graphs/]

  run_gnn:
    cmd: python -m src.training.run_gnn --config configs/gnn.yaml
    deps: [data/processed/graphs/, configs/gnn.yaml]
    outs: [experiments/gnn_results.json]
```

`dvc repro` runs the full pipeline from raw CSVs to final results.
This is what makes the paper reproducible for external reviewers.

### A5. GitHub Release
- Clean README with installation instructions and quick start
- `CITATION.cff` file for academic citation
- GitHub release tag matching the bioRxiv version (e.g. `v1.0.0-preprint`)
- All DVC-tracked data accessible via `dvc pull`

---

## Track B — Paper

### Target venue
bioRxiv preprint (immediate), then submission to a journal (e.g. Medical Image
Analysis, NeuroImage, or Nature Scientific Reports).

### Paper structure

**Abstract** (250 words max)
- Clinical problem: RANO assessment is subjective and delayed
- What we built: open-source longitudinal radiomics pipeline with GNN + UQ
- Key result: macro F1, comparison to baselines, CP coverage
- Claim: reusable framework for any longitudinal radiomics dataset

**1. Introduction**
- GBM clinical context (brief)
- Limitations of current radiomics papers: cross-sectional, no UQ, not reusable
- What this paper contributes: longitudinal, graph-based, calibrated, open-source
- Literature review: verify claim against existing work before writing

**2. Methods**
- Dataset: LUMIERE description, n_effective, label shift, leakage controls
- Preprocessing pipeline (sub-steps 1-6 from Phase 0)
- Graph construction (2-node design rationale, edge features)
- Feature selection (mRMR + Stability Selection — declare as methodological contribution)
- Baseline hierarchy (LR, LightGBM, LSTM) — why each was included
- Temporal GNN architecture (TumorGraphNet + TemporalAttention)
- Conformal Prediction setup (RAPS, calibration split, coverage levels)
- Cross-validation (StratifiedGroupKFold, normalisation inside fold)

**3. Results**
- Comparison table (all models, all metrics)
- Ablation study results (A1-A5)
- Conformal Prediction coverage and set sizes
- Δt ablation result (leakage quantification)
- Attention weight visualisation

**4. Discussion**
- Does GNN beat XGBoost? Honest interpretation either way.
- Temporal MI Stability as feature selection contribution
- Clinical relevance of prediction sets
- Limitations: 2-node graph, single institution, inter-rater variability unknown

**5. Future Work**
Reference FUTURE.md directly:
- Third node: peritumoral edema
- Raw MRI volumes (3D CNN / ViT)
- Neural ODE for irregular time
- Multi-institutional validation
- Prospective clinical integration

**Supplementary Material**
- Full feature list with stability scores
- Per-fold metrics (not just mean±std)
- CP conditional coverage per class

---

## Definition of Done for Phase 5

- [ ] All configs in YAML, no hardcoded hyperparameters in source
- [ ] All CLI entry points working
- [ ] `dvc repro` runs from raw CSVs to final results without manual steps
- [ ] New dataset tutorial written and tested on a synthetic CSV
- [ ] README complete with installation, quick start, citation
- [ ] `CITATION.cff` added
- [ ] Paper draft complete (all 5 sections)
- [ ] Literature review done — claim verified
- [ ] bioRxiv submission
- [ ] GitHub release tagged
