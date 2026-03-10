# Phase 5 — Framework + Paper

## Objective
Package the pipeline as a reusable open-source framework, write the preprint,
and publish to bioRxiv + GitHub.

---

## Two Parallel Tracks

```
Track A — Framework          Track B — Paper
──────────────────           ──────────────────
Parameterisation             Methods section
Documentation                Results section
Tutorial notebook            Discussion section
CLI entry point              Abstract + Introduction
GitHub release               bioRxiv submission
```

---

## Track A — Framework

### A1. Config files (all in `configs/`)
```
dataset_config.yaml          — LUMIERE-specific params (generalised in Phase 5)
graph_config.yaml            — node order, edge features, label mapping
log_transform_features.yaml  — 514 feature names to log1p transform (30 excluded)
feature_selector.yaml        — mRMR parameters, stability threshold τ
logistic_baseline.yaml
gbm_baseline.yaml
lstm_baseline.yaml
gnn.yaml
conformal.yaml               — alpha levels, RAPS lambda
```

### A2. CLI Entry Points
```bash
python -m src.preprocessing.build_dataset
python -m src.graphs.build_graphs
python -m src.training.run_baselines --config configs/gbm_baseline.yaml
python -m src.training.run_gnn --config configs/gnn.yaml
python -m src.uncertainty.run_conformal --config configs/conformal.yaml
python -m src.interpretability.run_explanations --config configs/gnn.yaml
```

### A3. New Dataset Tutorial
`docs/NEW_DATASET_TUTORIAL.md` — minimum sections:
1. Required CSV format (radiomic features, labels, patient IDs, timepoints)
2. Timepoint parser: replacing `parse_week` for non-LUMIERE formats
3. Required segmentation labels (minimum: 2 compartments)
4. Config files to modify (dataset_config.yaml minimum)
5. How to interpret audit output and n_effective
6. How to interpret the clinical summary output

### A4. DVC Pipeline
`dvc.yaml` stages: preprocess → build_graphs → run_gnn → run_conformal
`dvc repro` runs the full pipeline from raw CSVs to final results.

### A5. GitHub Release
- README with installation, quick start, citation
- `CITATION.cff`
- GitHub release tag matching bioRxiv version (e.g. `v1.0.0-preprint`)

---

## Track B — Paper

**Target**: bioRxiv preprint, then Medical Image Analysis / NeuroImage / Scientific Reports

### Paper structure

**Abstract** (250 words max)
- Clinical problem: RANO assessment is subjective and delayed
- What we built: longitudinal radiomics pipeline with 3-node GNN + UQ
- Key result: macro F1, comparison to baselines, CP coverage
- Claim: reusable framework for any longitudinal radiomics dataset

**1. Introduction**
- GBM clinical context
- Limitations of current radiomics papers: cross-sectional, no UQ, not reusable
- What this paper contributes: longitudinal, 3-node graph, calibrated, open-source
- Literature review: verify claim before writing

**2. Methods**
- Dataset: LUMIERE, DeepBraTumIA segmentation, n_effective=231, 64 patients
- Data quality: 39 scans absent from CSV (extraction failed), 70 all-NaN scans,
  63 scans dropped (any-NaN per label block — catches partial PyRadiomics failures)
- Patient-025 exclusion (temporal reference frame error), Patient-039 loss,
  Patient-042 duplicate resolution — all documented in Methods
- Preprocessing pipeline (sub-steps 1-7 from Phase 0)
- Graph construction: 3-node triangular topology, edge features
- Feature selection: mRMR + Stability Selection
- Baseline hierarchy and ablation study (A1-A6)
- Temporal GNN architecture
- Conformal Prediction setup
- Interpretability: Integrated Gradients + attention weights + clinical summary format
- Cross-validation: StratifiedGroupKFold, normalization inside fold

**3. Results**
- Comparison table (all models, all metrics)
- Ablation study results (A1–A6), including 2-node vs 3-node
- Conformal Prediction coverage and set sizes
- interval_weeks-only ablation (leakage quantification)
- Feature importance: SHAP (LightGBM), IG (GNN) — do they agree?
- Attention weight analysis: temporal profiles and edge weights per disease stage
- Clinical summary examples (2-3 representative patients, one per class)

**4. Discussion**
- Does GNN beat XGBoost? Honest interpretation either way.
- Does the 3rd node (Edema) add value vs 2-node? Honest result either way.
- Temporal MI Stability as feature selection contribution
- Clinical relevance of prediction sets
- Limitations: single institution, inter-rater variability unknown,
  automated segmentation errors not QC'd, n=64 patients (small Response class)

**5. Future Work** — reference FUTURE.md:
- Raw MRI volumes (3D CNN / ViT)
- Neural ODE for irregular time
- Multi-institutional validation
- Prospective clinical integration

---

## Definition of Done for Phase 5

- [ ] `src/utils/lumiere_io.py` generalised to `DatasetConfig` abstraction
- [ ] All configs in YAML, no hardcoded hyperparameters in source
- [ ] All CLI entry points working
- [ ] `dvc repro` runs from raw CSVs to final results
- [ ] New dataset tutorial written and tested on synthetic CSV
       (including timepoint parser section)
- [ ] Interpretability outputs documented in paper (SHAP, IG, attention, clinical summary)
- [ ] README complete with installation, quick start, citation
- [ ] `CITATION.cff` added
- [ ] Paper draft complete (all 5 sections)
- [ ] Literature review done
- [ ] bioRxiv submission
- [ ] GitHub release tagged
