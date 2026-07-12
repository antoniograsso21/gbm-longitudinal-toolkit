# Step 7 — Framework ⏳

## Objective
Generalise the LUMIERE-specific pipeline into a reusable open-source framework.
Add CLI entry points, DVC pipeline, and a new-dataset tutorial.

---

## Tasks

### 7.1 — Config files (all in `configs/`)
```
dataset_config.yaml      — dataset-specific params (generalised in Step 7)
graph_config.yaml        — node order, edge features, label mapping
features_builder.yaml    — feature engineering params (epsilon, volume cols)
feature_selector.yaml    — mRMR params, stability threshold
logistic_baseline.yaml
lgbm_baseline.yaml
lstm_baseline.yaml
gnn.yaml
conformal.yaml
```

### 7.2 — CLI Entry Points
```bash
uv run -m src.preprocessing.dataset_builder
uv run -m src.graphs.graphs_builder
uv run -m src.training.run_baselines --config configs/lgbm_baseline.yaml
uv run -m src.training.run_gnn --config configs/gnn.yaml
uv run -m src.uncertainty.run_conformal --config configs/conformal.yaml
```

### 7.3 — DVC Pipeline
```yaml
# dvc.yaml
stages:
  audit: ...
  preprocess: ...
  validate: ...
  build_graphs: ...
  run_baselines: ...
  run_gnn: ...
  run_conformal: ...
```
`dvc repro` runs full pipeline from raw CSVs to final results.

### 7.4 — New Dataset Tutorial
`docs/NEW_DATASET_TUTORIAL.md`:
1. Required CSV format
2. Timepoint parser (replacing `parse_week` for non-LUMIERE formats)
3. Required segmentation labels (minimum: 2 compartments)
4. Config files to modify
5. How to interpret audit output and n_effective

### 7.5 — GitHub Release
- README with installation, quick start, citation
- `CITATION.cff`
- Release tag matching bioRxiv version (e.g. `v1.0.0-preprint`)

---

## Definition of Done

- [ ] All configs in YAML, no hardcoded hyperparameters in source
- [ ] All CLI entry points working
- [ ] `dvc repro` runs from raw CSVs to final results
- [ ] New dataset tutorial written and tested on synthetic CSV
- [ ] README complete
- [ ] `CITATION.cff` added
- [ ] GitHub release tagged