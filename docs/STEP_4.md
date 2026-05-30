# Step 4 — Graph Construction + Temporal GNN ⏳

## Objective
Build patient graph sequences from the paired dataset and train the temporal GNN.

**Input**:
- `data/processed/preprocessing/dataset_engineered.parquet`
- `configs/selected_features.yaml` (from Step 3)

**Output**:
- `data/processed/graphs/3node/{patient_id}.pt`
- `data/processed/gnn/gnn_{ablation}_{topology}_results.json`
- MLflow experiment `gnn/`

---

## 4.1 — Graph Construction

### Graph topology
3 nodes: NC (Necrosis=0), CE (Contrast-enhancing=1), ED (Edema=2)
Triangular bidirectional topology — 6 undirected edges × 2 directions = 12 directed edges.

```
Necrosis ←→ Contrast-enhancing
    ↖              ↗
         Edema
```

Node feature ownership is prefix-based: selected `NC_*`, `CE_*`, and `ED_*`
radiomic/delta columns are assigned only to their corresponding node, with
shorter node rows zero-padded to a common width for PyG batching. In the current
Step 3 selected feature set, ED has no majority-vote radiomic features, so the
ED node is retained topologically but carries only scalar context plus padding.
This must be declared when interpreting any 3-node GNN result.

### Node features per node
```
[selected radiomic features from selected_features.yaml, prefix-filtered per node
 anchored delta features (delta_f for each selected radiomic f, if present)
 is_baseline_scan
 time_from_diagnosis_weeks
 scan_index]
```

### Edge features per directed edge
```
[volumetric_ratio: volume(src) / volume(tgt)
 interval_weeks]
```

### Graph label
```
y = target_encoded (RANO class at t+1)
```

**File**: `src/graphs/graph_builder.py`
**Key classes**: `GraphConfig`, `PatientGraphSequence`
**Validation**: `graphs_validator.py — asserts` — asserts shape invariants on every graph

HD-GLIO-AUTO 2-node graphs remain a deferred A6 ablation. The current engineered
parquet is DeepBraTumIA-only and has no `NE_*` columns; the builder now fails
explicitly rather than creating a misleading scalar-only NE node.

**Current dry-run status (2026-05-30)**:
- 3-node graph builder dry-run passes on DeepBraTumIA: 231 graphs, 64 patients,
  sequence length min=1, max=14, mean=3.6, node feature dim=67, edge dim=2.
- 3-node graphs have been saved under `data/processed/graphs/3node/` and validated
  by `src.validation.graphs_validator` with zero FAIL results.
- Current node widths before padding: NC=19, CE=67, ED=3. ED has no selected
  radiomic/delta features in the current `selected_features.yaml`.
- 2-node topology is intentionally blocked until an HD-GLIO-AUTO engineered parquet exists.

Training note: `run_gnn.py` re-collates node features from fold-normalised
`dataset_engineered.parquet` using the fold-specific MI selection, and reuses the
saved `.pt` files for graph topology, edge attributes, labels, and sequence metadata.
This preserves the Step 3 rule that feature selection and scaling live inside CV.

---

## 4.2 — Temporal GNN Architecture

```
Patient graph sequence: [G_1, G_2, ..., G_T]
        ↓
[1] GNN Message Passing (per timepoint)
    GATv2Conv — learns attention weights per edge
    1 layer preferred; 2 only if ablation A1 shows meaningful gain
        ↓
[2] Temporal Attention
    Sinusoidal encoding with actual interval_weeks as position (not step index)
    Multi-head attention over graph embeddings [h_1, ..., h_T]
        ↓
[3] Classifier head
    Linear(hidden, 3) → Softmax
        ↓
Output: P(Progressive), P(Stable), P(Response) at T+1
```

**Files**:
- `src/models/gnn.py` — TumorGraphNet (GATv2Conv)
- `src/models/temporal_attention.py` — continuous-time encoding
- `src/models/tumor_gnn.py` — full TumorTemporalGNN

**Loss**: CrossEntropyLoss with class weights (76%/11%/13% imbalance)
**Optimiser**: AdamW | **Scheduler**: ReduceLROnPlateau | **Patience**: 20

```yaml
# configs/gnn.yaml
hidden: [32, 64]
heads: [1, 2]
dropout: [0.2, 0.3]
learning_rate: [1e-3, 5e-4]
batch_size: 8
max_epochs: 200
patience: 20
```

---

## 4.3 — Ablation Study

| Ablation | What is removed | Question answered |
|----------|----------------|-------------------|
| A1: Cross-sectional GNN | Temporal attention | Does history help? |
| A2: No graph (= LSTM from Step 3) | GNN message passing | Does graph structure help? |
| A3: No delta features | Δf columns | Does rate of change help? |
| A4: No Δt encoding | Temporal positional encoding | Does irregular time matter? |
| A5: Temporal only (= ablation B from Step 3) | All radiomic features | Leakage quantification |
| A6: 2-node GNN (HD-GLIO-AUTO) | Edema node | Does edema add signal? |

A2 = LSTM from Step 3 — already computed.
A5 = temporal-only ablation B from Step 3 — already computed.
A3 is currently blocked until a no-delta graph/feature-collation path is added.
A6 is currently blocked until HD-GLIO-AUTO preprocessing/feature engineering produces
an engineered parquet with `NE_*` columns.

---

## Current Implementation Status (2026-05-30)

Implemented:
- `src/graphs/graph_builder.py` with 3-node graph construction and fail-fast structural validation
- `src/validation/graphs_validator.py`
- `src/models/gnn.py` (`TumorGraphNet`, GATv2Conv)
- `src/models/temporal_attention.py` (continuous-time encoding + temporal attention)
- `src/models/tumor_gnn.py` (`TumorTemporalGNN`)
- `src/training/run_gnn.py` with CV scaffold, fold-wise feature selection reuse,
  class-weighted loss, AdamW, ReduceLROnPlateau, early stopping, and MLflow logging
- `configs/gnn.yaml`
- Fast GNN smoke run completed for `full` / `3node`:
  `data/processed/gnn/gnn_full_3node_results.json`

Known constraints:
- Full hyperparameter grid search is not implemented yet; `run_gnn.py` currently uses
  the first value from each list in `configs/gnn.yaml`.
- Training uses whole-fold patient tensors rather than mini-batches. This is acceptable
  for n=231 as a first correctness pass, but `batch_size` is not yet operational.
- A3 and A6 are intentionally blocked rather than silently producing invalid ablations.
- GNN must be compared against LightGBM A, B, and D from Step 3; beating LR alone is insufficient.

---

## Definition of Done

- [x] `src/graphs/graph_builder.py` implemented for 3-node DeepBraTumIA graphs
- [x] 3-node graph builder dry-run passes structural assertions on 231 graphs / 64 patients
- [x] `src/validation/graphs_validator.py` implemented
- [x] `TumorGraphNet`, `TemporalAttentionEncoder`, and `TumorTemporalGNN` implemented
- [x] CV training runner scaffold implemented in `src/training/run_gnn.py`
- [x] Class-weighted loss, AdamW, ReduceLROnPlateau, early stopping, and random seeds wired
- [x] `configs/gnn.yaml` present
- [x] 3-node graphs saved to `data/processed/graphs/3node/{patient_id}.pt`
- [x] `graphs_validator.py` run successfully on saved 3-node graph artifacts
- [x] GNN fast smoke run completed and JSON report saved
- [ ] Full 5-fold GNN run completed for full, A1, and A4
- [ ] A3 no-delta ablation implemented and run
- [ ] HD-GLIO-AUTO 2-node engineered parquet produced; A6 graph builder/run implemented and validated
- [ ] Comparison table from Step 3 completed with GNN results
- [ ] Formal unit tests added for graph builder, temporal padding/collation, and model forward pass
