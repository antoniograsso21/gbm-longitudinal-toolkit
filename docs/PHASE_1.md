# Phase 1 — Graph Construction

## Objective
Transform the flat paired dataset into a sequence of graphs per patient,
one graph per timepoint, ready for the GNN in Phase 3.

**Input**: `data/processed/dataset_paired.parquet`
**Output**: `data/processed/graphs/{patient_id}.pt`

---

## Graph Design

### Three-node topology
Each graph represents one tumor scan with three nodes:

```
Necrosis ←→ Contrast-enhancing
    ↖              ↗
         Edema
```

Six directed edges (3 bidirectional pairs). Each edge carries:
- `volumetric_ratio`: volume of source node / volume of target node
- `delta_t_weeks`: temporal interval (monitored for leakage — see Phase 2 ablation)

This design is directly motivated by the DeepBraTumIA segmentation labels
and removes the main limitation of the original 2-node design. The triangular
topology is fully extensible: adding a fourth node requires zero changes to
the downstream GNN architecture.

### Node features
For each node (label) at timepoint T:
```
x = [selected radiomic features (~20-30 after Phase 1 feature selection)
     delta features (Δf normalised by delta_t_weeks)
     is_baseline_scan (binary)
     time_from_diagnosis_weeks
     scan_index]
```

### Graph-level label
```
y = RANO class at T+1  (0=Progressive, 1=Stable, 2=Response)
```

---

## Step 1.1 — GraphBuilder

**File**: `src/graphs/graph_builder.py`

Builds a single `torch_geometric.data.Data` object from one row.

```python
@dataclass
class GraphConfig:
    node_feature_cols: dict[str, list[str]]  # {label: [feature columns]}
    edge_feature_cols: list[str]             # volumetric_ratio, delta_t_weeks
    label_col: str
    label_mapping: dict[str, int]            # {"Progressive": 0, "Stable": 1, "Response": 2}
    node_order: list[str]                    # ["Necrosis", "Contrast-enhancing", "Edema"]

def build_graph(row: pd.Series, config: GraphConfig) -> Data:
    ...
```

Node ordering is fixed by `node_order` — consistent across all graphs.
Edge index for the triangular topology:
```python
# 3 nodes: 0=Necrosis, 1=Contrast-enhancing, 2=Edema
# 6 directed edges (bidirectional)
edge_index = torch.tensor([
    [0, 1, 1, 0, 0, 2, 2, 0, 1, 2, 2, 1],
    [1, 0, 0, 1, 2, 0, 0, 2, 2, 1, 1, 2],
], dtype=torch.long)
```

---

## Step 1.2 — DeltaGraphBuilder

**File**: `src/graphs/delta_graph.py`

Reads delta columns already computed in Phase 0 (sub-step 7) and assigns
them as a secondary node feature tensor. Keeps static and dynamic features
separate for interpretability.

For baseline scans (`is_baseline_scan=True`): delta features are zero vectors.

---

## Step 1.3 — TemporalSequence

**File**: `src/graphs/temporal_sequence.py`

```python
@dataclass
class PatientGraphSequence:
    patient_id: str
    graphs: list[Data]       # ordered chronologically
    n_timepoints: int
    label_sequence: list[int]
```

Saved as `data/processed/graphs/{patient_id}.pt` via `torch.save`.

---

## Step 1.4 — Feature Selection (mRMR + Stability Selection)

Reduce from ~1284 columns (3 labels × 4 sequences × 107 features) to ~20-30.
Executed on the training fold only (inside CV).

**Algorithm**: mRMR
```
max I(xi; y) - (1/|S|) * sum I(xi; xj ∈ S)
```

**MI estimator**: Kraskov k-NN (appropriate for continuous variables, small n)
**Stability Selection**: B=100 bootstrap replicates, keep features with P(xi selected) > τ=0.7

Output: `configs/selected_features.yaml` — versioned and logged to MLflow.

**2-node vs 3-node ablation note**: feature selection is run separately for
both DeepBraTumIA (3-node) and HD-GLIO-AUTO (2-node) to support the graph
topology ablation in Phase 3.

---

## Validation

```python
assert all(g.x.shape[0] == 3 for g in graphs)          # 3 nodes
assert all(g.edge_index.shape == (2, 12) for g in graphs)  # 6 bidirectional edges
assert all(g.edge_attr.shape == (12, 2) for g in graphs)   # 2 edge features
assert all(g.y.shape == (1,) for g in graphs)
```

---

## Definition of Done for Phase 1

- [ ] GraphBuilder implemented and unit tested with synthetic data
- [ ] 3-node topology verified (Necrosis, Contrast-enhancing, Edema)
- [ ] Edge index correct for triangular bidirectional topology
- [ ] DeltaGraphBuilder implemented and unit tested
- [ ] TemporalSequence serialisation working (save/load roundtrip test)
- [ ] Feature selection pipeline implemented (mRMR + Stability Selection)
- [ ] `configs/graph_config.yaml` defined
- [ ] All patient graphs saved and DVC tracked
- [ ] HD-GLIO-AUTO graphs also built (2-node) for ablation in Phase 3
