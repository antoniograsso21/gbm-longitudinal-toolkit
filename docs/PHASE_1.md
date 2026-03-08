# Phase 1 — Graph Construction

## Objective
Transform the flat paired dataset produced in Phase 0 into a sequence of
graphs per patient, one graph per timepoint, ready for the GNN in Phase 3.

**Input**: `data/processed/dataset_paired.parquet`
**Output**: `data/processed/graphs/{patient_id}.pt` — one PyTorch Geometric
Data object per patient, containing the full temporal graph sequence.

---

## Where Phase 1 Sits in the Project

```
dataset_paired.parquet  (Phase 0 output)
        │
        ▼
[Step 1.1] GraphBuilder       src/graphs/graph_builder.py
        │   "How do I represent one timepoint as a graph?"
        │
        ▼
[Step 1.2] DeltaGraphBuilder  src/graphs/delta_graph.py
        │   "How do I encode change between consecutive timepoints?"
        │
        ▼
[Step 1.3] TemporalSequence   src/graphs/temporal_sequence.py
            "How do I assemble a patient's history into one object?"
            Output: data/processed/graphs/{patient_id}.pt
```

---

## Graph Design

### Why a graph at all?
With only 2 nodes (ET and NC) the graph is topologically trivial — it is
essentially a feature vector with an explicit relational structure.
The value of the GNN lies in the temporal inductive bias, not topological
complexity. This is a declared limitation (see CONTEXT.md, Technical Decision 6).

The graph formulation is the correct abstraction for two reasons:
1. It is extensible: adding a third node (Edema, see FUTURE.md) requires
   zero changes to the downstream model architecture.
2. PyTorch Geometric natively handles sequences of graphs with temporal edges,
   making the temporal attention module (Phase 3) cleaner to implement.

### Node structure
Each graph has exactly 2 nodes:
- Node 0: ET (Enhancing Tumor)
- Node 1: NC (Necrotic Core)

Node features for timepoint T:
```
x = [radiomic_features (selected ~20-30 after Phase 1 feature selection)
     delta_features (Δf normalised by delta_weeks)
     is_baseline_scan (binary flag)
     time_from_diagnosis_weeks
     scan_index]
```

### Edge structure
One undirected edge between ET and NC (bidirectional in PyG: 2 directed edges).

Edge features:
```
edge_attr = [volumetric_ratio,   # ET_volume / NC_volume
             delta_t_weeks]      # temporal interval — monitor importance in paper
```

Volumetric ratio encodes the relative size relationship between the two
compartments, which is a known predictor of tumor aggressiveness.
delta_t_weeks is included as an edge feature so its importance can be
explicitly monitored and reported (Clinical Workflow Leakage control b).

### Graph-level label
```
y = RANO class at T+1  (0=Progressive, 1=Stable, 2=Response)
```

---

## Step 1.1 — GraphBuilder

**File**: `src/graphs/graph_builder.py`

Responsible for building a single `torch_geometric.data.Data` object
from one row of the paired dataset.

```python
@dataclass
class GraphConfig:
    node_feature_cols: list[str]    # ET_* and NC_* selected features
    edge_feature_cols: list[str]    # volumetric_ratio, delta_t_weeks
    label_col: str                  # "label_t1"
    label_mapping: dict[str, int]   # {"Progressive": 0, "Stable": 1, "Response": 2}

def build_graph(row: pd.Series, config: GraphConfig) -> Data:
    ...
```

The config is loaded from `configs/graph_config.yaml` — never hardcoded.

---

## Step 1.2 — DeltaGraphBuilder

**File**: `src/graphs/delta_graph.py`

Encodes the change between two consecutive graphs:
```
Δnode_features = (features_T - features_{T-1}) / delta_weeks
```

This is already computed in Phase 0 as delta columns.
The DeltaGraphBuilder simply reads those columns and assigns them as
a secondary node feature tensor, keeping static and dynamic features separate.

For baseline scans (is_baseline_scan=True): delta features are zero vectors.

---

## Step 1.3 — TemporalSequence

**File**: `src/graphs/temporal_sequence.py`

Assembles all graphs for a patient into a single object:

```python
@dataclass
class PatientGraphSequence:
    patient_id: str
    graphs: list[Data]          # one per timepoint, ordered chronologically
    n_timepoints: int
    label_sequence: list[int]   # RANO class at each T+1
```

Saved as `data/processed/graphs/{patient_id}.pt` via `torch.save`.

---

## Step 1.4 — Feature Selection (mRMR + Stability Selection)

Before building the final graphs, reduce the feature space from ~856 to
~20-30 features. This step executes on the training fold only (inside CV).

**Algorithm**: mRMR (Minimum Redundancy Maximum Relevance)
```
max I(xi; y) - (1/|S|) * sum I(xi; xj ∈ S)
```

**MI estimator**: Kraskov (k-nearest neighbours), appropriate for
continuous variables on small n.

**Stability Selection**: repeat mRMR on B=100 bootstrap replicates.
Keep only features where P(xi selected) > τ=0.7.

Rationale: on n=68 patients, standard mRMR selects different features
at each fold. Stability Selection ensures only features that are
consistently informative survive.

**Implementation**: `src/preprocessing/feature_selector.py`

Output: `configs/selected_features.yaml` — the list of selected features,
versioned and logged to MLflow alongside model results.

---

## Validation

After building all graphs, run:
```python
assert all(g.x.shape[1] == n_features for g in graphs)
assert all(g.edge_index.shape == (2, 2) for g in graphs)  # bidirectional
assert all(g.y.shape == (1,) for g in graphs)
```

---

## Definition of Done for Phase 1

- [ ] GraphBuilder implemented and unit tested with synthetic data
- [ ] DeltaGraphBuilder implemented and unit tested
- [ ] TemporalSequence serialisation working (save/load roundtrip test)
- [ ] Feature selection pipeline implemented (mRMR + Stability Selection)
- [ ] `configs/graph_config.yaml` defined
- [ ] All patient graphs saved to `data/processed/graphs/` and DVC tracked
- [ ] Graph structure verified: 2 nodes, 1 bidirectional edge, correct feature dims
