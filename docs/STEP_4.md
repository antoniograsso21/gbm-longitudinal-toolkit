# Step 4 — Graph Construction + Temporal GNN ⏳

## Objective
Build patient graph sequences from the paired dataset and train the temporal GNN.

**Input**:
- `data/processed/dataset_paired.parquet`
- `configs/selected_features.yaml` (from Step 3)

**Output**:
- `data/processed/graphs/{patient_id}.pt`
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

Node feature distinctiveness is guaranteed by construction: each node uses only
its own label-prefixed columns (NC_*, CE_*, ED_*), so GATv2Conv sees different
initial representations even before message passing.

### Node features per node
```
[selected radiomic features (~20-30, from selected_features.yaml)
 delta features (Δlog(f) / interval_weeks)
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

**File**: `src/graphs/graphs_builder.py`
**Key classes**: `GraphConfig`, `PatientGraphSequence`
**Validation**: `graphs_validator.py — asserts` — asserts shape invariants on every graph

Also build HD-GLIO-AUTO 2-node graphs for ablation (same architecture, 2 nodes).

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

---

## Definition of Done

- [ ] `graphs_builder.py` implemented and unit tested
- [ ] `graphs_validator.py` passing all structural assertions
- [ ] 3-node graphs built and saved for all 64 patients
- [ ] 2-node graphs built for HD-GLIO-AUTO (ablation A6)
- [ ] `graphs_validator.py — asserts` passing all structural assertions
- [ ] TumorTemporalGNN implemented and unit tested
- [ ] CV training loop with early stopping and MLflow logging
- [ ] Class-weighted loss implemented
- [ ] Full ablation study A1–A6 run and logged
- [ ] Comparison table from Step 3 completed with GNN results
- [ ] Random seeds fixed and logged
- [ ] `configs/gnn.yaml` committed