# Phase 3 — Temporal GNN

## Objective
Build and evaluate the core model: a temporal GNN that processes a sequence
of 3-node tumor graphs per patient to predict the next RANO state.

**Input**: `data/processed/graphs/{patient_id}.pt`
**Output**: MLflow experiment `gnn/` with full ablation study results

---

## Architecture Overview

```
Patient graph sequence: [G_1, G_2, ..., G_T]
        │
        ▼
[1] GNN Message Passing (per timepoint)
    3-node triangular graph: Necrosis ↔ Contrast-enhancing ↔ Edema
    Each node updates its representation via GATv2Conv
        │
        ▼
[2] Temporal Attention
    Sequence of graph embeddings [h_1, ..., h_T]
    Multi-head attention with continuous-time positional encoding (Δt-based)
        │
        ▼
[3] Classifier head
    Linear(hidden, 3) → Softmax
        │
        ▼
Output: P(Progressive), P(Stable), P(Response) for T+1
```

---

## Component 1 — GNN Message Passing

**File**: `src/models/gnn.py`
**Layer**: GATv2Conv — learns attention weights per edge, interpretable

With 3 nodes and 6 directed edges, message passing is:
"each node updates its representation by attending to its two neighbours,
weighted by learnable attention over the edge features (volumetric ratio, Δt)."

```python
class TumorGraphNet(nn.Module):
    def __init__(self, in_channels: int, hidden: int, heads: int = 2):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden, heads=heads,
                               edge_dim=2, concat=True)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=1,
                               edge_dim=2, concat=False)

    def forward(self, data: Data) -> Tensor:
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = self.conv2(x, data.edge_index, data.edge_attr)
        return x.mean(dim=0)  # global mean pool: 3 nodes -> 1 vector
```

---

## Component 2 — Temporal Attention

**File**: `src/models/temporal_attention.py`

Irregular time handling: LUMIERE intervals are not fixed. Standard positional
encodings assume fixed spacing. Instead, encode actual Δt:

```python
def temporal_encoding(delta_t_weeks: Tensor, d_model: int) -> Tensor:
    # Sinusoidal encoding with actual Δt as position (not step index)
    ...
```

---

## Component 3 — Full Model

**File**: `src/models/tumor_gnn.py`

```python
class TumorTemporalGNN(nn.Module):
    def forward(self, graph_sequence: list[Data]) -> Tensor:
        embeddings = [self.graph_encoder(g) for g in graph_sequence]
        embeddings = torch.stack(embeddings)
        patient_embedding = self.temporal_attn(embeddings)
        return self.classifier(patient_embedding)
```

**Loss**: CrossEntropyLoss with class weights (handles 77%/11%/12% imbalance)
**Optimiser**: AdamW | **Scheduler**: ReduceLROnPlateau | **Early stopping**: patience=20

---

## Ablation Study (mandatory for paper)

| Ablation | What is removed | Question answered |
|----------|----------------|-------------------|
| A1: Cross-sectional GNN | Temporal attention | Does history help? |
| A2: No graph (LSTM) | GNN message passing | Does graph structure help? |
| A3: No delta features | Δf columns | Does rate of change help? |
| A4: No Δt encoding | Temporal positional encoding | Does irregular time matter? |
| A5: delta_t only | All radiomic features | Leakage quantification |
| A6: 2-node GNN (HD-GLIO-AUTO) | Edema node | Does edema add signal? |

A2 = Baseline 3 (LSTM) from Phase 2 — already computed.
A5 = delta_t ablation from Phase 2 — already computed.
A6 is new: trains the same GNN architecture on HD-GLIO-AUTO graphs (2 nodes).
Directly answers the scientific question of whether the 3rd node adds value.

---

## Interpretability

For the paper:
1. GATv2 attention weights per edge: which compartment relationships dominate?
   (Necrosis→Edema? Enhancing→Necrosis?) stratified by disease stage
2. Temporal attention weights: baseline scan vs recent scan importance
3. Feature importance from Phase 1 feature selection: consistently selected features

---

## Definition of Done for Phase 3

- [ ] TumorGraphNet (3-node) implemented and unit tested
- [ ] TemporalAttention with continuous-time encoding implemented
- [ ] TumorTemporalGNN full model implemented
- [ ] CV training loop with early stopping and MLflow logging
- [ ] Class-weighted loss implemented
- [ ] Full ablation study (A1–A6) run and logged
- [ ] Comparison table from Phase 2 completed with GNN results
- [ ] Attention weight visualisation for paper
- [ ] Random seeds fixed and logged
