# Phase 3 — Temporal GNN

## Objective
Build and evaluate the core model: a temporal graph neural network that
processes a sequence of tumor graphs per patient to predict the next RANO state.

**Input**: `data/processed/graphs/{patient_id}.pt` (Phase 1 output)
**Output**: MLflow experiment `gnn/` with full ablation study results

---

## Architecture Overview

```
Patient graph sequence: [G_1, G_2, ..., G_T]
        │
        ▼
[1] GNN Message Passing (per timepoint)
    For each G_t: node features updated via graph convolution
    ET and NC exchange information through the edge (volumetric ratio, Δt)
        │
        ▼
[2] Temporal Attention
    Sequence of graph embeddings [h_1, h_2, ..., h_T]
    Multi-head attention over the temporal dimension
    Handles irregular intervals via positional encoding of Δt
        │
        ▼
[3] Classifier head
    Aggregated temporal embedding → Linear(hidden, 3) → Softmax
        │
        ▼
Output: P(Progressive), P(Stable), P(Response) for T+1
```

---

## Component 1 — GNN Message Passing

**File**: `src/models/gnn.py`
**Class**: `TumorGraphNet`

With only 2 nodes, the message passing step is equivalent to:
"ET and NC each update their representation by aggregating features
from their neighbour, weighted by the edge attributes."

This is simple but principled: it learns a shared representation
that explicitly models the ET-NC relationship.

**Layer choice**: GATv2Conv (Graph Attention Network v2)
- Learns attention weights over the single ET↔NC edge
- More expressive than GCN (fixed normalisation) on small graphs
- Interpretable: the attention weight is a learnable measure of
  how much ET should attend to NC (and vice versa) at each timepoint

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
        return x.mean(dim=0)  # global mean pool (2 nodes → 1 vector)
```

**Config** (`configs/gnn.yaml`):
```yaml
in_channels: ~selected_features   # from feature selection
hidden: [32, 64]
heads: [1, 2, 4]
dropout: [0.2, 0.3]
```

---

## Component 2 — Temporal Attention

**File**: `src/models/temporal_attention.py`
**Class**: `TemporalAttention`

Processes the sequence of per-timepoint graph embeddings.

**Irregular time handling**: LUMIERE intervals are not fixed.
Standard positional encodings assume fixed spacing and are incorrect here.
Instead, use a continuous-time positional encoding based on Δt:

```python
# Sinusoidal encoding of actual temporal interval
def temporal_encoding(delta_t_weeks: Tensor, d_model: int) -> Tensor:
    # Standard sin/cos encoding with actual Δt as position
    ...
```

This encodes "how much time has passed" rather than "which step in the sequence",
which is both biologically correct and honest about the irregular sampling.

**Architecture**:
```
Input: (seq_len, hidden)  — one vector per timepoint
  → Add temporal positional encoding (based on Δt, not step index)
  → Multi-head self-attention (heads=2, dropout=0.1)
  → Mean pooling over sequence
  → Output: (hidden,)  — patient-level embedding
```

---

## Component 3 — Full Model

**File**: `src/models/tumor_gnn.py`
**Class**: `TumorTemporalGNN`

```python
class TumorTemporalGNN(nn.Module):
    def __init__(self, config: GNNConfig):
        self.graph_encoder = TumorGraphNet(...)
        self.temporal_attn = TemporalAttention(...)
        self.classifier = nn.Linear(config.hidden, 3)

    def forward(self, graph_sequence: list[Data]) -> Tensor:
        embeddings = [self.graph_encoder(g) for g in graph_sequence]
        embeddings = torch.stack(embeddings)          # (seq_len, hidden)
        patient_embedding = self.temporal_attn(embeddings)
        return self.classifier(patient_embedding)     # (3,) logits
```

---

## Training Setup

**Same CV as Phase 2** (StratifiedGroupKFold, group=patient).
**Same metrics** (macro F1, MCC, AUC per class).
**Same seed fixing** (Python, NumPy, PyTorch).

**Loss function**: cross-entropy with class weights inversely proportional
to class frequency. Handles 72%/14%/14% imbalance.

```python
weights = compute_class_weight("balanced",
                               classes=[0, 1, 2],
                               y=y_train)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
```

**Optimiser**: AdamW (weight decay as implicit regularisation on small n)
**Scheduler**: ReduceLROnPlateau (patience=10)
**Early stopping**: patience=20 epochs on validation macro F1

---

## Ablation Study (mandatory for paper)

The ablation isolates the contribution of each architectural component.
All ablations use the same CV splits as the full model.

| Ablation | What is removed | Question answered |
|----------|----------------|-------------------|
| A1: Cross-sectional GNN | Temporal attention | Does history help? |
| A2: No graph (LSTM) | GNN message passing | Does graph structure help? |
| A3: No delta features | Δf columns | Does rate of change help? |
| A4: No Δt encoding | Temporal positional encoding | Does irregular time matter? |
| A5: Δt only | All radiomic features | Leakage quantification |

A1 vs Full model → measures contribution of temporal modelling (tests A3).
A2 is Baseline 3 from Phase 2 → already computed.
A5 is the Δt ablation from Phase 2 → already computed.

---

## Interpretability

For the paper, report:
1. GATv2 attention weights averaged across patients — which feature
   direction (ET→NC or NC→ET) dominates at each disease stage?
2. Temporal attention weights — which timepoints are most predictive?
   (baseline scan vs recent scan)
3. Feature importance from Phase 1 feature selection — which radiomic
   features were most consistently selected?

These do not require SHAP or additional libraries — they are direct
outputs of the attention mechanism.

---

## Definition of Done for Phase 3

- [ ] TumorGraphNet implemented and unit tested
- [ ] TemporalAttention with continuous-time encoding implemented
- [ ] TumorTemporalGNN full model implemented
- [ ] CV training loop with early stopping and MLflow logging
- [ ] Class-weighted loss implemented
- [ ] Full ablation study (A1–A5) run and logged
- [ ] Comparison table from Phase 2 completed with GNN results
- [ ] Attention weight visualisation for paper
- [ ] Random seeds fixed and logged
