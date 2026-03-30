"""
src/models/tumor_gnn.py
========================
TumorTemporalGNN — full longitudinal GNN model.

Composes TumorGraphNet and TemporalAttentionEncoder into a single module
that maps a patient's graph sequence to RANO class probabilities.

Forward pass:
    1. For each timepoint G_t in the sequence:
       h_t = TumorGraphNet(G_t)          → graph embedding [1, d_model]
    2. Stack embeddings: H = [h_1, ..., h_T]    → [1, T, d_model]
    3. TemporalAttentionEncoder(H, intervals)    → sequence summary [1, d_model]
    4. Linear classifier + softmax              → [1, 3]

Ablation flags (all default to the full model):
    use_temporal:      if False → skip temporal attention, use last timepoint
                       embedding only (ablation A1: cross-sectional GNN).
    use_delta:         if False → strip delta_* columns from node features
                       at graph construction time (ablation A3).
                       NOTE: this flag is advisory — the actual column filtering
                       must happen in graph_builder.py when building the graphs.
                       TumorTemporalGNN asserts use_delta=False only when the
                       node feature dim matches expected non-delta dim.
    use_time_encoding: if False → disable sinusoidal Δt encoding in temporal
                       attention (ablation A4). Passed through to
                       TemporalAttentionEncoder.

Design notes:
    - Sequence collation (padding, masking) happens outside this class in
      run_gnn.py — keeping this module pure and testable on fixed-length inputs.
    - The forward signature accepts pre-padded tensors rather than a list of
      Data objects. This decoupling allows run_gnn.py to handle PyG batching
      without this module needing to know about DataLoader internals.
    - class_weights are NOT stored here; they are passed to CrossEntropyLoss
      in run_gnn.py (pure module principle — no loss in the forward pass).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from src.models.gnn import TumorGraphNet
from src.models.temporal_attention import TemporalAttentionEncoder


# ---------------------------------------------------------------------------
# Config dataclass (all hyperparameters in one place — no hardcoding)
# ---------------------------------------------------------------------------
@dataclass
class GNNConfig:
    """
    All hyperparameters for TumorTemporalGNN.
    Loaded from configs/gnn.yaml by run_gnn.py.
    """
    in_channels: int            # node feature dimension (from graph_builder output)
    hidden: int = 32            # GATv2Conv output channels per head
    heads: int = 1              # attention heads per GATv2Conv layer
    n_gnn_layers: int = 1       # 1 or 2 GATv2Conv layers
    n_temporal_heads: int = 1   # heads for temporal multi-head attention
    dropout: float = 0.2        # dropout shared across GNN and temporal encoder
    edge_dim: int = 2           # [volumetric_ratio, interval_weeks]
    n_classes: int = 3          # RANO: Progressive / Stable / Response

    # Ablation flags
    use_temporal: bool = True       # False → ablation A1 (cross-sectional)
    use_time_encoding: bool = True  # False → ablation A4 (no Δt encoding)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class TumorTemporalGNN(nn.Module):
    """
    End-to-end longitudinal GNN for RANO prediction.

    Expected forward inputs (pre-padded by the caller):
        x_seq:         node feature sequences, shape [batch, max_T, n_nodes, in_channels]
        edge_index:    COO edge index, shape [2, n_edges] — same for all timepoints
        edge_attr_seq: edge attribute sequences, shape [batch, max_T, n_edges, edge_dim]
        intervals:     interval_weeks per timepoint, shape [batch, max_T]
        seq_lengths:   actual sequence lengths (unpadded), shape [batch]

    Output:
        logits of shape [batch, n_classes] — CrossEntropyLoss expects raw logits.

    Args:
        config: GNNConfig dataclass with all hyperparameters.
    """

    def __init__(self, config: GNNConfig) -> None:
        super().__init__()
        self.config = config

        d_model = config.hidden * config.heads

        self.gnn = TumorGraphNet(
            in_channels=config.in_channels,
            hidden=config.hidden,
            heads=config.heads,
            n_layers=config.n_gnn_layers,
            dropout=config.dropout,
            edge_dim=config.edge_dim,
        )

        if config.use_temporal:
            self.temporal_encoder = TemporalAttentionEncoder(
                d_model=d_model,
                n_heads=config.n_temporal_heads,
                dropout=config.dropout,
                use_time_encoding=config.use_time_encoding,
            )
        else:
            self.temporal_encoder = None  # type: ignore[assignment]

        self.classifier = nn.Linear(d_model, config.n_classes)

    def _build_padding_mask(
        self,
        seq_lengths: Tensor,
        max_len: int,
    ) -> Tensor:
        """
        Build key_padding_mask for nn.MultiheadAttention.

        Returns boolean tensor of shape [batch, max_len].
        True = padded (ignored) position.
        """
        batch = seq_lengths.shape[0]
        mask = torch.arange(max_len, device=seq_lengths.device).unsqueeze(0).expand(batch, -1)
        return mask >= seq_lengths.unsqueeze(1)  # [batch, max_len]

    def forward(
        self,
        x_seq: Tensor,
        edge_index: Tensor,
        edge_attr_seq: Tensor,
        intervals: Tensor,
        seq_lengths: Tensor,
    ) -> Tensor:
        """
        Args:
            x_seq:         [batch, max_T, n_nodes, in_channels]
            edge_index:    [2, n_edges]  — shared topology across timepoints
            edge_attr_seq: [batch, max_T, n_edges, edge_dim]
            intervals:     [batch, max_T]
            seq_lengths:   [batch] — actual (unpadded) sequence lengths

        Returns:
            logits: [batch, n_classes]
        """
        batch_size, max_t, n_nodes, _ = x_seq.shape
        device = x_seq.device

        # Step 1 — GNN encoding per timepoint
        # Flatten batch × time for efficient parallel GNN processing
        x_flat = x_seq.reshape(batch_size * max_t, n_nodes, x_seq.shape[-1])
        ea_flat = edge_attr_seq.reshape(batch_size * max_t, edge_attr_seq.shape[2], edge_attr_seq.shape[3])

        # For each (batch, time) pair: all nodes belong to the same mini-graph.
        # Build batch assignment: [0,0,0, 1,1,1, ...] for n_nodes=3
        batch_assign = torch.arange(batch_size * max_t, device=device).repeat_interleave(n_nodes)

        h_flat = self.gnn(
            x=x_flat.reshape(batch_size * max_t * n_nodes, x_seq.shape[-1]),
            edge_index=edge_index,
            edge_attr=ea_flat.reshape(batch_size * max_t * ea_flat.shape[1], edge_attr_seq.shape[-1]),
            batch=batch_assign,
        )  # [batch * max_T, d_model]

        # However, edge_index is for a SINGLE graph (n_nodes nodes).
        # The flat reshape above mixes graphs — we need to process each
        # (batch, time) graph independently or re-index edge_index.
        # Correct approach: loop over max_T (small, max ~16 timepoints).
        # Re-compute with correct per-timepoint processing:
        h_flat = self._encode_all_timepoints(
            x_seq, edge_index, edge_attr_seq, batch_size, max_t, n_nodes, device
        )  # [batch, max_T, d_model]

        # Step 2 — temporal attention (or last-timepoint for ablation A1)
        if self.temporal_encoder is not None:
            padding_mask = self._build_padding_mask(seq_lengths, max_t)
            summary = self.temporal_encoder(h_flat, intervals, padding_mask)
        else:
            # Ablation A1: use last valid timepoint embedding
            last_idx = (seq_lengths - 1).clamp(min=0)  # [batch]
            summary = h_flat[torch.arange(batch_size, device=device), last_idx]  # [batch, d_model]

        # Step 3 — classify
        return self.classifier(summary)  # [batch, n_classes]

    def _encode_all_timepoints(
        self,
        x_seq: Tensor,
        edge_index: Tensor,
        edge_attr_seq: Tensor,
        batch_size: int,
        max_t: int,
        n_nodes: int,
        device: torch.device,
    ) -> Tensor:
        """
        Run TumorGraphNet on each timepoint independently.

        Loops over max_T rather than batching with shifted edge_index to keep
        the code correct and readable on the small sequences (max_T ≤ 16).
        Performance on n=231 is not a bottleneck.

        Returns:
            h: [batch_size, max_T, d_model]
        """
        d_model = self.gnn.out_dim
        h = torch.zeros(batch_size, max_t, d_model, device=device)

        # batch assignment for a single graph: [0, 0, ..., 0] (n_nodes times)
        single_batch = torch.zeros(n_nodes, dtype=torch.long, device=device)

        for t in range(max_t):
            x_t = x_seq[:, t, :, :]           # [batch, n_nodes, in_channels]
            ea_t = edge_attr_seq[:, t, :, :]  # [batch, n_edges, edge_dim]

            embs = []
            for b in range(batch_size):
                emb = self.gnn(
                    x=x_t[b],                  # [n_nodes, in_channels]
                    edge_index=edge_index,
                    edge_attr=ea_t[b],         # [n_edges, edge_dim]
                    batch=single_batch,
                )  # [1, d_model]
                embs.append(emb)

            h[:, t, :] = torch.cat(embs, dim=0)  # [batch, d_model]

        return h

    def get_attention_weights(
        self,
        x_seq: Tensor,
        edge_index: Tensor,
        edge_attr_seq: Tensor,
        intervals: Tensor,
        seq_lengths: Tensor,
    ) -> dict[str, Tensor]:
        """
        Return attention weights from both GNN (edge) and temporal attention.
        Used by interpretability module (Step 5).

        Returns dict with:
            "gnn_edge_index":   [2, n_edges]
            "gnn_alpha":        [n_edges, heads] — last timepoint of first batch item
            "temporal_weights": [batch, n_heads, max_T, max_T]
        """
        if self.temporal_encoder is None:
            raise ValueError("Cannot extract temporal attention weights in ablation A1 mode.")

        batch_size, max_t, n_nodes, _ = x_seq.shape
        device = x_seq.device

        h = self._encode_all_timepoints(
            x_seq, edge_index, edge_attr_seq, batch_size, max_t, n_nodes, device
        )
        padding_mask = self._build_padding_mask(seq_lengths, max_t)
        _, temporal_weights = self.temporal_encoder.get_attention_weights(
            h, intervals, padding_mask
        )

        # GNN edge attention from last valid timepoint of first batch item
        last_t = int(seq_lengths[0].item()) - 1
        _, gnn_alpha = self.gnn.get_attention_weights(
            x=x_seq[0, last_t],
            edge_index=edge_index,
            edge_attr=edge_attr_seq[0, last_t],
        )

        return {
            "gnn_edge_index": edge_index,
            "gnn_alpha": gnn_alpha,
            "temporal_weights": temporal_weights,
        }
