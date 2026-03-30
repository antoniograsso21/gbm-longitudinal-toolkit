"""
src/models/gnn.py
==================
TumorGraphNet — per-timepoint GNN message passing module.

Takes a single PyG Data object (one scan) and produces a graph-level
embedding via GATv2Conv + global mean pooling.

Architecture:
    GATv2Conv(in_channels, hidden, heads, edge_dim=2, concat=True)
        → optional second GATv2Conv layer (ablation A1 flag)
        → global_mean_pool
        → Linear projection to hidden_out
        → output embedding of shape [1, hidden_out]

Design decisions:
    - GATv2Conv is used over GATConv because it computes attention as a
      function of (Wh_i || Wh_j), making attention scores dependent on
      both source and target — provably more expressive on heterogeneous
      nodes (Brody et al. 2022).
    - edge_dim=2 passes [volumetric_ratio, interval_weeks] into the
      attention computation — allows the GNN to modulate message passing
      by relative compartment size and temporal proximity.
    - global_mean_pool aggregates across the 3 (or 2) nodes: this is the
      minimal permutation-invariant aggregation. Sum and max variants are
      ablatable via pool_type parameter but not recommended on n=3 nodes
      where mean is the most numerically stable.
    - 1 GATv2Conv layer is the default. Use 2 only if ablation A1 shows
      a meaningful gain — STEP_4.md guideline.
    - All hyperparameters are arguments (no hardcoded defaults in the
      forward pass). Caller (run_gnn.py) owns configuration.

Usage (standalone test):
    from src.models.gnn import TumorGraphNet
    from torch_geometric.data import Data
    import torch
    g = Data(x=torch.randn(3, 20), edge_index=..., edge_attr=torch.randn(12, 2), y=torch.tensor([0]))
    net = TumorGraphNet(in_channels=20, hidden=32, heads=1, n_layers=1)
    emb = net(g.x, g.edge_index, g.edge_attr, batch=torch.zeros(3, dtype=torch.long))
    # emb.shape == (1, 32)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_mean_pool


class TumorGraphNet(nn.Module):
    """
    Per-timepoint GNN encoder for a single tumor compartment graph.

    Args:
        in_channels:  number of node input features.
        hidden:       GATv2Conv output channels per head.
        heads:        number of attention heads per GATv2Conv layer.
        n_layers:     number of GATv2Conv layers (1 or 2). Default 1.
                      Use 2 only when ablation A1 confirms gain.
        dropout:      dropout probability applied after each conv layer.
        edge_dim:     edge feature dimensionality (default 2: volumetric_ratio + interval_weeks).
        pool_type:    graph pooling aggregation ("mean" only in V1).

    Output:
        Graph-level embedding tensor of shape [batch_size, hidden * heads].
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        heads: int = 1,
        n_layers: int = 1,
        dropout: float = 0.2,
        edge_dim: int = 2,
        pool_type: str = "mean",
    ) -> None:
        super().__init__()

        if n_layers not in (1, 2):
            raise ValueError(f"n_layers must be 1 or 2, got {n_layers}")
        if pool_type != "mean":
            raise ValueError(f"pool_type '{pool_type}' not supported in V1; use 'mean'")

        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)

        # Layer 1
        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden,
            heads=heads,
            concat=True,          # output: hidden * heads
            dropout=dropout,
            edge_dim=edge_dim,
        )

        # Optional layer 2 (ablation A1)
        if n_layers == 2:
            self.conv2 = GATv2Conv(
                in_channels=hidden * heads,
                out_channels=hidden,
                heads=heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim,
            )
        else:
            self.conv2 = None  # type: ignore[assignment]

        self.out_dim = hidden * heads

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """
        Forward pass: node features → graph embedding.

        Args:
            x:          node features, shape [total_nodes, in_channels].
            edge_index: COO edge index, shape [2, total_edges].
            edge_attr:  edge features, shape [total_edges, edge_dim].
            batch:      node-to-graph assignment, shape [total_nodes].
                        Use torch.zeros(n_nodes, dtype=torch.long) for single graph.

        Returns:
            Graph embedding tensor of shape [batch_size, hidden * heads].
        """
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.dropout(x)

        # Layer 2 (optional)
        if self.conv2 is not None:
            x = self.conv2(x, edge_index, edge_attr)
            x = torch.relu(x)
            x = self.dropout(x)

        # Global mean pooling: [total_nodes, hidden*heads] → [batch_size, hidden*heads]
        return global_mean_pool(x, batch)

    def get_attention_weights(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Return attention weights from conv1 for interpretability (Step 5).

        Returns:
            (edge_index, alpha) where alpha has shape [n_edges, heads].
        """
        _, (edge_index_out, alpha) = self.conv1(
            x, edge_index, edge_attr, return_attention_weights=True
        )
        return edge_index_out, alpha
