"""
src/models/temporal_attention.py
==================================
Temporal attention module for irregular longitudinal graph sequences.

Two components:

1. ContinuousTimeEncoding
   Encodes actual inter-scan interval (interval_weeks) using sinusoidal
   functions with learnable frequency parameters.
   Rationale: scan intervals in LUMIERE are irregular (1–50 weeks, mean ~13.7w).
   A fixed step-index encoding (0, 1, 2, ...) would treat a 2-week and a
   50-week gap identically — the model would be blind to irregular time.
   Using interval_weeks as the continuous position variable allows the
   attention mechanism to learn that distant scans contribute differently
   from recent ones.

2. TemporalAttentionEncoder
   Multi-head self-attention over the sequence of graph embeddings
   [h_1, ..., h_T] with sinusoidal time encodings added to each embedding.
   Returns the attention-weighted summary of the sequence (CLS-style mean
   pooling over the attended sequence), ready for the classifier head.

Design decisions:
    - Sinusoidal encoding follows the formulation in Xu et al. (2019)
      "Self-attention with Functional Time Representation Learning", adapted
      here to use interval_weeks as the time variable instead of step index.
    - Frequencies are initialised from a geometric sequence and kept fixed
      (not learnable). Learnable frequencies are listed in FUTURE.md — on
      n=231 they are likely to overfit.
    - Padding mask is mandatory: sequences have variable length (1–16 scans
      per patient); nn.MultiheadAttention requires an explicit key_padding_mask
      to prevent padded positions from contributing to attention scores.
    - Ablation A4 (no Δt encoding) is supported via use_time_encoding=False
      in TemporalAttentionEncoder — drops the sinusoidal addition without
      changing anything else in the forward graph.

Usage (standalone):
    enc = ContinuousTimeEncoding(d_model=32)
    attn = TemporalAttentionEncoder(d_model=32, n_heads=2)
    # h: [batch, seq_len, 32], intervals: [batch, seq_len], mask: [batch, seq_len] bool
    out = attn(h, intervals, padding_mask=mask)  # [batch, 32]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class ContinuousTimeEncoding(nn.Module):
    """
    Sinusoidal encoding for continuous, irregular time intervals.

    For a time value t (interval_weeks) and embedding dimension d_model:
        PE(t, 2i)   = sin(t / 10000^(2i / d_model))
        PE(t, 2i+1) = cos(t / 10000^(2i / d_model))

    The encoding is added to the graph embedding at each timepoint so the
    attention module can distinguish scans separated by different time gaps.

    Args:
        d_model: embedding dimension (must match graph embedding output dim).
        max_weeks: maximum expected interval in weeks (used to scale divisors).
                   Default 200w — well above the LUMIERE maximum of 50w.
    """

    def __init__(self, d_model: int, max_weeks: float = 200.0) -> None:
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even for sinusoidal encoding, got {d_model}. "
                "Consider padding the embedding dimension."
            )

        self.d_model = d_model

        # Pre-compute frequency divisors — fixed, not learnable.
        # Shape: [d_model // 2]
        i = torch.arange(0, d_model // 2, dtype=torch.float)
        div_term = torch.pow(10000.0, 2 * i / d_model)
        self.register_buffer("div_term", div_term)  # [d_model // 2]

    def forward(self, t: Tensor) -> Tensor:
        """
        Compute sinusoidal encoding for a batch of time values.

        Args:
            t: interval tensor of shape [batch, seq_len] — interval_weeks values.

        Returns:
            Encoding tensor of shape [batch, seq_len, d_model].
        """
        # t: [batch, seq_len] → [batch, seq_len, 1]
        t = t.unsqueeze(-1)

        # div_term: [d_model // 2] → [1, 1, d_model // 2]
        div = self.div_term.view(1, 1, -1)  # type: ignore[attr-defined]

        sin_enc = torch.sin(t / div)   # [batch, seq_len, d_model // 2]
        cos_enc = torch.cos(t / div)   # [batch, seq_len, d_model // 2]

        # Interleave sin and cos: [batch, seq_len, d_model]
        enc = torch.stack([sin_enc, cos_enc], dim=-1)
        return enc.reshape(t.shape[0], t.shape[1], self.d_model)


class TemporalAttentionEncoder(nn.Module):
    """
    Multi-head self-attention over a padded sequence of graph embeddings.

    Aggregates a variable-length sequence [h_1, ..., h_T] into a single
    fixed-size representation by:
        1. Adding sinusoidal time encodings (or skipping — ablation A4).
        2. Applying nn.MultiheadAttention with a key_padding_mask that
           zeros out padded positions.
        3. Mean-pooling the attended output over non-padded positions.

    The mean-pool over attended outputs (rather than a CLS token) avoids
    introducing an additional learnable parameter on n=231 — consistent with
    the minimal-parameterisation principle for this dataset size.

    Args:
        d_model:           embedding dimension (must match TumorGraphNet.out_dim).
        n_heads:           number of attention heads.
        dropout:           dropout on attention weights.
        use_time_encoding: if False, skip sinusoidal addition (ablation A4).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        dropout: float = 0.1,
        use_time_encoding: bool = True,
    ) -> None:
        super().__init__()

        self.use_time_encoding = use_time_encoding

        if use_time_encoding:
            self.time_enc = ContinuousTimeEncoding(d_model=d_model)

        # nn.MultiheadAttention expects batch_first=True for [batch, seq, dim] layout
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: Tensor,
        intervals: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Encode a padded sequence of graph embeddings with temporal attention.

        Args:
            h:             padded graph embeddings, shape [batch, max_seq_len, d_model].
            intervals:     interval_weeks per timepoint, shape [batch, max_seq_len].
                           Padded positions should carry 0.0.
            padding_mask:  boolean mask, shape [batch, max_seq_len].
                           True = padded (ignored) position.
                           None = no padding (all positions valid).

        Returns:
            Sequence summary tensor of shape [batch, d_model].
        """
        # Step 1 — optional sinusoidal time encoding
        if self.use_time_encoding:
            time_emb = self.time_enc(intervals)  # [batch, seq_len, d_model]
            h = h + time_emb

        # Step 2 — multi-head self-attention
        # key_padding_mask: True positions are ignored in attention
        attended, _ = self.attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=padding_mask,
        )  # attended: [batch, seq_len, d_model]

        # Step 3 — residual + layer norm (pre-norm style for training stability on small n)
        attended = self.layer_norm(h + attended)

        # Step 4 — mean pool over non-padded positions
        if padding_mask is not None:
            # Invert mask: True = valid position
            valid = (~padding_mask).float().unsqueeze(-1)  # [batch, seq_len, 1]
            n_valid = valid.sum(dim=1).clamp(min=1.0)      # [batch, 1]
            summary = (attended * valid).sum(dim=1) / n_valid  # [batch, d_model]
        else:
            summary = attended.mean(dim=1)  # [batch, d_model]

        return summary

    def get_attention_weights(
        self,
        h: Tensor,
        intervals: Tensor,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Return attention weights for interpretability (Step 5).

        Returns:
            (summary, attn_weights) where attn_weights has shape
            [batch, n_heads, seq_len, seq_len].
        """
        if self.use_time_encoding:
            h = h + self.time_enc(intervals)

        attended, attn_weights = self.attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=False,  # keep per-head weights
        )
        attended = self.layer_norm(h + attended)

        if padding_mask is not None:
            valid = (~padding_mask).float().unsqueeze(-1)
            n_valid = valid.sum(dim=1).clamp(min=1.0)
            summary = (attended * valid).sum(dim=1) / n_valid
        else:
            summary = attended.mean(dim=1)

        return summary, attn_weights
