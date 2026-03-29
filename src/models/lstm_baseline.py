"""
src/models/lstm_baseline.py
============================
LSTM baseline for the LUMIERE pipeline (Step 3 — T3.4).

Temporal model without graph structure. Isolates the contribution of graph
topology relative to temporal modelling alone.

Expected result: LSTM ≈ LightGBM D or worse, given mean sequence length ~3.6.
This is an honest scientific result — declare in paper.

Input:
    Sequences of shape (seq_len, n_features) per patient.
    n_features = full_feature_set from AnchoredFoldSelectionResult (same as LightGBM D).
    Variable-length sequences handled via pack_padded_sequence.

Architecture (fixed — no architecture search, only hidden_size/lr grid):
    LSTM(hidden, layers, dropout)
      → last hidden state (h_n[-1])
      → Linear(hidden, 3)

Loss: CrossEntropyLoss with class weights (76/11/13 imbalance).
Optimiser: Adam. Scheduler: ReduceLROnPlateau(patience=5, monitor=val_loss).
Early stopping: monitors val_loss — more stable than macro_f1 on ~6 val patients.

Design:
    All functions are pure — no MLflow, no file I/O, no printing.
    Side effects live exclusively in run_lstm_baseline.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from src.training.metrics import FoldMetrics, compute_metrics


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TumorLSTM(nn.Module):
    """
    Single LSTM classifier for longitudinal tumor feature sequences.

    Args:
        input_size:  number of features per timepoint.
        hidden_size: LSTM hidden units.
        num_layers:  stacked LSTM layers.
        dropout:     dropout between LSTM layers (ignored if num_layers=1).
        n_classes:   output classes (3 for RANO).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       padded input tensor, shape (batch, max_seq_len, input_size).
            lengths: actual sequence lengths per sample, shape (batch,).

        Returns:
            logits of shape (batch, n_classes).
        """
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers, batch, hidden_size) — take last layer
        last_hidden = h_n[-1]
        return self.classifier(last_hidden)


# ---------------------------------------------------------------------------
# Sequence builder (pure)
# ---------------------------------------------------------------------------

def build_patient_sequences(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    scan_indices: np.ndarray,
) -> list[dict]:
    """
    Group rows by patient and build chronological feature sequences.

    Each patient yields one dict:
        seq:    float32 array (seq_len, n_features) — ordered by scan_index
        label:  int — RANO label of the last timepoint (already label-shifted)
        length: int — sequence length

    Args:
        X:            normalised feature matrix, shape (n_samples, n_features).
        y:            integer label array, shape (n_samples,).
        patient_ids:  patient ID per row, shape (n_samples,).
        scan_indices: 0-based scan ordinal per row, shape (n_samples,).

    Returns:
        List of dicts, one per patient.

    Raises:
        ValueError: if any patient has zero rows after grouping.
    """
    sequences: list[dict] = []
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        order = np.argsort(scan_indices[mask])
        seq = X[mask][order].astype(np.float32)
        if len(seq) == 0:
            raise ValueError(f"Patient {pid} has zero rows — check CV split.")
        label = int(y[mask][order][-1])
        sequences.append({"seq": seq, "label": label, "length": len(seq)})
    return sequences


def collate_sequences(
    batch: list[dict],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a list of sequence dicts into padded batch tensors.

    Returns:
        x_padded: (batch, max_seq_len, n_features)
        lengths:  (batch,) — actual lengths
        labels:   (batch,) — integer labels
    """
    seqs = [torch.tensor(s["seq"]) for s in batch]
    lengths = torch.tensor([s["length"] for s in batch], dtype=torch.long)
    labels = torch.tensor([s["label"] for s in batch], dtype=torch.long)
    x_padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    return x_padded, lengths, labels


# ---------------------------------------------------------------------------
# Typed result
# ---------------------------------------------------------------------------

@dataclass
class LSTMFoldResult:
    """Full result for a single CV fold of the LSTM baseline."""
    fold: int
    best_hidden_size: int
    best_num_layers: int
    best_dropout: float
    best_lr: float
    metrics: FoldMetrics
    n_train_patients: int
    n_test_patients: int
    n_features: int
    n_epochs_trained: int


# ---------------------------------------------------------------------------
# Training loop (pure — returns result + best_val_loss, no side effects)
# ---------------------------------------------------------------------------

def train_lstm_fold(
    train_sequences: list[dict],
    val_sequences: list[dict],
    test_sequences: list[dict],
    fold: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 100,
    patience: int = 15,
    class_weights: torch.Tensor | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[LSTMFoldResult, float]:
    """
    Train LSTM on one CV fold with early stopping on validation loss.

    Returns:
        (LSTMFoldResult, best_val_loss)
        best_val_loss is used by _grid_search_lstm for config selection.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = train_sequences[0]["seq"].shape[1]
    dev = torch.device(device)

    model = TumorLSTM(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(dev)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(dev) if class_weights is not None else None
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    n_epochs_trained = 0

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        np.random.shuffle(train_sequences)
        for i in range(0, len(train_sequences), batch_size):
            batch = train_sequences[i : i + batch_size]
            x, lengths, labels = collate_sequences(batch)
            x, lengths, labels = x.to(dev), lengths.to(dev), labels.to(dev)
            optimiser.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_sequences), batch_size):
                batch = val_sequences[i : i + batch_size]
                x, lengths, labels = collate_sequences(batch)
                x, lengths, labels = x.to(dev), lengths.to(dev), labels.to(dev)
                logits = model(x, lengths)
                val_loss += criterion(logits, labels).item() * len(batch)
        val_loss /= len(val_sequences)
        scheduler.step(val_loss)

        n_epochs_trained = epoch + 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # --- Evaluate on test ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    all_preds: list[int] = []
    all_probas: list[np.ndarray] = []
    all_true: list[int] = []

    with torch.no_grad():
        for i in range(0, len(test_sequences), batch_size):
            batch = test_sequences[i : i + batch_size]
            x, lengths, labels = collate_sequences(batch)
            x, lengths = x.to(dev), lengths.to(dev)
            logits = model(x, lengths)
            probas = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probas, axis=1)
            all_preds.extend(preds.tolist())
            all_probas.append(probas)
            all_true.extend(labels.tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    y_proba = np.vstack(all_probas)

    fold_metrics = compute_metrics(
        fold=fold,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    return LSTMFoldResult(
        fold=fold,
        best_hidden_size=hidden_size,
        best_num_layers=num_layers,
        best_dropout=dropout,
        best_lr=learning_rate,
        metrics=fold_metrics,
        n_train_patients=len(train_sequences),
        n_test_patients=len(test_sequences),
        n_features=n_features,
        n_epochs_trained=n_epochs_trained,
    ), best_val_loss


# ---------------------------------------------------------------------------
# Class weight computation (pure)
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Args:
        y:         integer label array.
        n_classes: number of classes.

    Returns:
        Float tensor of shape (n_classes,).
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = 1.0 / np.where(counts == 0, 1.0, counts)
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)
