"""
src/training/run_gnn.py
========================
Entry point for the Temporal GNN (Step 4).

Runs StratifiedGroupKFold CV on the 3-node graph sequences and logs results
to MLflow experiment 'gnn/'. Ablations A1, A3, A4, A6 are run via config flags.

Ablation map:
    A1: use_temporal=False       — cross-sectional GNN (no temporal attention)
    A3: use_delta=False          — graphs must be re-built without delta columns
                                   (graph_builder.py --no-delta flag, deferred to Step 5)
    A4: use_time_encoding=False  — temporal attention without sinusoidal Δt encoding
    A6: topology=2node           — HD-GLIO-AUTO 2-node graphs

A2 and A5 reference Step 3 LSTM and temporal-only LightGBM results respectively
(already computed — no re-training needed).

Pipeline per fold (identical to run_lgbm_baseline.py pattern):
    1. StratifiedGroupKFold splits (same seed=42 → same splits as Step 3)
    2. fit_transform_fold (StandardScaler on train only)
    3. select_features_fold_anchored_cached (reuse Step 3 cache)
    4. Collate patient sequences from graph .pt files
    5. Train TumorTemporalGNN with AdamW + ReduceLROnPlateau + early stopping
    6. Evaluate on test fold, log to MLflow

Side effects are confined to main(). All model logic lives in gnn.py,
temporal_attention.py, tumor_gnn.py (pure modules).

Output artifacts:
    data/processed/gnn/gnn_{ablation}_results.json
    MLflow experiment: gnn/

Usage:
    uv run python src/training/run_gnn.py
    uv run python src/training/run_gnn.py --ablation A1
    uv run python src/training/run_gnn.py --fast
    uv run python src/training/run_gnn.py --topology 2node --ablation A6
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

from src.graphs.graph_builder import (
    GRAPHS_DIR,
    PatientGraphSequence,
    TopologyType,
)
from src.models.tumor_gnn import GNNConfig, TumorTemporalGNN
from src.training.cross_validation import build_cv_splits
from src.training.metrics import FoldMetrics, aggregate_cv_results, compute_metrics
from src.training.training_utils import (
    build_run_info,
    fit_transform_fold,
    load_random_config,
    select_features_fold_anchored_cached,
)
from src.utils.lumiere_io import build_full_feature_set, print_section

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PARQUET_PATH = Path("data/processed/preprocessing/dataset_engineered.parquet")
OUTPUT_DIR = Path("data/processed/gnn")
GNN_CONFIG_PATH = "configs/gnn.yaml"
RANDOM_STATE_PATH = "configs/random_state.yaml"
MLFLOW_EXPERIMENT = "gnn"

AblationType = Literal["full", "A1", "A3", "A4", "A6"]
ABLATIONS_ALL: list[AblationType] = ["full", "A1", "A4"]  # A6 via --topology 2node


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class GNNFoldResult:
    fold: int
    ablation: str
    topology: str
    metrics: FoldMetrics
    n_train_patients: int
    n_test_patients: int
    n_features: int
    best_hidden: int
    best_heads: int
    best_lr: float
    n_epochs_trained: int


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _load_gnn_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Sequence loader — loads .pt files for a given patient list
# ---------------------------------------------------------------------------
def load_sequences(
    patient_ids: list[str],
    topology: TopologyType,
) -> dict[str, PatientGraphSequence]:
    """
    Load PatientGraphSequence .pt files for the given patient IDs.

    Returns:
        Dict mapping patient_id → PatientGraphSequence.

    Raises:
        FileNotFoundError: if any patient's .pt file is missing.
    """
    graphs_dir = GRAPHS_DIR / topology
    sequences: dict[str, PatientGraphSequence] = {}
    missing = []
    for pid in patient_ids:
        pt_path = graphs_dir / f"{pid}.pt"
        if not pt_path.exists():
            missing.append(pid)
            continue
        sequences[pid] = torch.load(pt_path, weights_only=False)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} graph files missing in {graphs_dir}: {missing[:5]}. "
            "Run graph_builder.py first."
        )
    return sequences


# ---------------------------------------------------------------------------
# Sequence collation — variable-length → padded batch tensors
# ---------------------------------------------------------------------------
def collate_patient_sequences(
    sequences: list[PatientGraphSequence],
    feature_cols: list[str],
    all_feature_cols: list[str],
    X_scaled: np.ndarray,
    patient_row_map: dict[str, list[int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a list of PatientGraphSequence objects into padded batch tensors.

    Node features are sourced from the normalised parquet (X_scaled) rather
    than the raw .pt files. This ensures that:
    - Normalisation (StandardScaler fit on train fold) is correctly applied.
    - Feature selection (feature_cols) is correctly applied.
    - The graph structure (edge_index, edge_attr) from .pt files is reused.

    Args:
        sequences:       list of PatientGraphSequence for this batch.
        feature_cols:    selected feature columns for this fold.
        all_feature_cols: full feature column list (aligned with X_scaled columns).
        X_scaled:        normalised feature matrix for the fold split (train or test).
        patient_row_map: maps patient_id → list of row indices in X_scaled.
        device:          torch device.

    Returns:
        x_seq:         [batch, max_T, n_nodes, n_node_features]
        edge_index:    [2, n_edges]  — shared topology
        edge_attr_seq: [batch, max_T, n_edges, edge_dim]
        intervals:     [batch, max_T]
        seq_lengths:   [batch]
        labels:        [batch]
    """
    # Determine node count and edge structure from first graph
    sample_graph: Data = sequences[0].graphs[0]
    n_nodes = sample_graph.x.shape[0]
    n_edges = sample_graph.edge_index.shape[1]
    edge_dim = sample_graph.edge_attr.shape[1]
    edge_index = sample_graph.edge_index.to(device)

    # Identify node-specific and scalar feature indices in feature_cols
    # Node prefixes: NC_, CE_, ED_ (or NE_, CE_ for 2node)
    topology = sequences[0].topology
    node_prefixes = ["NC", "CE", "ED"] if topology == "3node" else ["NE", "CE"]

    # Build per-node column index maps within feature_cols
    col_set = set(feature_cols)
    node_col_indices: dict[str, list[int]] = {}
    for prefix in node_prefixes:
        node_col_indices[prefix] = [
            i for i, c in enumerate(feature_cols)
            if c.startswith(f"{prefix}_") and c in col_set
        ]

    # Scalar features: cols in feature_cols that don't belong to any node prefix
    all_node_cols: set[int] = set(idx for idxs in node_col_indices.values() for idx in idxs)
    scalar_indices = [i for i in range(len(feature_cols)) if i not in all_node_cols]

    # n_node_features: per-node radiomic + shared scalar
    n_per_node = max(len(v) for v in node_col_indices.values()) + len(scalar_indices)

    max_t = max(seq.n_timepoints for seq in sequences)
    batch_size = len(sequences)

    x_seq = torch.zeros(batch_size, max_t, n_nodes, n_per_node, device=device)
    edge_attr_seq = torch.zeros(batch_size, max_t, n_edges, edge_dim, device=device)
    intervals = torch.zeros(batch_size, max_t, device=device)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    for b, seq in enumerate(sequences):
        t_len = seq.n_timepoints
        seq_lengths[b] = t_len
        labels[b] = seq.label_sequence[-1]  # target = last timepoint label

        row_indices = patient_row_map[seq.patient_id]
        assert len(row_indices) == t_len, (
            f"Patient {seq.patient_id}: row_indices length {len(row_indices)} "
            f"≠ n_timepoints {t_len}"
        )

        for t, (row_idx, graph) in enumerate(zip(row_indices, seq.graphs)):
            feat_row = X_scaled[row_idx]  # [n_all_features] normalised

            # Build node features from selected columns
            for node_i, prefix in enumerate(node_prefixes):
                node_vals = feat_row[[feature_cols.index(c)
                                      for c in feature_cols
                                      if c.startswith(f"{prefix}_") and c in col_set]]
                scalar_vals = feat_row[scalar_indices]
                x_seq[b, t, node_i, :len(node_vals)] = torch.tensor(
                    node_vals, dtype=torch.float, device=device
                )
                x_seq[b, t, node_i, len(node_vals):len(node_vals) + len(scalar_vals)] = (
                    torch.tensor(scalar_vals, dtype=torch.float, device=device)
                )

            edge_attr_seq[b, t] = graph.edge_attr.to(device)
            intervals[b, t] = graph.interval_weeks

    return x_seq, edge_index, edge_attr_seq, intervals, seq_lengths, labels


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------
def _compute_class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = 1.0 / np.where(counts == 0, 1.0, counts)
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop (pure — returns result)
# ---------------------------------------------------------------------------
def _train_fold(
    train_sequences: list[PatientGraphSequence],
    val_sequences: list[PatientGraphSequence],
    test_sequences: list[PatientGraphSequence],
    feature_cols: list[str],
    all_feature_cols: list[str],
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    train_patient_row_map: dict[str, list[int]],
    val_patient_row_map: dict[str, list[int]],
    test_patient_row_map: dict[str, list[int]],
    fold: int,
    ablation: str,
    gnn_cfg: GNNConfig,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> GNNFoldResult:
    """Train TumorTemporalGNN on one CV fold with early stopping on val loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    topology: TopologyType = train_sequences[0].topology

    # Collate batches
    train_batch = collate_patient_sequences(
        train_sequences, feature_cols, all_feature_cols,
        X_train_scaled, train_patient_row_map, device,
    )
    val_batch = collate_patient_sequences(
        val_sequences, feature_cols, all_feature_cols,
        X_val_scaled, val_patient_row_map, device,
    )
    test_batch = collate_patient_sequences(
        test_sequences, feature_cols, all_feature_cols,
        X_test_scaled, test_patient_row_map, device,
    )

    # Update in_channels from actual data
    gnn_cfg = GNNConfig(
        **{**asdict(gnn_cfg), "in_channels": train_batch[0].shape[-1]}
    )
    model = TumorTemporalGNN(gnn_cfg).to(device)

    y_train_np = train_batch[5].cpu().numpy()
    class_weights = _compute_class_weights(y_train_np).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimiser = torch.optim.AdamW(model.parameters(), lr=gnn_cfg.learning_rate if hasattr(gnn_cfg, "learning_rate") else 1e-3)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", patience=patience // 4, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    n_epochs_trained = 0

    for epoch in range(max_epochs):
        model.train()
        x_seq, edge_index, edge_attr_seq, intervals, seq_lengths, labels = train_batch
        optimiser.zero_grad()
        logits = model(x_seq, edge_index, edge_attr_seq, intervals, seq_lengths)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        # Validation
        model.eval()
        with torch.no_grad():
            x_seq_v, ei_v, ea_v, iv_v, sl_v, lv_v = val_batch
            val_logits = model(x_seq_v, ei_v, ea_v, iv_v, sl_v)
            val_loss = criterion(val_logits, lv_v).item()

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

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test fold
    model.eval()
    with torch.no_grad():
        x_seq_t, ei_t, ea_t, iv_t, sl_t, lt_t = test_batch
        test_logits = model(x_seq_t, ei_t, ea_t, iv_t, sl_t)
        y_pred = test_logits.argmax(dim=-1).cpu().numpy()
        y_proba = torch.softmax(test_logits, dim=-1).cpu().numpy()
        y_true = lt_t.cpu().numpy()

    fold_metrics = compute_metrics(
        fold=fold,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    return GNNFoldResult(
        fold=fold,
        ablation=ablation,
        topology=topology,
        metrics=fold_metrics,
        n_train_patients=len(train_sequences),
        n_test_patients=len(test_sequences),
        n_features=train_batch[0].shape[-1],
        best_hidden=gnn_cfg.hidden,
        best_heads=gnn_cfg.heads,
        best_lr=getattr(gnn_cfg, "learning_rate", 1e-3),
        n_epochs_trained=n_epochs_trained,
    )


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------
def _log_fold_metrics(fm: FoldMetrics, ablation: str) -> None:
    p = f"{ablation}_fold_{fm.fold}"
    mlflow.log_metric(f"{p}_macro_f1",          fm.macro_f1)
    mlflow.log_metric(f"{p}_mcc",               fm.mcc)
    mlflow.log_metric(f"{p}_auroc_progressive",  fm.auroc_progressive)
    mlflow.log_metric(f"{p}_auroc_stable",       fm.auroc_stable)
    mlflow.log_metric(f"{p}_auroc_response",     fm.auroc_response)
    mlflow.log_metric(f"{p}_prauc_progressive",  fm.prauc_progressive)
    mlflow.log_metric(f"{p}_prauc_stable",       fm.prauc_stable)
    mlflow.log_metric(f"{p}_prauc_response",     fm.prauc_response)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    ablation: AblationType = "full",
    topology: TopologyType = "3node",
    fast: bool = False,
    verbose: bool = False,
) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print_section(f"Step 4 — Temporal GNN (ablation={ablation}, topology={topology})")
    if fast:
        print("  ⚠️  FAST MODE — 2 folds, 10 epochs max. Smoke test only.")

    seed, n_jobs = load_random_config(RANDOM_STATE_PATH)
    cfg_raw = _load_gnn_config(GNN_CONFIG_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"{PARQUET_PATH} not found. Run features_builder.py (Step 2) first."
        )

    df = pd.read_parquet(PARQUET_PATH)
    all_feature_cols = build_full_feature_set(df)
    y = df["target_encoded"].values
    groups = df["Patient"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  n_effective: {len(df)} | n_patients: {groups.nunique()}")

    run_info = build_run_info(
        seed=seed,
        parquet_path=str(PARQUET_PATH),
        n_rows=int(df.shape[0]),
        n_patients=int(groups.nunique()),
        script_path=str(Path(__file__)),
    )

    cv_splits = build_cv_splits(
        X=df[all_feature_cols],
        y=pd.Series(y),
        groups=groups,
        n_splits=2 if fast else 5,
        seed=seed,
    )

    # Build ablation-specific GNNConfig flags
    ablation_flags: dict[str, bool] = {
        "use_temporal":      ablation != "A1",
        "use_time_encoding": ablation != "A4",
    }

    max_epochs = 10 if fast else cfg_raw.get("max_epochs", 200)
    patience = 5  if fast else cfg_raw.get("patience", 20)
    batch_size = cfg_raw.get("batch_size", 8)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    fold_metrics_list: list[FoldMetrics] = []
    fold_results_raw: list[dict] = []

    with mlflow.start_run(run_name=f"gnn_{ablation}_{topology}"):
        mlflow.log_params({
            "ablation": ablation,
            "topology": topology,
            "seed": seed,
            "fast": fast,
            **ablation_flags,
        })

        for fold_split in cv_splits.folds:
            print_section(f"  Fold {fold_split.fold}")

            X_train_df = df.iloc[fold_split.train_idx]
            X_test_df  = df.iloc[fold_split.test_idx]
            y_train    = y[fold_split.train_idx]
            y_test     = y[fold_split.test_idx]

            # Normalise
            X_train_scaled, X_test_scaled = fit_transform_fold(
                X_train_df, X_test_df, all_feature_cols
            )

            # Feature selection (reuses Step 3 cache)
            X_train_df_scaled = pd.DataFrame(X_train_scaled, columns=all_feature_cols)
            selection = select_features_fold_anchored_cached(
                X_train=X_train_df_scaled,
                y_train=y_train,
                fold=fold_split.fold,
                seed=seed,
                fast=fast,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            feature_cols = selection.full_feature_set

            # Load graph sequences for train/test patients
            train_patients = list(X_train_df["Patient"].unique())
            test_patients  = list(X_test_df["Patient"].unique())

            train_seqs_dict = load_sequences(train_patients, topology)
            test_seqs_dict  = load_sequences(test_patients, topology)

            # Val split: 10% of train patients (random, patient-level)
            rng = np.random.default_rng(seed)
            n_val_pat = max(1, round(len(train_patients) * 0.1))
            val_patients = list(rng.choice(train_patients, size=n_val_pat, replace=False))
            train_patients_inner = [p for p in train_patients if p not in set(val_patients)]

            # Build row index maps: patient_id → sorted row indices in X_*_scaled
            def _build_row_map(df_split: pd.DataFrame, scaled: np.ndarray, patients: list[str]) -> dict:
                row_map: dict[str, list[int]] = {}
                for pid in patients:
                    mask = (df_split["Patient"] == pid).values
                    indices_in_split = np.where(mask)[0].tolist()
                    # Sort by scan_index (chronological)
                    scan_idx = df_split.iloc[indices_in_split]["scan_index"].values
                    order = np.argsort(scan_idx)
                    row_map[pid] = [indices_in_split[o] for o in order]
                return row_map

            train_row_map = _build_row_map(X_train_df.reset_index(drop=True), X_train_scaled, train_patients_inner)
            val_row_map   = _build_row_map(X_train_df.reset_index(drop=True), X_train_scaled, val_patients)
            test_row_map  = _build_row_map(X_test_df.reset_index(drop=True), X_test_scaled, test_patients)

            train_seqs = [train_seqs_dict[p] for p in train_patients_inner if p in train_seqs_dict]
            val_seqs   = [train_seqs_dict[p] for p in val_patients       if p in train_seqs_dict]
            test_seqs  = [test_seqs_dict[p]  for p in test_patients       if p in test_seqs_dict]

            if not train_seqs or not test_seqs:
                print(f"  Fold {fold_split.fold}: skipping — insufficient sequences")
                continue

            print(
                f"  Patients — train: {len(train_seqs)} | "
                f"val: {len(val_seqs)} | test: {len(test_seqs)}"
            )

            # Build GNNConfig (in_channels resolved inside _train_fold)
            gnn_cfg = GNNConfig(
                in_channels=0,  # placeholder, resolved in collation
                hidden=cfg_raw.get("hidden", [32])[0],
                heads=cfg_raw.get("heads", [1])[0],
                dropout=cfg_raw.get("dropout", [0.2])[0],
                **ablation_flags,
            )

            result = _train_fold(
                train_sequences=train_seqs,
                val_sequences=val_seqs if val_seqs else test_seqs,
                test_sequences=test_seqs,
                feature_cols=feature_cols,
                all_feature_cols=all_feature_cols,
                X_train_scaled=X_train_scaled,
                X_val_scaled=X_train_scaled,   # val uses same scaled array, different rows
                X_test_scaled=X_test_scaled,
                train_patient_row_map=train_row_map,
                val_patient_row_map=val_row_map if val_seqs else test_row_map,
                test_patient_row_map=test_row_map,
                fold=fold_split.fold,
                ablation=ablation,
                gnn_cfg=gnn_cfg,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                seed=seed,
                device=device,
            )

            print(
                f"  Fold {fold_split.fold}: "
                f"macro_f1={result.metrics.macro_f1:.4f} | "
                f"mcc={result.metrics.mcc:.4f} | "
                f"epochs={result.n_epochs_trained}"
            )

            _log_fold_metrics(result.metrics, ablation)
            fold_metrics_list.append(result.metrics)
            fold_results_raw.append(asdict(result))

        if fold_metrics_list:
            aggregated = aggregate_cv_results(fold_metrics_list)
            print_section("Aggregated Results")
            print(f"  macro_f1 : {aggregated.macro_f1_mean:.4f} ± {aggregated.macro_f1_std:.4f}")
            print(f"  mcc      : {aggregated.mcc_mean:.4f} ± {aggregated.mcc_std:.4f}")

            report = {
                "schema_version": "gnn.v1",
                "model": "TumorTemporalGNN",
                "ablation": ablation,
                "topology": topology,
                "seed": seed,
                "run_info": run_info,
                "fold_results": fold_results_raw,
                "aggregated": asdict(aggregated),
            }
            report_path = OUTPUT_DIR / f"gnn_{ablation}_{topology}_results.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(str(report_path))
            print(f"\n  Report → {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal GNN (Step 4)")
    parser.add_argument(
        "--ablation",
        choices=["full", "A1", "A3", "A4", "A6"],
        default="full",
        help="Ablation variant to run.",
    )
    parser.add_argument(
        "--topology",
        choices=["3node", "2node"],
        default="3node",
        help="Graph topology (2node = HD-GLIO-AUTO, ablation A6).",
    )
    parser.add_argument("--fast", action="store_true",
                        help="Smoke test: 2 folds, 10 epochs max.")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose feature selection output.")
    args = parser.parse_args()
    main(
        ablation=args.ablation,
        topology=args.topology,
        fast=args.fast,
        verbose=args.verbose,
    )
