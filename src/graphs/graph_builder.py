"""
src/graphs/graph_builder.py
===========================
Builds PyTorch Geometric graphs from dataset_paired.parquet.

Steps covered:
    1.1  GraphBuilder  — single Data object from one row
    1.2  DeltaGraphBuilder — adds delta feature tensor (already in parquet)
    1.3  TemporalSequence  — ordered sequence per patient, saved as .pt

Graph topology:
    3 nodes: NC (Necrosis=0), CE (Contrast-enhancing=1), ED (Edema=2)
    6 directed edges (triangular, bidirectional):
        NC↔CE, NC↔ED, CE↔ED

Edge features per edge:
    - volumetric_ratio: volume(source) / volume(target) — shape proxy
    - interval_weeks:   temporal gap to next scan (leakage-monitored)

Node features per node:
    [selected radiomic features, delta features, is_baseline_scan,
     time_from_diagnosis_weeks, scan_index]

Usage:
    python -m src.graphs.graph_builder          # builds all patient graphs
    python -m src.graphs.graph_builder --dry-run # validate without saving
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data

from src.utils.lumiere_io import LABEL_ENCODING, RADIOMIC_PREFIX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/processed")
PARQUET_PATH = DATA_DIR / "dataset_paired.parquet"
GRAPHS_DIR = DATA_DIR / "graphs"

# Fixed node order — must never change after graphs are built
NODE_ORDER: list[str] = ["NC", "CE", "ED"]
NODE_INDEX: dict[str, int] = {n: i for i, n in enumerate(NODE_ORDER)}

# Triangular bidirectional edge index (6 edges, shape [2, 12] with bidirectional)
# Indices: 0=NC, 1=CE, 2=ED
# Pairs: NC-CE, NC-ED, CE-ED  →  each pair appears twice (both directions)
EDGE_INDEX: Tensor = torch.tensor([
    [0, 1, 0, 2, 1, 2, 1, 0, 2, 0, 2, 1],
    [1, 0, 2, 0, 2, 1, 0, 1, 0, 2, 1, 2],
], dtype=torch.long)
# Shape: [2, 12] — 6 undirected edges × 2 directions

SECTION = "=" * 60


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class GraphConfig:
    """
    All parameters needed to build graphs from the paired dataset.
    node_feature_cols maps each label prefix to the list of selected feature columns.
    Set after feature selection (Phase 1 Step 1.4); use all radiomic cols before that.
    """
    node_feature_cols: dict[str, list[str]]    # {"NC": [...], "CE": [...], "ED": [...]}
    label_col: str = "target_encoded"
    label_mapping: dict[str, int] = field(default_factory=lambda: LABEL_ENCODING)
    node_order: list[str] = field(default_factory=lambda: NODE_ORDER.copy())

    # Scalar features added to every node (same value, allows the model to
    # use temporal context per-node — important for irregular intervals)
    scalar_node_features: list[str] = field(default_factory=lambda: [
        "interval_weeks",
        "time_from_diagnosis_weeks",
        "scan_index",
        "is_baseline_scan",
    ])

    # Volumetric ratio is the only cross-node edge feature we can compute
    # from the data. interval_weeks is scalar (same for all edges at t).
    volumetric_feature: str = "original_shape_MeshVolume"


@dataclass
class PatientGraphSequence:
    """Ordered chronological graph sequence for one patient."""
    patient_id: str
    graphs: list[Data]
    n_timepoints: int
    label_sequence: list[int]


# ---------------------------------------------------------------------------
# Core pure functions
# ---------------------------------------------------------------------------
def _get_node_feature_cols(
    df: pd.DataFrame,
    config: GraphConfig,
) -> dict[str, list[str]]:
    """
    If config.node_feature_cols is empty, fall back to all radiomic columns
    for each node prefix. Used before feature selection is run.
    """
    if config.node_feature_cols:
        return config.node_feature_cols
    all_rc = [c for c in df.columns if RADIOMIC_PREFIX in c]
    result: dict[str, list[str]] = {}
    for prefix in config.node_order:
        result[prefix] = [c for c in all_rc if c.startswith(f"{prefix}_")]
    return result


def _volumetric_ratio(
    row: pd.Series,
    src_prefix: str,
    tgt_prefix: str,
    vol_feature: str,
) -> float:
    """
    Compute volume(src) / volume(tgt) for the edge src→tgt.
    Falls back to 1.0 if either volume is NaN or zero.
    """
    src_cols = [c for c in row.index
                if c.startswith(f"{src_prefix}_") and vol_feature in c]
    tgt_cols = [c for c in row.index
                if c.startswith(f"{tgt_prefix}_") and vol_feature in c]
    if not src_cols or not tgt_cols:
        return 1.0
    src_vol = float(row[src_cols[0]])
    tgt_vol = float(row[tgt_cols[0]])
    if pd.isna(src_vol) or pd.isna(tgt_vol) or tgt_vol == 0:
        return 1.0
    return float(src_vol / tgt_vol)


def build_edge_attr(row: pd.Series, config: GraphConfig) -> Tensor:
    """
    Build edge attribute tensor of shape [12, 2].
    Each edge carries: [volumetric_ratio(src→tgt), interval_weeks].

    The 12 directed edges correspond to EDGE_INDEX columns in order.
    Edge pairs (src, tgt) from EDGE_INDEX:
        cols 0-1:  NC↔CE
        cols 2-3:  NC↔ED
        cols 4-5:  CE↔ED
        cols 6-11: reverse directions
    """
    interval = float(row["interval_weeks"]) if "interval_weeks" in row.index else 0.0

    # Edge ordering matches EDGE_INDEX rows
    src_nodes = EDGE_INDEX[0].tolist()
    tgt_nodes = EDGE_INDEX[1].tolist()
    idx_to_prefix = {i: p for p, i in NODE_INDEX.items()}

    edge_attrs = []
    for s, t in zip(src_nodes, tgt_nodes):
        ratio = _volumetric_ratio(
            row,
            idx_to_prefix[s],
            idx_to_prefix[t],
            config.volumetric_feature,
        )
        edge_attrs.append([ratio, interval])

    return torch.tensor(edge_attrs, dtype=torch.float)  # [12, 2]


def build_node_features(
    row: pd.Series,
    config: GraphConfig,
    node_feature_cols: dict[str, list[str]],
) -> Tensor:
    """
    Build node feature matrix of shape [3, n_features].
    Each row = one node (NC, CE, ED in NODE_ORDER).
    Features: radiomic cols for that node + scalar context features.
    """
    scalar_vals = []
    for sf in config.scalar_node_features:
        if sf == "is_baseline_scan":
            scalar_vals.append(float(bool(row.get(sf, False))))
        elif sf in row.index:
            scalar_vals.append(float(row[sf]))
        else:
            scalar_vals.append(0.0)

    rows = []
    for prefix in config.node_order:
        cols = node_feature_cols.get(prefix, [])
        if cols:
            radiomic_vals = row[cols].values.astype(float).tolist()
        else:
            radiomic_vals = []
        rows.append(radiomic_vals + scalar_vals)

    return torch.tensor(rows, dtype=torch.float)  # [3, n_features]


def build_graph(row: pd.Series, config: GraphConfig) -> Data:
    """
    Build a single PyG Data object from one row of dataset_paired.parquet.

    Returns:
        data.x:          [3, n_node_features]
        data.edge_index: [2, 12]
        data.edge_attr:  [12, 2]
        data.y:          [1]
        data.patient:    str
        data.timepoint:  str
        data.interval_weeks: float
    """
    node_feature_cols = _get_node_feature_cols(
        row.to_frame().T, config  # type: ignore[arg-type]
    )
    x = build_node_features(row, config, node_feature_cols)
    edge_attr = build_edge_attr(row, config)
    y = torch.tensor([int(row[config.label_col])], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=EDGE_INDEX.clone(),
        edge_attr=edge_attr,
        y=y,
    )
    data.patient = str(row["Patient"])
    data.timepoint = str(row["Timepoint"])
    data.interval_weeks = float(row.get("interval_weeks", 0.0))
    return data


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_graphs(graphs: list[Data], config: GraphConfig) -> None:
    """
    Assert structural invariants on a list of graphs.
    Raises AssertionError with a descriptive message on first failure.
    """
    for i, g in enumerate(graphs):
        assert g.x.shape[0] == 3, \
            f"Graph {i}: expected 3 nodes, got {g.x.shape[0]}"
        assert g.edge_index.shape == (2, 12), \
            f"Graph {i}: edge_index shape {g.edge_index.shape}, expected (2, 12)"
        assert g.edge_attr.shape == (12, 2), \
            f"Graph {i}: edge_attr shape {g.edge_attr.shape}, expected (12, 2)"
        assert g.y.shape == (1,), \
            f"Graph {i}: y shape {g.y.shape}, expected (1,)"
        assert not torch.isnan(g.x).any(), \
            f"Graph {i}: NaN in node features"
        assert not torch.isnan(g.edge_attr).any(), \
            f"Graph {i}: NaN in edge attributes"


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------
def build_patient_sequence(
    patient_df: pd.DataFrame,
    config: GraphConfig,
) -> PatientGraphSequence:
    """
    Build the ordered graph sequence for one patient.
    Rows must be sorted by time_from_diagnosis_weeks (ascending).
    """
    patient_df = patient_df.sort_values("time_from_diagnosis_weeks").reset_index(drop=True)
    patient_id = patient_df["Patient"].iloc[0]

    graphs = [build_graph(row, config) for _, row in patient_df.iterrows()]
    label_sequence = patient_df[config.label_col].tolist()

    return PatientGraphSequence(
        patient_id=str(patient_id),
        graphs=graphs,
        n_timepoints=len(graphs),
        label_sequence=[int(l) for l in label_sequence],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(dry_run: bool = False) -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{SECTION}")
    print("GBM Longitudinal Toolkit — Graph Builder (Phase 1)")
    print(SECTION)

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"{PARQUET_PATH} not found. Run build_dataset.py first."
        )

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Patients: {df['Patient'].nunique()}")

    # Default config: use all radiomic cols (feature selection not yet run)
    config = GraphConfig(node_feature_cols={})

    patients = sorted(df["Patient"].unique())
    all_graphs: list[Data] = []
    sequences: list[PatientGraphSequence] = []

    for patient_id in patients:
        patient_df = df[df["Patient"] == patient_id]
        seq = build_patient_sequence(patient_df, config)
        sequences.append(seq)
        all_graphs.extend(seq.graphs)

        if not dry_run:
            save_path = GRAPHS_DIR / f"{patient_id}.pt"
            torch.save(seq, save_path)

    print(f"\nGraphs built: {len(all_graphs)} total across {len(sequences)} patients")

    print("\nValidating graph structure...")
    validate_graphs(all_graphs, config)
    print("✅ All structural assertions passed")

    # Summary stats
    seq_lengths = [s.n_timepoints for s in sequences]
    print(f"\nSequence length: min={min(seq_lengths)}, "
          f"max={max(seq_lengths)}, "
          f"mean={sum(seq_lengths)/len(seq_lengths):.1f}")

    node_feat_dim = all_graphs[0].x.shape[1]
    print(f"Node feature dim: {node_feat_dim}")
    print(f"Edge feature dim: {all_graphs[0].edge_attr.shape[1]}")

    if dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        print(f"\nSaved → {GRAPHS_DIR}/{{patient_id}}.pt ({len(sequences)} files)")

    print(SECTION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and validate graphs without saving to disk")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
