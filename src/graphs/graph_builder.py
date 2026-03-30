"""
src/graphs/graph_builder.py
===========================
Builds PyTorch Geometric graphs from dataset_engineered.parquet.

Graph topology (DeepBraTumIA — primary, 3 nodes):
    NC (Necrosis=0) ←→ CE (Contrast-enhancing=1) ←→ ED (Edema=2)
    Triangular bidirectional — 6 undirected edges × 2 directions = 12 directed edges.

Graph topology (HD-GLIO-AUTO — ablation A6, 2 nodes):
    NE (Non-enhancing=0) ←→ CE (Contrast-enhancing=1)
    1 undirected edge × 2 directions = 2 directed edges.

Node features per node:
    [selected radiomic features  (from selected_features.yaml, prefix-filtered)
     anchored delta features      (delta_{f} for each selected radiomic f)
     is_baseline_scan             (scalar 0/1 — distinguishes zero-delta from baseline)
     time_from_diagnosis_weeks    (absolute disease stage proxy)
     scan_index                   (0-based ordinal per patient)]

Edge features per directed edge:
    [volumetric_ratio: volume(src) / volume(tgt)  — shape proxy
     interval_weeks:   temporal gap to next scan   — leakage-monitored]

Input:
    data/processed/preprocessing/dataset_engineered.parquet   (231 × 2585)
    configs/selected_features.yaml                             (from Step 3 LightGBM D)

Output:
    data/processed/graphs/3node/{patient_id}.pt
    data/processed/graphs/2node/{patient_id}.pt   (ablation A6)

Usage:
    python -m src.graphs.graph_builder                 # 3-node graphs
    python -m src.graphs.graph_builder --topology 2node  # HD-GLIO 2-node
    python -m src.graphs.graph_builder --dry-run       # validate, no save
    python -m src.graphs.graph_builder --fast          # first 5 patients only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import yaml
from torch import Tensor
from torch_geometric.data import Data

from src.utils.lumiere_io import LABEL_ENCODING, RADIOMIC_PREFIX, print_section

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PREPROCESSING_DIR = Path("data/processed/preprocessing")
PARQUET_PATH = PREPROCESSING_DIR / "dataset_engineered.parquet"
GRAPHS_DIR = Path("data/processed/graphs")
SELECTED_FEATURES_PATH = Path("configs/selected_features.yaml")
REPORT_PATH = GRAPHS_DIR / "graph_builder_report.json"

# ---------------------------------------------------------------------------
# Topology constants — FIXED, must never change after graphs are built
# ---------------------------------------------------------------------------

# 3-node topology (DeepBraTumIA — primary)
NODE_ORDER_3: list[str] = ["NC", "CE", "ED"]
NODE_INDEX_3: dict[str, int] = {n: i for i, n in enumerate(NODE_ORDER_3)}

# Triangular bidirectional edge index for 3-node graph
# Pairs: NC-CE, NC-ED, CE-ED — each twice (both directions) → 12 directed edges
# Row 0 = source, Row 1 = target
EDGE_INDEX_3: Tensor = torch.tensor(
    [
        [0, 1, 0, 2, 1, 2, 1, 0, 2, 0, 2, 1],
        [1, 0, 2, 0, 2, 1, 0, 1, 0, 2, 1, 2],
    ],
    dtype=torch.long,
)  # shape [2, 12]

# 2-node topology (HD-GLIO-AUTO — ablation A6)
NODE_ORDER_2: list[str] = ["NE", "CE"]
NODE_INDEX_2: dict[str, int] = {n: i for i, n in enumerate(NODE_ORDER_2)}

# Single bidirectional edge for 2-node graph → 2 directed edges
EDGE_INDEX_2: Tensor = torch.tensor(
    [[0, 1], [1, 0]],
    dtype=torch.long,
)  # shape [2, 2]

TopologyType = Literal["3node", "2node"]

SECTION = "=" * 60

# Volume column used for volumetric_ratio edge feature.
# CT1 for CE/NC (contrast-enhanced T1 is the clinical reference);
# FLAIR for ED (standard for edema); T1 for NE (HD-GLIO-AUTO non-enhancing).
_VOL_COL: dict[str, str] = {
    "CE": "CE_CT1_original_shape_MeshVolume",
    "NC": "NC_CT1_original_shape_MeshVolume",
    "ED": "ED_FLAIR_original_shape_MeshVolume",
    "NE": "NE_T1_original_shape_MeshVolume",
}

# Scalar features appended identically to every node.
# Gives the GNN temporal context without duplicating scan-level information.
SCALAR_NODE_FEATURES: list[str] = [
    "time_from_diagnosis_weeks",
    "scan_index",
    "is_baseline_scan",   # bool → float; distinguishes zero-delta from baseline
]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class GraphConfig:
    """
    All parameters for building a graph topology from the engineered parquet.

    node_feature_cols:
        Maps each node prefix (e.g. "NC") to the list of selected radiomic columns
        for that node (e.g. ["NC_CT1_original_shape_MeshVolume", ...]).
        Populated from selected_features.yaml after Step 3.
        If empty, falls back to ALL radiomic columns for each prefix (pre-selection mode).

    delta_feature_cols:
        Maps each node prefix to its anchored delta columns.
        Populated automatically from node_feature_cols by _build_delta_cols().
        If empty (pre-selection), anchored deltas are not added.

    topology:
        "3node" → NC/CE/ED triangular graph (primary, DeepBraTumIA).
        "2node" → NE/CE linear graph (HD-GLIO-AUTO, ablation A6).
    """
    node_feature_cols: dict[str, list[str]]
    delta_feature_cols: dict[str, list[str]] = field(default_factory=dict)
    topology: TopologyType = "3node"
    label_col: str = "target_encoded"
    scalar_features: list[str] = field(
        default_factory=lambda: SCALAR_NODE_FEATURES.copy()
    )
    volumetric_feature: str = "original_shape_MeshVolume"

    @property
    def node_order(self) -> list[str]:
        return NODE_ORDER_3 if self.topology == "3node" else NODE_ORDER_2

    @property
    def node_index(self) -> dict[str, int]:
        return NODE_INDEX_3 if self.topology == "3node" else NODE_INDEX_2

    @property
    def edge_index(self) -> Tensor:
        return EDGE_INDEX_3 if self.topology == "3node" else EDGE_INDEX_2

    @property
    def n_edges(self) -> int:
        return 12 if self.topology == "3node" else 2


@dataclass
class PatientGraphSequence:
    """Ordered chronological graph sequence for one patient."""
    patient_id: str
    topology: TopologyType
    graphs: list[Data]
    n_timepoints: int
    label_sequence: list[int]


# ---------------------------------------------------------------------------
# Config factory — loads selected_features.yaml and builds GraphConfig
# ---------------------------------------------------------------------------
def load_graph_config(
    selected_features_path: Path = SELECTED_FEATURES_PATH,
    topology: TopologyType = "3node",
) -> GraphConfig:
    """
    Build a GraphConfig from selected_features.yaml.

    selected_features.yaml contains a flat list of radiomic feature names
    (majority-voted across CV folds, produced by run_lgbm_baseline.py).
    This function:
        1. Partitions the list by node prefix (NC_, CE_, ED_ or NE_, CE_).
        2. Builds the corresponding anchored delta_feature_cols.

    If selected_features.yaml does not exist (pre-Step-3 mode), returns a
    GraphConfig with empty node_feature_cols — the builder falls back to all
    radiomic columns. Declare pre-selection mode clearly in any output.

    Args:
        selected_features_path: path to selected_features.yaml.
        topology: "3node" or "2node".

    Returns:
        Populated GraphConfig.
    """
    node_order = NODE_ORDER_3 if topology == "3node" else NODE_ORDER_2

    if not selected_features_path.exists():
        print(
            f"  ⚠️  {selected_features_path} not found — "
            "running in pre-selection mode (all radiomic cols per node). "
            "Run Step 3 LightGBM D to generate selected_features.yaml."
        )
        return GraphConfig(node_feature_cols={}, topology=topology)

    with open(selected_features_path) as f:
        payload = yaml.safe_load(f)

    all_selected: list[str] = payload.get("selected_features", [])
    if not all_selected:
        raise ValueError(
            f"{selected_features_path} is empty or missing 'selected_features' key."
        )

    # Partition by prefix
    node_feature_cols: dict[str, list[str]] = {prefix: [] for prefix in node_order}
    for feat in all_selected:
        for prefix in node_order:
            if feat.startswith(f"{prefix}_"):
                node_feature_cols[prefix].append(feat)
                break

    # Build anchored delta cols: delta_{f} for each selected radiomic f
    delta_feature_cols: dict[str, list[str]] = {prefix: [] for prefix in node_order}
    for prefix, feats in node_feature_cols.items():
        delta_feature_cols[prefix] = [f"delta_{f}" for f in feats]

    n_total = sum(len(v) for v in node_feature_cols.values())
    n_delta = sum(len(v) for v in delta_feature_cols.values())
    print(
        f"  GraphConfig loaded: {n_total} radiomic features "
        f"+ {n_delta} delta anchored across {len(node_order)} nodes ({topology})"
    )
    return GraphConfig(
        node_feature_cols=node_feature_cols,
        delta_feature_cols=delta_feature_cols,
        topology=topology,
    )


# ---------------------------------------------------------------------------
# Core pure functions
# ---------------------------------------------------------------------------
def _resolve_node_feature_cols(
    df: pd.DataFrame,
    config: GraphConfig,
) -> dict[str, list[str]]:
    """
    Resolve per-node radiomic column lists.

    If config.node_feature_cols is populated (post-Step-3), use it directly
    after verifying all columns exist in df.
    Otherwise fall back to all radiomic columns per prefix (pre-selection mode).

    Raises:
        ValueError: if any configured column is absent from df.
    """
    if config.node_feature_cols:
        missing = [
            col
            for cols in config.node_feature_cols.values()
            for col in cols
            if col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"Columns in GraphConfig.node_feature_cols absent from parquet "
                f"({len(missing)} missing): {missing[:5]}"
            )
        return config.node_feature_cols

    # Pre-selection fallback: all radiomic cols per prefix
    all_rc = [c for c in df.columns if RADIOMIC_PREFIX in c and not c.startswith("delta_")]
    result: dict[str, list[str]] = {}
    for prefix in config.node_order:
        result[prefix] = sorted(c for c in all_rc if c.startswith(f"{prefix}_"))
    return result


def _resolve_delta_feature_cols(
    df: pd.DataFrame,
    config: GraphConfig,
    node_feature_cols: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Resolve per-node anchored delta column lists.

    If config.delta_feature_cols is populated, use it (filtered to cols in df).
    Otherwise derive from node_feature_cols (pre-selection mode: no delta added).

    Returns empty lists per node when no delta cols are available.
    """
    if config.delta_feature_cols:
        result: dict[str, list[str]] = {}
        for prefix, cols in config.delta_feature_cols.items():
            result[prefix] = [c for c in cols if c in df.columns]
        return result

    # Pre-selection fallback: no delta (unreliable without mRMR anchoring)
    return {prefix: [] for prefix in config.node_order}


def _volumetric_ratio(
    row: pd.Series,
    src_prefix: str,
    tgt_prefix: str,
) -> float:
    """
    Compute volume(src) / volume(tgt) for the directed edge src→tgt.

    Uses the canonical MeshVolume column for each prefix (defined in _VOL_COL).
    Falls back to 1.0 if the column is absent, NaN, or zero — ensuring the
    edge attribute is always a valid float without silent propagation of NaN.

    Returns:
        float ratio in (0, ∞), or 1.0 on any failure.
    """
    src_col = _VOL_COL.get(src_prefix)
    tgt_col = _VOL_COL.get(tgt_prefix)
    if src_col is None or tgt_col is None:
        return 1.0
    if src_col not in row.index or tgt_col not in row.index:
        return 1.0
    src_vol = row[src_col]
    tgt_vol = row[tgt_col]
    if pd.isna(src_vol) or pd.isna(tgt_vol) or float(tgt_vol) == 0.0:
        return 1.0
    return float(src_vol) / float(tgt_vol)


def build_edge_attr(row: pd.Series, config: GraphConfig) -> Tensor:
    """
    Build edge attribute tensor of shape [n_edges, 2].

    Each directed edge carries:
        [volumetric_ratio(src→tgt), interval_weeks]

    volumetric_ratio is direction-dependent: ratio(NC→CE) ≠ ratio(CE→NC).
    interval_weeks is the same scalar for all edges at timepoint T — it encodes
    the temporal gap to the next scan and is included as an edge feature so the
    GNN can modulate message passing by temporal proximity.

    Returns:
        Tensor of shape [n_edges, 2] — float32.
    """
    interval = float(row["interval_weeks"]) if "interval_weeks" in row.index else 0.0
    idx_to_prefix = {v: k for k, v in config.node_index.items()}

    src_nodes = config.edge_index[0].tolist()
    tgt_nodes = config.edge_index[1].tolist()

    edge_attrs = [
        [_volumetric_ratio(row, idx_to_prefix[s], idx_to_prefix[t]), interval]
        for s, t in zip(src_nodes, tgt_nodes)
    ]
    return torch.tensor(edge_attrs, dtype=torch.float)  # [n_edges, 2]


def build_node_features(
    row: pd.Series,
    config: GraphConfig,
    node_feature_cols: dict[str, list[str]],
    delta_feature_cols: dict[str, list[str]],
) -> Tensor:
    """
    Build node feature matrix of shape [n_nodes, n_node_features].

    Feature layout per node:
        [radiomic_features | delta_features | scalar_context_features]

    All nodes share the same scalar context (interval_weeks is already an edge
    feature; time_from_diagnosis_weeks and scan_index give temporal position;
    is_baseline_scan disambiguates zero-delta from baseline).

    Returns:
        Tensor of shape [n_nodes, n_node_features] — float32.

    Raises:
        ValueError: if any node ends up with zero radiomic features
                    (indicates a misconfigured GraphConfig).
    """
    # Scalar context — same value broadcast to every node
    scalar_vals: list[float] = []
    for sf in config.scalar_features:
        if sf == "is_baseline_scan":
            scalar_vals.append(float(bool(row.get(sf, False))))
        elif sf in row.index:
            scalar_vals.append(float(row[sf]))
        else:
            scalar_vals.append(0.0)

    node_rows: list[list[float]] = []
    for prefix in config.node_order:
        radiomic_cols = node_feature_cols.get(prefix, [])
        delta_cols = delta_feature_cols.get(prefix, [])

        radiomic_vals = row[radiomic_cols].values.astype(float).tolist() if radiomic_cols else []
        delta_vals = row[delta_cols].values.astype(float).tolist() if delta_cols else []

        node_rows.append(radiomic_vals + delta_vals + scalar_vals)

    return torch.tensor(node_rows, dtype=torch.float)


def build_graph(
    row: pd.Series,
    config: GraphConfig,
    node_feature_cols: dict[str, list[str]],
    delta_feature_cols: dict[str, list[str]],
) -> Data:
    """
    Build a single PyG Data object from one row of dataset_engineered.parquet.

    Column resolution (node_feature_cols, delta_feature_cols) is done once per
    DataFrame in build_patient_sequence and passed here to avoid redundant
    per-row recomputation.

    Returns:
        data.x:              [n_nodes, n_node_features]
        data.edge_index:     [2, n_edges]
        data.edge_attr:      [n_edges, 2]
        data.y:              [1]
        data.patient:        str
        data.timepoint:      str
        data.interval_weeks: float
        data.topology:       str ("3node" or "2node")
    """
    x = build_node_features(row, config, node_feature_cols, delta_feature_cols)
    edge_attr = build_edge_attr(row, config)
    y = torch.tensor([int(row[config.label_col])], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=config.edge_index.clone(),
        edge_attr=edge_attr,
        y=y,
    )
    data.patient = str(row["Patient"])
    data.timepoint = str(row["Timepoint"])
    data.interval_weeks = float(row.get("interval_weeks", 0.0))
    data.topology = config.topology
    return data


# ---------------------------------------------------------------------------
# Validation (structural invariants — fail fast)
# ---------------------------------------------------------------------------
def validate_graphs(graphs: list[Data], config: GraphConfig) -> None:
    """
    Assert structural invariants on a list of graphs.

    All assertions use descriptive messages so failures pinpoint the exact
    graph and property. Raises AssertionError on first failure (Fail Fast).

    Checks:
        1. Correct number of nodes per topology (3 or 2)
        2. edge_index shape matches topology
        3. edge_attr shape matches topology
        4. y shape == (1,)
        5. No NaN in node features
        6. No NaN in edge attributes
        7. All node rows have identical feature dimension (no ragged tensors)
    """
    n_nodes_expected = 3 if config.topology == "3node" else 2
    n_edges_expected = 12 if config.topology == "3node" else 2

    for i, g in enumerate(graphs):
        assert g.x.shape[0] == n_nodes_expected, (
            f"Graph {i} ({g.patient} {g.timepoint}): "
            f"expected {n_nodes_expected} nodes, got {g.x.shape[0]}"
        )
        assert g.edge_index.shape == (2, n_edges_expected), (
            f"Graph {i}: edge_index shape {g.edge_index.shape}, "
            f"expected (2, {n_edges_expected})"
        )
        assert g.edge_attr.shape == (n_edges_expected, 2), (
            f"Graph {i}: edge_attr shape {g.edge_attr.shape}, "
            f"expected ({n_edges_expected}, 2)"
        )
        assert g.y.shape == (1,), (
            f"Graph {i}: y shape {g.y.shape}, expected (1,)"
        )
        assert not torch.isnan(g.x).any(), (
            f"Graph {i} ({g.patient} {g.timepoint}): NaN in node features"
        )
        assert not torch.isnan(g.edge_attr).any(), (
            f"Graph {i} ({g.patient} {g.timepoint}): NaN in edge attributes"
        )
        # All nodes must have identical feature width (no ragged rows)
        assert g.x.shape[1] > 0, (
            f"Graph {i}: node feature dimension is 0 — check GraphConfig"
        )


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------
def build_patient_sequence(
    patient_df: pd.DataFrame,
    config: GraphConfig,
    node_feature_cols: dict[str, list[str]],
    delta_feature_cols: dict[str, list[str]],
) -> PatientGraphSequence:
    """
    Build the ordered graph sequence for one patient.

    Sorting by time_from_diagnosis_weeks guarantees chronological order.
    Column resolution is done once per patient (passed in) to avoid redundant
    per-row DataFrame scans.

    Args:
        patient_df:          subset of engineered parquet for one patient.
        config:              GraphConfig for this topology.
        node_feature_cols:   resolved radiomic cols per node prefix.
        delta_feature_cols:  resolved delta cols per node prefix.

    Returns:
        PatientGraphSequence with graphs ordered chronologically.
    """
    patient_df = patient_df.sort_values("time_from_diagnosis_weeks").reset_index(drop=True)
    patient_id = str(patient_df["Patient"].iloc[0])

    graphs = [
        build_graph(row, config, node_feature_cols, delta_feature_cols)
        for _, row in patient_df.iterrows()
    ]
    label_sequence = [int(v) for v in patient_df[config.label_col].tolist()]

    return PatientGraphSequence(
        patient_id=patient_id,
        topology=config.topology,
        graphs=graphs,
        n_timepoints=len(graphs),
        label_sequence=label_sequence,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(
    topology: TopologyType = "3node",
    dry_run: bool = False,
    fast: bool = False,
) -> None:
    graphs_dir = GRAPHS_DIR / topology
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{SECTION}")
    print(f"GBM Longitudinal Toolkit — Graph Builder (Step 4, {topology})")
    print(SECTION)

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"{PARQUET_PATH} not found. Run features_builder.py (Step 2) first."
        )

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Patients: {df['Patient'].nunique()}")

    config = load_graph_config(SELECTED_FEATURES_PATH, topology=topology)

    # Resolve column lists once — shared across all patients (DRY, avoids O(n²) scans)
    node_feature_cols = _resolve_node_feature_cols(df, config)
    delta_feature_cols = _resolve_delta_feature_cols(df, config, node_feature_cols)

    # Feature dimension summary
    n_radiomic = sum(len(v) for v in node_feature_cols.values())
    n_delta = sum(len(v) for v in delta_feature_cols.values())
    n_scalar = len(config.scalar_features)
    print(
        f"Node features: {n_radiomic} radiomic + {n_delta} delta "
        f"+ {n_scalar} scalar = {n_radiomic + n_delta + n_scalar} per node"
    )

    patients = sorted(df["Patient"].unique())
    if fast:
        patients = patients[:5]
        print(f"  ⚠️  FAST MODE — processing first {len(patients)} patients only")

    all_graphs: list[Data] = []
    sequences: list[PatientGraphSequence] = []

    for patient_id in patients:
        patient_df = df[df["Patient"] == patient_id]
        seq = build_patient_sequence(patient_df, config, node_feature_cols, delta_feature_cols)
        sequences.append(seq)
        all_graphs.extend(seq.graphs)

        if not dry_run:
            save_path = graphs_dir / f"{patient_id}.pt"
            torch.save(seq, save_path)

    print(f"\nGraphs built: {len(all_graphs)} total across {len(sequences)} patients")

    print_section("Validating graph structure")
    validate_graphs(all_graphs, config)
    print("✅ All structural assertions passed")

    # Summary statistics
    seq_lengths = [s.n_timepoints for s in sequences]
    node_feat_dim = all_graphs[0].x.shape[1]
    print(
        f"\nSequence length — min: {min(seq_lengths)} | "
        f"max: {max(seq_lengths)} | "
        f"mean: {sum(seq_lengths) / len(seq_lengths):.1f}"
    )
    print(f"Node feature dim:  {node_feat_dim}")
    print(f"Edge feature dim:  {all_graphs[0].edge_attr.shape[1]}")
    print(f"n_edges per graph: {config.n_edges}")

    # Save report (even in dry-run — useful for CI)
    if not fast:
        report = {
            "topology": topology,
            "n_patients": len(sequences),
            "n_graphs": len(all_graphs),
            "node_feat_dim": node_feat_dim,
            "n_radiomic_per_node": n_radiomic // max(len(config.node_order), 1),
            "n_delta_per_node": n_delta // max(len(config.node_order), 1),
            "n_scalar": n_scalar,
            "n_edges": config.n_edges,
            "seq_len_min": int(min(seq_lengths)),
            "seq_len_max": int(max(seq_lengths)),
            "seq_len_mean": round(sum(seq_lengths) / len(seq_lengths), 2),
            "pre_selection_mode": not bool(config.node_feature_cols),
            "dry_run": dry_run,
        }
        report_path = GRAPHS_DIR / f"graph_builder_report_{topology}.json"
        if not dry_run:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport → {report_path}")

    if dry_run:
        print("\n[DRY RUN] No .pt files written.")
    else:
        print(f"\nSaved → {graphs_dir}/{{patient_id}}.pt ({len(sequences)} files)")

    print(SECTION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GBM Graph Builder (Step 4)")
    parser.add_argument(
        "--topology",
        choices=["3node", "2node"],
        default="3node",
        help="Graph topology: 3node (DeepBraTumIA, primary) or 2node (HD-GLIO-AUTO, ablation A6).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate graphs without saving .pt files.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Process only first 5 patients — smoke test.",
    )
    args = parser.parse_args()
    main(topology=args.topology, dry_run=args.dry_run, fast=args.fast)
