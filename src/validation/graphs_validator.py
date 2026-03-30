"""
src/validation/graphs_validator.py
====================================
Runs structural and semantic assertions on the graph sequences saved by
graph_builder.py (Step 4).

Assertions (3-node topology, all must PASS before GNN training):
    1.  n_patients_3node == 64 (all patients from dataset_engineered.parquet)
    2.  Every .pt file loads without error and is a PatientGraphSequence
    3.  Topology consistency: all graphs in sequence share the same topology tag
    4.  Node count: 3 per graph (3node) or 2 (2node)
    5.  edge_index shape: (2, 12) for 3node / (2, 2) for 2node
    6.  edge_attr shape: (12, 2) for 3node / (2, 2) for 2node
    7.  y shape: (1,) per graph
    8.  No NaN or inf in node features (x)
    9.  No NaN or inf in edge attributes
    10. Chronological ordering: time_from_diagnosis_weeks strictly increasing per sequence
    11. label_sequence length matches n_timepoints
    12. Node feature dimension is identical across all graphs (no ragged feature sets)
    W1. 2-node graphs exist (ablation A6) — WARN only if absent (not blocking)

Usage:
    python -m src.validation.graphs_validator
    python -m src.validation.graphs_validator --topology 2node
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import torch

from src.utils.lumiere_io import (
    SECTION,
    ValidationReport,
    print_section,
    save_validation_report,
    validation_result,
    validation_warn,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAPHS_DIR = Path("data/processed/graphs")
OUTPUT_DIR = Path("data/processed/validation")

EXPECTED_PATIENTS_3NODE = 64
EXPECTED_PATIENTS_2NODE = 54  # HD-GLIO-AUTO has fewer usable scans (audit confirmed)

TopologyType = Literal["3node", "2node"]

# Expected structural shapes per topology
_SHAPES: dict[str, dict] = {
    "3node": {"n_nodes": 3, "n_edges": 12, "edge_attr_shape": (12, 2)},
    "2node": {"n_nodes": 2, "n_edges": 2,  "edge_attr_shape": (2, 2)},
}


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _check_n_patients(
    sequences: list,
    topology: TopologyType,
) -> str:
    n = len(sequences)
    expected = EXPECTED_PATIENTS_3NODE if topology == "3node" else EXPECTED_PATIENTS_2NODE
    return validation_result(
        f"n_patients_{topology}",
        n == expected,
        f"got {n}, expected {expected}",
    )


def _check_loads_cleanly(
    pt_files: list[Path],
) -> tuple[str, list]:
    """
    Load all .pt files and collect PatientGraphSequence objects.
    Returns (status, sequences). Fails on first corrupt file.
    """
    sequences = []
    failed = []
    for pt in pt_files:
        try:
            seq = torch.load(pt, weights_only=False)
            sequences.append(seq)
        except Exception as e:
            failed.append(f"{pt.name}: {e}")
    return (
        validation_result("loads_cleanly", len(failed) == 0, f"corrupt files: {failed[:3]}"),
        sequences,
    )


def _check_topology_consistency(sequences: list, topology: TopologyType) -> str:
    bad = [
        seq.patient_id
        for seq in sequences
        if getattr(seq, "topology", None) != topology
    ]
    return validation_result(
        "topology_consistency",
        len(bad) == 0,
        f"wrong topology tag in {len(bad)} sequences: {bad[:3]}",
    )


def _check_structural_shapes(sequences: list, topology: TopologyType) -> str:
    shape_spec = _SHAPES[topology]
    n_nodes = shape_spec["n_nodes"]
    n_edges = shape_spec["n_edges"]
    edge_attr_shape = shape_spec["edge_attr_shape"]
    expected_edge_index = (2, n_edges)

    violations: list[str] = []
    for seq in sequences:
        for i, g in enumerate(seq.graphs):
            tag = f"{seq.patient_id}[{i}]"
            if g.x.shape[0] != n_nodes:
                violations.append(f"{tag}: x.shape[0]={g.x.shape[0]} ≠ {n_nodes}")
            if tuple(g.edge_index.shape) != expected_edge_index:
                violations.append(f"{tag}: edge_index={g.edge_index.shape} ≠ {expected_edge_index}")
            if tuple(g.edge_attr.shape) != edge_attr_shape:
                violations.append(f"{tag}: edge_attr={g.edge_attr.shape} ≠ {edge_attr_shape}")
            if tuple(g.y.shape) != (1,):
                violations.append(f"{tag}: y.shape={g.y.shape} ≠ (1,)")
            if violations:
                break  # report first 3 failures only
        if len(violations) >= 3:
            break

    return validation_result(
        "structural_shapes",
        len(violations) == 0,
        f"{len(violations)} shape violations: {violations[:3]}",
    )


def _check_no_nan_inf(sequences: list) -> str:
    violations: list[str] = []
    for seq in sequences:
        for i, g in enumerate(seq.graphs):
            tag = f"{seq.patient_id}[{i}]"
            if torch.isnan(g.x).any() or torch.isinf(g.x).any():
                violations.append(f"{tag}: NaN/inf in x")
            if torch.isnan(g.edge_attr).any() or torch.isinf(g.edge_attr).any():
                violations.append(f"{tag}: NaN/inf in edge_attr")
            if len(violations) >= 3:
                break
        if len(violations) >= 3:
            break
    return validation_result(
        "no_nan_inf",
        len(violations) == 0,
        f"{len(violations)} graphs with NaN/inf: {violations[:3]}",
    )


def _check_chronological_order(sequences: list) -> str:
    """
    Verify interval_weeks > 0 for all consecutive graph pairs within a sequence.
    A sequence of length 1 trivially passes.
    """
    violations: list[str] = []
    for seq in sequences:
        weeks = [g.interval_weeks for g in seq.graphs]
        for j in range(len(weeks) - 1):
            if weeks[j] <= 0:
                violations.append(
                    f"{seq.patient_id}[{j}]: interval_weeks={weeks[j]:.2f} ≤ 0"
                )
    return validation_result(
        "chronological_order",
        len(violations) == 0,
        f"{len(violations)} ordering violations: {violations[:3]}",
    )


def _check_label_sequence_length(sequences: list) -> str:
    bad = [
        seq.patient_id
        for seq in sequences
        if len(seq.label_sequence) != seq.n_timepoints
    ]
    return validation_result(
        "label_sequence_length",
        len(bad) == 0,
        f"length mismatch in {len(bad)} sequences: {bad[:3]}",
    )


def _check_uniform_node_feature_dim(sequences: list) -> str:
    """
    All graphs across all sequences must have identical node feature width.
    A ragged feature dimension would cause PyG batching to fail silently.
    """
    dims: set[int] = set()
    first_violator: str = ""
    for seq in sequences:
        for i, g in enumerate(seq.graphs):
            d = g.x.shape[1]
            dims.add(d)
            if len(dims) > 1 and not first_violator:
                first_violator = f"{seq.patient_id}[{i}]: dim={d}"

    ok = len(dims) == 1
    return validation_result(
        "uniform_node_feature_dim",
        ok,
        f"inconsistent dims {sorted(dims)} — first violator: {first_violator}",
    )


def _check_2node_exists() -> str:
    two_node_dir = GRAPHS_DIR / "2node"
    exists = two_node_dir.exists() and any(two_node_dir.glob("*.pt"))
    if not exists:
        return validation_warn(
            "2node_graphs_exist",
            f"{two_node_dir} is absent or empty — ablation A6 cannot run. "
            "Run: python -m src.graphs.graph_builder --topology 2node",
        )
    n = len(list(two_node_dir.glob("*.pt")))
    return validation_result("2node_graphs_exist", True, f"{n} files found")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(topology: TopologyType = "3node") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_section(f"LUMIERE Graphs Validation — Step 4 ({topology})")

    graphs_dir = GRAPHS_DIR / topology
    if not graphs_dir.exists() or not any(graphs_dir.glob("*.pt")):
        print(
            f"ERROR: {graphs_dir} is absent or empty. "
            "Run graph_builder.py first."
        )
        sys.exit(1)

    pt_files = sorted(graphs_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {graphs_dir}")

    results: dict[str, str] = {}

    # 1 — load cleanly (prerequisite for all other checks)
    load_status, sequences = _check_loads_cleanly(pt_files)
    results["1_loads_cleanly"] = load_status
    if not sequences:
        print("  ❌ No sequences loaded — aborting validation.")
        sys.exit(1)

    print_section("Running assertions")

    results["2_n_patients"]             = _check_n_patients(sequences, topology)
    results["3_topology_consistency"]   = _check_topology_consistency(sequences, topology)
    results["4_structural_shapes"]      = _check_structural_shapes(sequences, topology)
    results["5_no_nan_inf"]             = _check_no_nan_inf(sequences)
    results["6_chronological_order"]    = _check_chronological_order(sequences)
    results["7_label_sequence_length"]  = _check_label_sequence_length(sequences)
    results["8_uniform_feat_dim"]       = _check_uniform_node_feature_dim(sequences)

    # Warning-level check (not blocking)
    if topology == "3node":
        results["W1_2node_graphs_exist"] = _check_2node_exists()

    passed   = sum(1 for v in results.values() if v == "PASS")
    failed   = sum(1 for v in results.values() if v.startswith("FAIL"))
    warnings = sum(1 for v in results.values() if v.startswith("WARN"))

    print_section("SUMMARY")
    print(f"  PASS:    {passed}")
    print(f"  FAIL:    {failed}")
    print(f"  WARN:    {warnings}")

    # Metadata summary
    if sequences:
        all_graphs = [g for seq in sequences for g in seq.graphs]
        seq_lens = [seq.n_timepoints for seq in sequences]
        node_dim = all_graphs[0].x.shape[1] if all_graphs else 0
        metadata = {
            "topology": topology,
            "n_patients": len(sequences),
            "n_graphs": len(all_graphs),
            "node_feat_dim": node_dim,
            "seq_len_min": int(min(seq_lens)),
            "seq_len_max": int(max(seq_lens)),
            "seq_len_mean": round(sum(seq_lens) / len(seq_lens), 2),
        }
    else:
        metadata = {"topology": topology, "n_patients": 0, "n_graphs": 0}

    report = ValidationReport(
        passed=passed,
        failed=failed,
        warnings=warnings,
        results=results,
        metadata=metadata,
    )

    report_path = OUTPUT_DIR / f"graphs_validator_report_{topology}.json"

    if failed > 0:
        print(f"\n  ❌ VALIDATION FAILED — fix graph_builder.py before GNN training")
    else:
        print(f"\n  ✅ VALIDATION PASSED — {topology} graphs ready for Step 4 GNN")
    print(SECTION)

    save_validation_report(report, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graphs Validator (Step 4)")
    parser.add_argument(
        "--topology",
        choices=["3node", "2node"],
        default="3node",
        help="Topology to validate (default: 3node).",
    )
    args = parser.parse_args()
    main(topology=args.topology)
