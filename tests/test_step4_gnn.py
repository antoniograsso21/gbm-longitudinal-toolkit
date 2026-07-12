from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import torch

from src.graphs.graph_builder import GRAPHS_DIR, PatientGraphSequence
from src.models.tumor_gnn import GNNConfig, TumorTemporalGNN
from src.training.run_gnn import collate_patient_sequences
from src.utils.lumiere_io import build_full_feature_set


PARQUET_PATH = Path("data/processed/preprocessing/dataset_engineered.parquet")
GRAPHS_3NODE_DIR = GRAPHS_DIR / "3node"


def _require_step4_artifacts() -> list[Path]:
    pt_files = sorted(GRAPHS_3NODE_DIR.glob("*.pt"))
    if not pt_files:
        raise unittest.SkipTest(
            "3-node graph artifacts missing; run "
            "`uv run python -m src.graphs.graph_builder` first."
        )
    if not PARQUET_PATH.exists():
        raise unittest.SkipTest(f"{PARQUET_PATH} missing.")
    return pt_files


class Step4GraphAndGNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pt_files = _require_step4_artifacts()
        cls.sequences: list[PatientGraphSequence] = [
            torch.load(pt, weights_only=False) for pt in pt_files[:3]
        ]
        cls.df = pd.read_parquet(PARQUET_PATH)
        cls.all_feature_cols = build_full_feature_set(cls.df)

    def test_graph_artifacts_load_with_expected_shapes(self) -> None:
        seq = self.sequences[0]
        self.assertEqual(seq.topology, "3node")
        self.assertGreaterEqual(seq.n_timepoints, 1)
        graph = seq.graphs[0]
        self.assertEqual(tuple(graph.x.shape), (3, 67))
        self.assertEqual(tuple(graph.edge_index.shape), (2, 12))
        self.assertEqual(tuple(graph.edge_attr.shape), (12, 2))
        self.assertEqual(tuple(graph.y.shape), (1,))
        self.assertFalse(torch.isnan(graph.x).any())
        self.assertFalse(torch.isnan(graph.edge_attr).any())

    def test_collate_patient_sequences_keeps_node_specific_features_separate(self) -> None:
        feature_cols = [
            "CE_CT1_original_firstorder_10Percentile",
            "delta_CE_CT1_original_firstorder_10Percentile",
            "NC_CT1_original_shape_Maximum2DDiameterSlice",
            "delta_NC_CT1_original_shape_Maximum2DDiameterSlice",
            "time_from_diagnosis_weeks",
            "scan_index",
            "CE_vs_nadir",
            "delta_CE_vs_nadir",
        ]
        missing = [c for c in feature_cols if c not in self.all_feature_cols]
        self.assertEqual(missing, [])

        patients = [seq.patient_id for seq in self.sequences[:2]]
        df_split = (
            self.df[self.df["Patient"].isin(patients)]
            .sort_values(["Patient", "scan_index"])
            .reset_index(drop=True)
        )
        x_scaled = df_split[self.all_feature_cols].to_numpy(dtype=float)
        row_map: dict[str, list[int]] = {}
        for patient in patients:
            row_map[patient] = df_split.index[df_split["Patient"] == patient].tolist()

        batch = collate_patient_sequences(
            sequences=self.sequences[:2],
            feature_cols=feature_cols,
            all_feature_cols=self.all_feature_cols,
            X_scaled=x_scaled,
            patient_row_map=row_map,
            device=torch.device("cpu"),
        )
        x_seq, edge_index, edge_attr_seq, intervals, seq_lengths, labels = batch

        self.assertEqual(x_seq.shape[0], 2)
        self.assertEqual(x_seq.shape[2], 3)
        self.assertEqual(x_seq.shape[3], 6)
        self.assertEqual(tuple(edge_index.shape), (2, 12))
        self.assertEqual(edge_attr_seq.shape[-2:], (12, 2))
        self.assertEqual(tuple(seq_lengths.shape), (2,))
        self.assertEqual(tuple(labels.shape), (2,))
        self.assertTrue(torch.isfinite(intervals).all())
        self.assertTrue(torch.isfinite(x_seq).all())

        # ED has no node-specific feature in this test feature set; only shared
        # global context occupies the first slots (starting at len(node_vals)=0),
        # while padding zeros occupy the tail slots.
        ed_node_tail_slots = x_seq[:, :, 2, 4:]
        self.assertTrue(torch.allclose(ed_node_tail_slots, torch.zeros_like(ed_node_tail_slots)))

    def test_tumor_temporal_gnn_forward_returns_finite_logits(self) -> None:
        feature_cols = [
            "CE_CT1_original_firstorder_10Percentile",
            "NC_CT1_original_shape_Maximum2DDiameterSlice",
            "time_from_diagnosis_weeks",
            "scan_index",
        ]
        patients = [seq.patient_id for seq in self.sequences[:2]]
        df_split = (
            self.df[self.df["Patient"].isin(patients)]
            .sort_values(["Patient", "scan_index"])
            .reset_index(drop=True)
        )
        x_scaled = df_split[self.all_feature_cols].to_numpy(dtype=float)
        row_map = {
            patient: df_split.index[df_split["Patient"] == patient].tolist()
            for patient in patients
        }
        x_seq, edge_index, edge_attr_seq, intervals, seq_lengths, _ = collate_patient_sequences(
            sequences=self.sequences[:2],
            feature_cols=feature_cols,
            all_feature_cols=self.all_feature_cols,
            X_scaled=x_scaled,
            patient_row_map=row_map,
            device=torch.device("cpu"),
        )

        model = TumorTemporalGNN(GNNConfig(
            in_channels=x_seq.shape[-1],
            hidden=16,
            heads=1,
            dropout=0.0,
        ))
        logits = model(x_seq, edge_index, edge_attr_seq, intervals, seq_lengths)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
