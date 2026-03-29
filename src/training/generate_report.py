"""
src/training/generate_report.py
================================
Generates the comparison table from Step 3 baseline JSON results.
Reads only the 'aggregated' field — schema_version agnostic.

Usage:
    uv run python src/training/generate_report.py
    uv run python src/training/generate_report.py --output results/comparison.csv
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

BASELINES_DIR = Path("data/processed/baselines")

# Registry: (display_name, filename)
MODEL_REGISTRY: list[tuple[str, str]] = [
    ("Logistic Regression",  "lr_results.json"),
    ("LightGBM A (radiomic)", "lgbm_A_results.json"),
    ("LightGBM B (temporal)", "lgbm_B_results.json"),
    ("LightGBM C (radio+temp)", "lgbm_C_results.json"),
    ("LightGBM D (full)",    "lgbm_D_results.json"),
    ("LSTM",                 "lstm_results.json"),
    # Step 4 — add when available:
    # ("GNN 2-node",         "gnn_2node_results.json"),
    # ("GNN 3-node",         "gnn_3node_results.json"),
]

def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"

def load_row(display_name: str, path: Path) -> dict | None:
    if not path.exists():
        print(f"  ⚠️  Missing: {path.name} — skipped")
        return None
    agg = json.loads(path.read_text())["aggregated"]
    return {
        "Model":           display_name,
        "macro F1":        _fmt(agg["macro_f1_mean"],          agg["macro_f1_std"]),
        "MCC":             _fmt(agg["mcc_mean"],               agg["mcc_std"]),
        "AUC-PD":          _fmt(agg["auroc_progressive_mean"], agg["auroc_progressive_std"]),
        "AUC-SD":          _fmt(agg["auroc_stable_mean"],      agg["auroc_stable_std"]),
        "AUC-Resp":        _fmt(agg["auroc_response_mean"],    agg["auroc_response_std"]),
        "PR-AUC-Resp":     _fmt(agg["prauc_response_mean"],    agg["prauc_response_std"]),
        "PR-AUC-Stable":   _fmt(agg["prauc_stable_mean"],      agg["prauc_stable_std"]),
    }

def main(output: Path | None = None) -> None:
    rows = [
        row for name, fname in MODEL_REGISTRY
        if (row := load_row(name, BASELINES_DIR / fname)) is not None
    ]
    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"\n  Saved → {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    main(output=args.output)