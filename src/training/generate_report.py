"""
src/training/generate_report.py
================================
Generates the comparison table from Step 3 baseline JSON results.
Reads only the 'aggregated' field — schema_version agnostic.

Usage:
    uv run python src/training/generate_report.py
    uv run python src/training/generate_report.py --output results/comparison.csv
    uv run python src/training/generate_report.py --step3 --decimals 4
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
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

def _fmt(mean: float, std: float, *, decimals: int) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

def _parse_timestamp_utc(ts: str) -> datetime | None:
    try:
        # JSON timestamps are ISO 8601, usually with timezone.
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def load_row(display_name: str, path: Path, *, decimals: int) -> tuple[dict, datetime | None] | None:
    if not path.exists():
        print(f"  ⚠️  Missing: {path.name} — skipped")
        return None
    payload = json.loads(path.read_text())
    agg = payload["aggregated"]
    ts = None
    run_info = payload.get("run_info", {})
    if isinstance(run_info, dict) and isinstance(run_info.get("timestamp_utc"), str):
        ts = _parse_timestamp_utc(run_info["timestamp_utc"])
    row = {
        "Model":           display_name,
        "macro F1":        _fmt(agg["macro_f1_mean"],          agg["macro_f1_std"], decimals=decimals),
        "MCC":             _fmt(agg["mcc_mean"],               agg["mcc_std"], decimals=decimals),
        "AUC-PD":          _fmt(agg["auroc_progressive_mean"], agg["auroc_progressive_std"], decimals=decimals),
        "AUC-SD":          _fmt(agg["auroc_stable_mean"],      agg["auroc_stable_std"], decimals=decimals),
        "AUC-Resp":        _fmt(agg["auroc_response_mean"],    agg["auroc_response_std"], decimals=decimals),
        "PR-AUC-Resp":     _fmt(agg["prauc_response_mean"],    agg["prauc_response_std"], decimals=decimals),
        "PR-AUC-Stable":   _fmt(agg["prauc_stable_mean"],      agg["prauc_stable_std"], decimals=decimals),
    }
    return row, ts

def main(*, output: Path | None = None, decimals: int = 3, step3: bool = False) -> None:
    rows: list[dict] = []
    timestamps: list[datetime] = []
    for name, fname in MODEL_REGISTRY:
        loaded = load_row(name, BASELINES_DIR / fname, decimals=decimals)
        if loaded is None:
            continue
        row, ts = loaded
        rows.append(row)
        if ts is not None:
            timestamps.append(ts)

    df = pd.DataFrame(rows)
    if step3:
        run_date = max(timestamps).date().isoformat() if timestamps else "UNKNOWN"
        print(f"Run date: {run_date}. Reported as mean ± std across folds.\n")
    print(df.to_markdown(index=False))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"\n  Saved → {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--decimals", type=int, default=3, help="Decimal places for mean±std formatting.")
    parser.add_argument("--step3", action="store_true", help="Print copy-paste-ready block for docs/STEP_3.md.")
    args = parser.parse_args()
    main(output=args.output, decimals=args.decimals, step3=args.step3)