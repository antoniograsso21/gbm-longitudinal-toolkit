"""
diagnose_feature_selection.py
==============================
Standalone diagnostic for mRMR + Stability Selection on the LUMIERE dataset.

Does NOT run the full pipeline. Runs on a single fold (fold 0) with reduced
bootstrap replicates to produce a fast diagnostic report.

Answers:
    Q1. How many features survive variance filter per fold?
    Q2. What is the full bootstrap stability distribution (not just above tau)?
    Q3. Is the mRMR ranking consistent across replicates (rank correlation)?
    Q4. How many features pass at each tau level (0.1 to 0.7)?
    Q5. Are NC/ED features systematically suppressed vs CE?

Usage:
    python diagnose_feature_selection.py
    python diagnose_feature_selection.py --fold 0 --B 20 --n_select 50

Output:
    diagnose_feature_selection_report.json  — machine-readable
    Printed summary to stdout
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Paths — adjust if running from a different working directory
# ---------------------------------------------------------------------------
PARQUET_PATH = Path("data/processed/preprocessing/dataset_engineered.parquet")

# ---------------------------------------------------------------------------
# Constants (mirrors feature_selector.py)
# ---------------------------------------------------------------------------
TEMPORAL_COLS = frozenset({"interval_weeks", "scan_index", "time_from_diagnosis_weeks"})
NADIR_COLS = frozenset({"CE_vs_nadir", "weeks_since_nadir"})
DELTA_DERIVED_COLS = frozenset({"delta_CE_NC_ratio", "delta_CE_vs_nadir"})
NON_FEATURE_COLS = frozenset({
    "Patient", "Timepoint", "target", "target_encoded",
    "is_baseline_scan", "is_nadir_scan",
})
VARIANCE_THRESHOLD = 1e-6
RADIOMIC_PREFIX = "original_"


def _build_full_feature_set(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def _radiomic_cols_only(feature_cols: list[str]) -> list[str]:
    return [
        f for f in feature_cols
        if not f.startswith("delta_")
        and f not in TEMPORAL_COLS
        and f not in NADIR_COLS
        and f not in DELTA_DERIVED_COLS
        and RADIOMIC_PREFIX in f
    ]


def _get_node_prefix(col: str) -> str:
    """Return NC, CE, ED, or OTHER."""
    for prefix in ("NC", "CE", "ED"):
        if col.startswith(f"{prefix}_"):
            return prefix
    return "OTHER"


# ---------------------------------------------------------------------------
# Kraskov MI (minimal inline to avoid import issues)
# ---------------------------------------------------------------------------
def _mi_kraskov(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    try:
        from npeet.entropy_estimators import mi as kraskov_mi
        val = kraskov_mi(
            x.reshape(-1, 1).astype(float),
            y.reshape(-1, 1).astype(float),
            k=k,
        )
        return max(0.0, float(val))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Minimal mRMR (same logic as feature_selector.py)
# ---------------------------------------------------------------------------
def run_mrmr_minimal(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_select: int,
    k_mi: int = 3,
) -> list[str]:
    n_features = X.shape[1]
    n_select = min(n_select, n_features)

    relevance = np.array([_mi_kraskov(X[:, i], y, k=k_mi) for i in range(n_features)])
    selected: list[int] = []
    remaining = set(range(n_features))
    mi_cache: dict[tuple[int, int], float] = {}

    for _ in range(n_select):
        best_idx, best_score = -1, -np.inf
        for idx in remaining:
            if not selected:
                score = relevance[idx]
            else:
                red_vals = []
                for j in selected:
                    key = (min(idx, j), max(idx, j))
                    if key not in mi_cache:
                        mi_cache[key] = _mi_kraskov(X[:, idx], X[:, j], k=k_mi)
                    red_vals.append(mi_cache[key])
                score = relevance[idx] - float(np.mean(red_vals))
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [feature_names[i] for i in selected]


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------
def diagnose(
    fold: int = 0,
    B: int = 30,
    n_select: int = 50,
    seed: int = 42,
    k_mi: int = 3,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Feature Selection Diagnostic — fold={fold}, B={B}, n_select={n_select}")
    print(f"{'='*60}")

    # Load data
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"{PARQUET_PATH} not found.")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"\nLoaded: {df.shape[0]} rows × {df.shape[1]} columns")

    all_feature_cols = _build_full_feature_set(df)
    y = df["target_encoded"].values
    groups = df["Patient"].values

    # Build fold split (same as pipeline)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(cv.split(df[all_feature_cols], y, groups))
    train_idx, test_idx = splits[fold]
    print(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

    X_train_df = df.iloc[train_idx][all_feature_cols]
    y_train = y[train_idx]

    # Normalise (StandardScaler fit on train)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_train_df_scaled = pd.DataFrame(X_train_scaled, columns=all_feature_cols)

    # --- Q1: Variance filter ---
    print(f"\n{'─'*40}")
    print("Q1 — Variance filter")
    variances = X_train_scaled.var(axis=0)
    kept_mask = variances > VARIANCE_THRESHOLD
    n_dropped_variance = int((~kept_mask).sum())
    variance_passed = set(
        c for c, k in zip(all_feature_cols, kept_mask) if k
    )
    print(f"  Total features:           {len(all_feature_cols)}")
    print(f"  Dropped by variance<{VARIANCE_THRESHOLD}: {n_dropped_variance}")
    print(f"  Passed variance filter:   {len(variance_passed)}")

    radiomic_names = [
        f for f in all_feature_cols
        if f in variance_passed
        and not f.startswith("delta_")
        and f not in TEMPORAL_COLS
        and f not in NADIR_COLS
        and f not in DELTA_DERIVED_COLS
        and RADIOMIC_PREFIX in f
    ]
    print(f"  Radiomic candidates:      {len(radiomic_names)}")

    # Per-node breakdown
    node_counts = Counter(_get_node_prefix(c) for c in radiomic_names)
    for node, cnt in sorted(node_counts.items()):
        print(f"    {node}: {cnt}")

    # --- Q2 + Q3: Bootstrap stability distribution + rank consistency ---
    print(f"\n{'─'*40}")
    print(f"Q2/Q3 — Bootstrap stability (B={B} replicates)")

    radiomic_indices = [all_feature_cols.index(f) for f in radiomic_names]
    X_radiomic = X_train_scaled[:, radiomic_indices]

    sss = StratifiedShuffleSplit(n_splits=B, test_size=0.5, random_state=seed)
    replicate_indices = [idx for idx, _ in sss.split(X_radiomic, y_train)]

    all_selections: list[list[str]] = []
    all_rankings: list[list[str]] = []  # full ordered list from mRMR

    print(f"  Running {B} replicates (n_select={n_select})...")
    for rep_idx, idx in enumerate(replicate_indices):
        np.random.seed(seed + rep_idx)
        selected = run_mrmr_minimal(
            X_radiomic[idx], y_train[idx], radiomic_names,
            n_select=n_select, k_mi=k_mi
        )
        all_selections.append(selected)
        all_rankings.append(selected)  # mRMR returns ordered list
        if (rep_idx + 1) % 10 == 0:
            print(f"    replicate {rep_idx + 1}/{B} done")

    # Bootstrap stability scores
    selection_counts: dict[str, int] = {name: 0 for name in radiomic_names}
    for sel in all_selections:
        for name in sel:
            selection_counts[name] += 1

    stability_scores = {name: count / B for name, count in selection_counts.items()}
    sorted_stability = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Top-20 features by bootstrap stability:")
    print(f"  {'Rank':>4}  {'Stability':>9}  {'Node':>4}  Feature")
    print(f"  {'─'*4}  {'─'*9}  {'─'*4}  {'─'*40}")
    for rank, (fname, score) in enumerate(sorted_stability[:20], 1):
        node = _get_node_prefix(fname)
        short = fname.replace("original_", "").replace("_", "_")
        print(f"  {rank:>4}  {score:>9.3f}  {node:>4}  {short[:60]}")

    # Distribution at various tau levels
    print(f"\n  Features passing each tau threshold:")
    for tau in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        n_above = sum(1 for s in stability_scores.values() if s >= tau)
        node_breakdown = Counter(
            _get_node_prefix(f) for f, s in stability_scores.items() if s >= tau
        )
        bd_str = " | ".join(f"{k}:{v}" for k, v in sorted(node_breakdown.items()))
        print(f"    tau={tau:.2f}:  {n_above:>4} features   [{bd_str}]")

    # --- Q4: Rank correlation across replicates (mRMR consistency) ---
    print(f"\n{'─'*40}")
    print("Q4 — mRMR rank consistency across replicates")

    from scipy.stats import spearmanr

    # Build rank matrix: feature × replicate, rank within top-n_select
    rank_matrix = np.full((len(radiomic_names), B), np.nan)
    feat_to_idx = {f: i for i, f in enumerate(radiomic_names)}
    for rep_idx, sel in enumerate(all_rankings):
        for rank, feat in enumerate(sel):
            fi = feat_to_idx[feat]
            rank_matrix[fi, rep_idx] = rank

    # Compute mean pairwise Spearman correlation between replicate orderings
    # (sample 10 replicate pairs to keep this fast)
    rng = np.random.default_rng(seed)
    pairs = list(zip(
        rng.choice(B, size=min(20, B), replace=False),
        rng.choice(B, size=min(20, B), replace=False),
    ))
    pairs = [(a, b) for a, b in pairs if a != b][:15]

    rho_vals = []
    for a, b in pairs:
        # Use only features that appear in both replicates
        mask = ~np.isnan(rank_matrix[:, a]) & ~np.isnan(rank_matrix[:, b])
        if mask.sum() < 5:
            continue
        rho, _ = spearmanr(rank_matrix[mask, a], rank_matrix[mask, b])
        rho_vals.append(float(rho))

    if rho_vals:
        mean_rho = np.mean(rho_vals)
        print(f"  Mean Spearman ρ across {len(rho_vals)} replicate pairs: {mean_rho:.3f}")
        if mean_rho < 0.3:
            print("  ⚠️  LOW RANK CONSISTENCY — mRMR orderings are unstable across replicates.")
            print("     Likely cause: Kraskov MI variance at n~92 + high radiomic collinearity.")
            print("     Implication: stability scores are unreliable at any tau.")
        elif mean_rho < 0.6:
            print("  ⚠️  MODERATE RANK CONSISTENCY — stability scores are meaningful but noisy.")
        else:
            print("  ✅ HIGH RANK CONSISTENCY — mRMR orderings are stable.")

    # --- Q5: Node prefix breakdown of top-K stability ---
    print(f"\n{'─'*40}")
    print("Q5 — Node prefix breakdown at top stability levels")
    for topk in [10, 20, 50]:
        top_feats = [f for f, _ in sorted_stability[:topk]]
        nd = Counter(_get_node_prefix(f) for f in top_feats)
        print(f"  Top-{topk}: " + " | ".join(f"{k}:{v}" for k, v in sorted(nd.items())))

    # --- Summary and recommendation ---
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")

    n_at_04 = sum(1 for s in stability_scores.values() if s >= 0.40)
    n_at_03 = sum(1 for s in stability_scores.values() if s >= 0.30)
    n_at_02 = sum(1 for s in stability_scores.values() if s >= 0.20)

    print(f"  Radiomic candidates (post variance filter): {len(radiomic_names)}")
    print(f"  Features with stability ≥ 0.40 (current):  {n_at_04}")
    print(f"  Features with stability ≥ 0.30:            {n_at_03}")
    print(f"  Features with stability ≥ 0.20:            {n_at_02}")
    print()

    if n_at_04 < 10 and n_at_03 >= 15:
        print("  RECOMMENDATION: lower tau to 0.30")
        print("  RATIONALE: tau=0.40 is too strict for n~185/fold with 1284 radiomic features.")
        print("  Lowering to 0.30 is defensible on this dataset size — declare in Methods.")
    elif n_at_03 < 10 and n_at_02 >= 15:
        print("  RECOMMENDATION: lower tau to 0.20 AND investigate rank consistency.")
        print("  RATIONALE: Very few stable features even at tau=0.30 suggests Kraskov")
        print("  MI instability. Consider increasing k_mi from 3 to 5.")
    elif n_at_04 >= 15:
        print("  ✅ tau=0.40 appears calibrated correctly. 4-feature result may be a bug")
        print("  elsewhere (majority vote threshold, fold aggregation).")
        print("  Check: does each individual fold have ≥4 features, or only the YAML aggregate?")
    else:
        print("  ⚠️  Marginal situation. Run with B=100 for production-grade diagnosis.")

    # Save report
    report = {
        "fold": fold,
        "B": B,
        "n_select": n_select,
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_radiomic_candidates": len(radiomic_names),
        "n_dropped_variance": n_dropped_variance,
        "node_breakdown_candidates": dict(node_counts),
        "stability_at_tau": {
            str(tau): sum(1 for s in stability_scores.values() if s >= tau)
            for tau in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
        },
        "top20_stability": [(f, round(s, 4)) for f, s in sorted_stability[:20]],
        "mean_rank_spearman_rho": round(float(np.mean(rho_vals)), 3) if rho_vals else None,
        "node_breakdown_top10": dict(Counter(_get_node_prefix(f) for f, _ in sorted_stability[:10])),
        "node_breakdown_top20": dict(Counter(_get_node_prefix(f) for f, _ in sorted_stability[:20])),
    }

    report_path = Path("diagnose_feature_selection_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {report_path}")
    print(f"{'='*60}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",     type=int, default=0)
    parser.add_argument("--B",        type=int, default=30,
                        help="Bootstrap replicates (30 for fast diagnosis, 100 for production).")
    parser.add_argument("--n_select", type=int, default=50)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--k_mi",     type=int, default=3,
                        help="k for Kraskov MI estimator.")
    args = parser.parse_args()
    diagnose(fold=args.fold, B=args.B, n_select=args.n_select, seed=args.seed, k_mi=args.k_mi)
