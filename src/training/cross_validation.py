"""
src/training/cross_validation.py
=================================
Cross-validation split builder for the LUMIERE baseline pipeline.

Design decisions:
- StratifiedGroupKFold: group=Patient prevents intra-patient leakage;
  stratify=target preserves class balance per fold.
- Leakage guard: asserts no patient appears in both train and test
  within any fold. Raises immediately on violation (Fail Fast).
- Returns typed CVSplits dataclass — no raw tuple juggling downstream.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------------------------
# Typed result
# ---------------------------------------------------------------------------

@dataclass
class FoldSplit:
    """Indices for a single CV fold."""
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class CVSplits:
    """All fold splits for one CV run."""
    n_splits: int
    folds: list[FoldSplit]


# ---------------------------------------------------------------------------
# Seed loader
# ---------------------------------------------------------------------------

def load_seed(config_path: str = "configs/random_state.yaml") -> int:
    """
    Load the global random seed from YAML.

    Args:
        config_path: path to random_state.yaml.

    Returns:
        Integer seed value.

    Raises:
        KeyError: if 'seed' key is missing from the YAML.
        FileNotFoundError: if config_path does not exist.
    """
    seed, _ = load_random_config(config_path)
    return seed


def load_random_config(config_path: str = "configs/random_state.yaml") -> tuple[int, int]:
    """
    Load seed and n_jobs from random_state.yaml.

    Args:
        config_path: path to random_state.yaml.

    Returns:
        Tuple (seed, n_jobs).

    Raises:
        KeyError: if 'seed' key is missing from the YAML.
        FileNotFoundError: if config_path does not exist.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "seed" not in cfg:
        raise KeyError(
            f"'seed' key not found in {config_path}. "
            "Expected format: 'seed: 42'"
        )
    seed = int(cfg["seed"])
    n_jobs = int(cfg.get("n_jobs", -1))
    return seed, n_jobs


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def build_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> CVSplits:
    """
    Build StratifiedGroupKFold splits with patient-level leakage guard.

    For each fold, asserts that no Patient appears in both train and test
    index sets. Raises ValueError immediately on violation.

    Args:
        X:        feature DataFrame (231 rows for LUMIERE).
        y:        target Series (target_encoded integers).
        groups:   patient ID Series aligned with X and y.
        n_splits: number of CV folds (default 5).
        seed:     random seed for reproducibility.

    Returns:
        CVSplits dataclass containing all fold splits.

    Raises:
        ValueError: if any patient leaks between train and test in any fold.
        ValueError: if X, y, groups have mismatched lengths.
    """
    if not (len(X) == len(y) == len(groups)):
        raise ValueError(
            f"Length mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}. "
            "All three must be aligned."
        )

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds: list[FoldSplit] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups)
    ):
        # Leakage guard — Fail Fast
        train_patients = set(groups.iloc[train_idx].unique())
        test_patients = set(groups.iloc[test_idx].unique())
        overlap = train_patients & test_patients
        if overlap:
            raise ValueError(
                f"Fold {fold_idx}: patient leakage detected. "
                f"Patients in both train and test: {sorted(overlap)}. "
                "This should never happen with StratifiedGroupKFold."
            )

        folds.append(FoldSplit(
            fold=fold_idx,
            train_idx=train_idx,
            test_idx=test_idx,
        ))

    return CVSplits(n_splits=n_splits, folds=folds)