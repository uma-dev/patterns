from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .criteria import weighted_gini


@dataclass(frozen=True)
class Split:
    feature_index: int
    threshold: float
    gini: float
    left_idx: np.ndarray
    right_idx: np.ndarray


def _candidate_thresholds(x_col: np.ndarray) -> np.ndarray:
    """
    Generate candidate thresholds for a numeric feature column.

    Production-ish simple approach:
    - sort unique values
    - thresholds are midpoints between consecutive unique values

    This avoids testing every unique value as threshold and reduces duplicates.
    """
    uniq = np.unique(x_col)
    if uniq.size <= 1:
        return np.array([], dtype=float)
    return (uniq[:-1] + uniq[1:]) / 2.0


def best_split_gini(X: np.ndarray, y: np.ndarray) -> Optional[Split]:
    """
    Find the best (feature, threshold) split that minimizes weighted Gini impurity.

    Returns:
      Split(...) if a valid split exists,
      None if no split improves / no valid thresholds.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y length mismatch.")

    n_samples, n_features = X.shape
    if n_samples < 2:
        return None

    best: Optional[Split] = None
    best_gini = float("inf")

    for j in range(n_features):
        col = X[:, j]
        thresholds = _candidate_thresholds(col)
        if thresholds.size == 0:
            continue

        for t in thresholds:
            left_idx = np.flatnonzero(col <= t)
            right_idx = np.flatnonzero(col > t)

            # must create two non-empty children
            if left_idx.size == 0 or right_idx.size == 0:
                continue

            g = weighted_gini(y[left_idx], y[right_idx])
            if g < best_gini:
                best_gini = g
                best = Split(
                    feature_index=j,
                    threshold=float(t),
                    gini=float(g),
                    left_idx=left_idx,
                    right_idx=right_idx,
                )

    return best
