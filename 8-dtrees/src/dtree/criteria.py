from __future__ import annotations

import numpy as np


def gini_impurity(y: np.ndarray) -> float:
    """
    Gini impurity for a 1D label array y.

    Gini(y) = 1 - sum_k p_k^2
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    n = y.size
    if n == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    p = counts / n
    return float(1.0 - np.sum(p * p))


def weighted_gini(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    Weighted Gini after a split into left/right.
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    n_l = y_left.size
    n_r = y_right.size
    n = n_l + n_r
    if n == 0:
        return 0.0
    return (n_l / n) * gini_impurity(y_left) + (n_r / n) * gini_impurity(y_right)
