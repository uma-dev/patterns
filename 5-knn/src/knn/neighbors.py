from __future__ import annotations
import numpy as np
from typing import Callable

Array = np.ndarray
Metric = Callable[[Array, Array], float]

def kneighbors(X_train: Array, x: Array, k: int, metric: Metric) -> np.ndarray:
    """
    Return indices of k nearest neighbors in X_train for single sample x.
    Uses partial sort (argpartition) for O(n) selection, then exact order for ties.
    """
    dists = np.apply_along_axis(lambda r: metric(r, x), 1, X_train)
    if k >= len(dists):
        return np.argsort(dists)
    idx_k = np.argpartition(dists, kth=k-1)[:k]
    order = np.lexsort((idx_k, dists[idx_k]))
    return idx_k[order]
