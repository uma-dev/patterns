from __future__ import annotations
import numpy as np
from typing import Tuple

def check_Xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features)")
    if len(X) != len(y):
        raise ValueError("X and y must have same number of samples")
    return X, y

def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = check_Xy(X, y)
    assert 0.0 < test_size < 1.0
    rng = np.random.default_rng(random_state)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
