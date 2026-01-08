from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from .metrics import accuracy
from .model import DecisionTreeBinary


@dataclass(frozen=True)
class HoldoutConfig:
    test_size: float = 0.30
    seed: int = 42


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Explicit stratified train/test split from scratch.

    For each class:
      - shuffle indices
      - allocate ceil(test_size * n_class) to test (or round; we use floor to keep stable)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y length mismatch.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    test_indices = []
    train_indices = []

    for c in classes:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)

        n_c = idx.size
        n_test = int(np.floor(test_size * n_c))
        # ensure at least 1 sample goes to train if possible
        if n_c >= 2 and n_test == n_c:
            n_test = n_c - 1

        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        test_indices.append(test_idx)
        train_indices.append(train_idx)

    test_idx = np.sort(np.concatenate(test_indices)
                       ) if test_indices else np.array([], dtype=int)
    train_idx = np.sort(np.concatenate(train_indices)
                        ) if train_indices else np.array([], dtype=int)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def evaluate_holdout_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: HoldoutConfig,
    model_factory: Callable[[], DecisionTreeBinary],
) -> float:
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=config.test_size, seed=config.seed
    )
    clf = model_factory()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy(y_test, y_pred)
