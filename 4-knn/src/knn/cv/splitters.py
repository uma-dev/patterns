from __future__ import annotations
import numpy as np
from typing import Iterator, Tuple

class KFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int | None = 42):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(idx)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = idx[start:stop]
            train_idx = np.concatenate([idx[:start], idx[stop:]])
            yield train_idx, test_idx
            current = stop

class LeaveOneOut:
    def split(self, X, y=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        idx = np.arange(n)
        for i in range(n):
            test_idx = idx[i:i+1]
            train_idx = np.concatenate([idx[:i], idx[i+1:]])
            yield train_idx, test_idx
