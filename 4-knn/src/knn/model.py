from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Iterable
from collections import Counter

from .neighbors import kneighbors
from .distances import l2

Array = np.ndarray
Metric = Callable[[Array, Array], float]

class NotFittedError(RuntimeError):
    ...

class KNNClassifier:
    """
    From-scratch KNN classifier.
    - Stores training data (lazy model).
    - Pluggable distance metric.
    - Deterministic tie-breaking via neighbor index order.
    """
    def __init__(self, k: int = 3, metric: Metric = l2):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.metric = metric

        self._X: Optional[Array] = None
        self._y: Optional[Array] = None
        self._classes: Optional[Array] = None
        self._class_to_idx: Optional[dict] = None

    def fit(self, X: Array, y: Iterable) -> "KNNClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if len(X) != len(y):
            raise ValueError("X and y must have same number of samples")

        # allow non-integer labels; store unique order
        classes = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.vectorize(class_to_idx.get)(y)

        self._X = X
        self._y = y_idx
        self._classes = classes
        self._class_to_idx = class_to_idx
        return self

    def _check_fitted(self):
        if self._X is None or self._y is None:
            raise NotFittedError("KNNClassifier is not fitted. Call fit(X, y) first.")

    def _predict_one_idx(self, x: Array) -> int:
        self._check_fitted()
        idxs = kneighbors(self._X, x, self.k, self.metric)
        labels = self._y[idxs]
        # majority vote
        return Counter(labels).most_common(1)[0][0]

    def predict(self, X: Array) -> Array:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y_idx = np.array([self._predict_one_idx(row) for row in X], dtype=int)
        # map back to original label dtype
        return self._classes[y_idx]

    def predict_proba(self, X: Array) -> Array:
        """
        Returns probabilities per class in self.classes_ order (macro-normalized counts).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        C = len(self._classes)
        out = np.zeros((len(X), C), dtype=float)
        for i, row in enumerate(X):
            idxs = kneighbors(self._X, row, self.k, self.metric)
            labels = self._y[idxs]
            counts = np.bincount(labels, minlength=C)
            out[i, :] = counts / self.k
        return out

    @property
    def classes_(self) -> Array:
        self._check_fitted()
        return self._classes
