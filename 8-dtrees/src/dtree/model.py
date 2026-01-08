from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .criteria import gini_impurity
from .split import best_split_gini, Split


@dataclass
class Node:
    # Split node fields
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    # Leaf fields
    is_leaf: bool = False
    prediction: Optional[int] = None  # class label at leaf
    depth: int = 0


class DecisionTreeBinary:
    """
    Simple binary decision tree classifier (CART-style) using Gini impurity.

    - Numeric features only
    - Split rule: X[:, j] <= threshold
    - Fixed max_depth (pre-pruning)
    """

    def __init__(self, *, max_depth: int = 4, min_samples_split: int = 2):
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_: Optional[Node] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeBinary":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape={X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D. Got shape={y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch.")
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples.")

        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(
                f"DecisionTreeBinary expects exactly 2 classes. Got {classes}")
        self.classes_ = classes

        self.root_ = self._build(X, y, depth=0)
        return self

    def _majority_class(self, y: np.ndarray) -> int:
        vals, counts = np.unique(y, return_counts=True)
        # deterministic tie-break: choose smaller label
        return int(vals[np.argmax(counts)])

    def _build(self, X: np.ndarray, y: np.ndarray, *, depth: int) -> Node:
        node = Node(depth=depth)

        # Stopping criteria
        if y.size == 0:
            node.is_leaf = True
            # fallback, should not happen normally
            node.prediction = int(self.classes_[0])
            return node

        # Pure node
        if np.unique(y).size == 1:
            node.is_leaf = True
            node.prediction = int(y[0])
            return node

        # Depth limit reached
        if depth >= self.max_depth:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return node

        # Not enough samples to split
        if y.size < self.min_samples_split:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return node

        # Find best split
        split: Optional[Split] = best_split_gini(X, y)
        if split is None:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return node

        # If split does not improve impurity, stop (educational pruning)
        parent_gini = gini_impurity(y)
        if split.gini >= parent_gini:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return node

        node.feature_index = split.feature_index
        node.threshold = split.threshold

        X_left, y_left = X[split.left_idx], y[split.left_idx]
        X_right, y_right = X[split.right_idx], y[split.right_idx]

        node.left = self._build(X_left, y_left, depth=depth + 1)
        node.right = self._build(X_right, y_right, depth=depth + 1)

        return node

    def _check_is_fitted(self) -> None:
        if self.root_ is None or self.classes_ is None:
            raise RuntimeError("Model not fitted. Call fit(X, y) first.")

    def _predict_one(self, x: np.ndarray) -> int:
        node = self.root_
        while node is not None and not node.is_leaf:
            j = node.feature_index
            t = node.threshold
            if j is None or t is None:
                break
            if x[j] <= t:
                node = node.left
            else:
                node = node.right

        # If something weird happened, fall back
        if node is None or node.prediction is None:
            return int(self.classes_[0])
        return int(node.prediction)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape={X.shape}")

        preds = np.array([self._predict_one(x) for x in X], dtype=int)
        return preds
