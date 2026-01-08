from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn import datasets


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray


def _to_binary_labels(y: np.ndarray) -> np.ndarray:
    """
    Convert labels with exactly 2 unique values into {0,1}, stable ordering.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(f"Expected binary labels, got classes={classes}")
    mapping = {classes[0]: 0, classes[1]: 1}
    return np.vectorize(mapping.get)(y).astype(int)


def load_iris_1_vs_2() -> DatasetBundle:
    """
    Iris but only classes 1 and 2 (versicolor vs virginica).
    Remaps labels to {0,1}.
    """
    d = datasets.load_iris()
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=int)

    mask = (y == 1) | (y == 2)
    X = X[mask]
    y = y[mask]

    y_bin = _to_binary_labels(y)  # {1,2} -> {0,1}
    return DatasetBundle(name="Iris (class 1 vs 2)", X=X, y=y_bin)


def load_wine_binary() -> DatasetBundle:
    """
    Wine is 3-class in sklearn. Since our tree is binary-only, we take a binary subset:
    class 0 vs class 1 (drop class 2), then remap to {0,1}.
    """
    d = datasets.load_wine()
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=int)

    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]

    y_bin = _to_binary_labels(y)  # {0,1} -> {0,1}
    return DatasetBundle(name="Wine (class 0 vs 1)", X=X, y=y_bin)


def load_breast_cancer() -> DatasetBundle:
    d = datasets.load_breast_cancer()
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=int)
    y_bin = _to_binary_labels(y)  # already binary, ensure {0,1}
    return DatasetBundle(name="Breast Cancer", X=X, y=y_bin)


def load_all_datasets() -> List[DatasetBundle]:
    return [load_iris_1_vs_2(), load_wine_binary(), load_breast_cancer()]
