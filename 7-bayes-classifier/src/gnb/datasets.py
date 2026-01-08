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


def load_iris() -> DatasetBundle:
    d = datasets.load_iris()
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=int)
    return DatasetBundle(name="Iris", X=X, y=y)


def load_wine() -> DatasetBundle:
    d = datasets.load_wine()
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=int)
    return DatasetBundle(name="Wine", X=X, y=y)


def load_breast_cancer() -> DatasetBundle:
    d = datasets.load_breast_cancer()
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=int)
    return DatasetBundle(name="Breast Cancer", X=X, y=y)


def load_all_datasets() -> List[DatasetBundle]:
    return [load_iris(), load_wine(), load_breast_cancer()]
