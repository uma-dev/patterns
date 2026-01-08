from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    openml_name: str


DATASETS: List[DatasetSpec] = [
    DatasetSpec("German Credit", "credit-g"),
    DatasetSpec("Australian Credit Approval", "australian"),
    DatasetSpec("Credit Approval", "credit-approval"),
    DatasetSpec("Default Credit Card Clients",
                "default-of-credit-card-clients"),
    DatasetSpec("Bank Marketing", "bank-marketing"),
    DatasetSpec("HELOC", "heloc"),
]


def load_openml_binary(openml_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load an OpenML dataset and return:
      X: pandas DataFrame (keeps categorical columns)
      y: numpy array encoded to {0,1}

    Raises if target is not binary (keeps the assignment simple).
    """
    ds = fetch_openml(name=openml_name, as_frame=True)
    X = ds.data
    y = ds.target

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"Target has {y.shape[1]} columns; expected 1.")
        y = y.iloc[:, 0]

    y = pd.Series(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if np.unique(y_enc).size != 2:
        raise ValueError(f"Dataset '{openml_name}' is not binary.")

    return X, y_enc.astype(int)
