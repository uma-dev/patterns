from __future__ import annotations
import numpy as np
from typing import Tuple

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rows = true, Cols = pred. Returns (matrix, labels_used)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    L = len(labels)
    lab2idx = {lab: i for i, lab in enumerate(labels)}
    M = np.zeros((L, L), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[lab2idx[t], lab2idx[p]] += 1
    return M, labels
