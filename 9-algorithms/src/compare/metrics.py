from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


@dataclass(frozen=True)
class FoldMetrics:
    sensitivity: float
    specificity: float
    balanced_acc: float
    f1: float


def specificity_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Specificity = TN / (TN + FP) for binary classification (positive=1).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    den = tn + fp
    return float(tn / den) if den != 0 else 0.0


def compute_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> FoldMetrics:
    return FoldMetrics(
        sensitivity=float(recall_score(y_true, y_pred, pos_label=1)),
        specificity=float(specificity_binary(y_true, y_pred)),
        balanced_acc=float(balanced_accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, pos_label=1)),
    )
