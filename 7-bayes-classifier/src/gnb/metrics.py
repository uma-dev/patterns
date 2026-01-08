from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    balanced_accuracy: float
    sensitivity: float
    specificity: float


def _safe_div(num: float, den: float) -> float:
    # Educational default: if denominator is 0, return 0.0 (no evidence).
    return float(num / den) if den != 0.0 else 0.0


def confusion_counts_ovr(y_true: np.ndarray, y_pred: np.ndarray, positive_label) -> Tuple[int, int, int, int]:
    """
    One-vs-rest confusion counts for a given class label.
    Returns (TP, TN, FP, FN).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have same shape. Got {yt.shape} vs {yp.shape}")

    pos_true = (yt == positive_label)
    pos_pred = (yp == positive_label)

    tp = int(np.sum(pos_true & pos_pred))
    tn = int(np.sum(~pos_true & ~pos_pred))
    fp = int(np.sum(~pos_true & pos_pred))
    fn = int(np.sum(pos_true & ~pos_pred))
    return tp, tn, fp, fn


def sensitivity_specificity_balanced(tp: int, tn: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Returns (sensitivity, specificity, balanced_accuracy).
    """
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    bal = 0.5 * (sens + spec)
    return sens, spec, bal


def classification_metrics_macro_ovr(y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
    """
    Multiclass-safe metrics using One-vs-Rest per class, then macro-average.

    - Sensitivity: macro-average of per-class TPR
    - Specificity: macro-average of per-class TNR
    - Balanced Accuracy: macro-average of per-class (TPR+TNR)/2
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have same shape. Got {yt.shape} vs {yp.shape}")

    classes = np.unique(yt)
    if classes.size == 0:
        raise ValueError("No classes found in y_true.")

    sens_list = []
    spec_list = []
    bal_list = []

    for c in classes:
        tp, tn, fp, fn = confusion_counts_ovr(yt, yp, c)
        sens, spec, bal = sensitivity_specificity_balanced(tp, tn, fp, fn)
        sens_list.append(sens)
        spec_list.append(spec)
        bal_list.append(bal)

    return MetricResult(
        balanced_accuracy=float(np.mean(bal_list)),
        sensitivity=float(np.mean(sens_list)),
        specificity=float(np.mean(spec_list)),
    )


def classification_metrics_per_class_ovr(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, MetricResult]:
    """
    Optional helper: returns per-class metrics (OVR) in a dict.
    Keyed by class label (as int if possible).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    classes = np.unique(yt)

    out: Dict[int, MetricResult] = {}
    for c in classes:
        tp, tn, fp, fn = confusion_counts_ovr(yt, yp, c)
        sens, spec, bal = sensitivity_specificity_balanced(tp, tn, fp, fn)
        key = int(c) if np.issubdtype(type(c), np.integer) or str(
            c).isdigit() else c  # best-effort
        out[key] = MetricResult(balanced_accuracy=bal,
                                sensitivity=sens, specificity=spec)

    return out
