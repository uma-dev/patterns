from __future__ import annotations
import numpy as np
from .confusion import confusion_matrix

def _per_class_tp_fp_fn_tn(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given multiclass confusion matrix M (rows=true, cols=pred),
    compute per-class TP, FP, FN, TN treating each class as "positive" one-vs-rest.
    """
    # TP: diagonal
    TP = np.diag(M).astype(float)
    # For class c:
    # FP = sum of col c except diagonal
    FP = M.sum(axis=0) - TP
    # FN = sum of row c except diagonal
    FN = M.sum(axis=1) - TP
    # TN = total - TP - FP - FN
    total = M.sum()
    TN = total - TP - FP - FN
    return TP, FP, FN, TN

def per_class_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (recall_per_class, specificity_per_class, labels_used)
    Sensitivity/Recall_c = TP_c / (TP_c + FN_c)
    Specificity_c       = TN_c / (TN_c + FP_c)
    """
    M, labs = confusion_matrix(y_true, y_pred, labels)
    TP, FP, FN, TN = _per_class_tp_fp_fn_tn(M)

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where((TP + FN) > 0, TP / (TP + FN), 0.0)
        spec   = np.where((TN + FP) > 0, TN / (TN + FP), 0.0)

    return recall, spec, labs

def sensitivity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None,
    average: str = "macro",
    pos_label=None
) -> float | np.ndarray:
    """
    average: "macro" | "micro" | "none"
    For binary with pos_label set and average="none", returns per-class recalls in label order.
    """
    rec, _, labs = per_class_sensitivity_specificity(y_true, y_pred, labels)
    if average == "none":
        return rec
    if average == "macro":
        return float(np.mean(rec)) if len(rec) else 0.0
    if average == "micro":
        # micro recall == global accuracy of positives (same as overall recall across classes)
        M, _ = confusion_matrix(y_true, y_pred, labels)
        TP = np.diag(M).sum()
        FN = (M.sum(axis=1) - np.diag(M)).sum()
        return float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    raise ValueError("average must be one of {'macro','micro','none'}")

def specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None,
    average: str = "macro",
    pos_label=None
) -> float | np.ndarray:
    spec = per_class_sensitivity_specificity(y_true, y_pred, labels)[1]
    if average == "none":
        return spec
    if average == "macro":
        return float(np.mean(spec)) if len(spec) else 0.0
    if average == "micro":
        # micro specificity is less standard; compute TN / (TN+FP) aggregated
        M, _ = confusion_matrix(y_true, y_pred, labels)
        TP = np.diag(M).sum()
        FP = (M.sum(axis=0) - np.diag(M)).sum()
        FN = (M.sum(axis=1) - np.diag(M)).sum()
        total = M.sum()
        TN = total - TP - FP - FN
        return float(TN / (TN + FP)) if (TN + FP) > 0 else 0.0
    raise ValueError("average must be one of {'macro','micro','none'}")

def balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None
) -> float:
    """
    Multiclass balanced accuracy (scikit-learn style): macro-average recall.
    """
    rec = sensitivity(y_true, y_pred, labels=labels, average="none")
    return float(np.mean(rec)) if len(rec) else 0.0
