import numpy as np
from knn.metrics import sensitivity, specificity, balanced_accuracy, per_class_sensitivity_specificity

def test_metrics_binary_easy():
    y_true = np.array([0,0,0,1,1,1])
    y_pred = np.array([0,0,1,1,1,0])
    # confusion:
    # 0: TP=2, FN=1, FP=1, TN=2  -> recall=2/3, spec=2/3
    # 1: TP=2, FN=1, FP=1, TN=2  -> recall=2/3, spec=2/3
    rec = sensitivity(y_true, y_pred, average="macro")
    spec = specificity(y_true, y_pred, average="macro")
    bal  = balanced_accuracy(y_true, y_pred)
    assert abs(rec - (2/3)) < 1e-9
    assert abs(spec - (2/3)) < 1e-9
    assert abs(bal  - (2/3)) < 1e-9

def test_per_class_values():
    y_true = np.array([0,1,2,0,1,2])
    y_pred = np.array([0,2,2,1,1,0])
    rec, spec, labs = per_class_sensitivity_specificity(y_true, y_pred)
    assert len(rec) == len(spec) == len(labs) == 3
