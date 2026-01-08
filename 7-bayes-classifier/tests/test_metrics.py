import numpy as np
from gnb.metrics import classification_metrics_macro_ovr


def test_metrics_binary_simple():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    # For class 1 (positive): TP=2, FN=0, FP=1, TN=1
    # sens=1.0, spec=0.5, bal=0.75
    # For class 0 (OVR): TP=1, FN=1, FP=0, TN=2
    # sens=0.5, spec=1.0, bal=0.75
    # macro averages -> sens=0.75, spec=0.75, bal=0.75
    r = classification_metrics_macro_ovr(y_true, y_pred)
    assert abs(r.balanced_accuracy - 0.75) < 1e-12
    assert abs(r.sensitivity - 0.75) < 1e-12
    assert abs(r.specificity - 0.75) < 1e-12
