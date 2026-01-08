import numpy as np
from knn.metrics.confusion import confusion_matrix

def test_confusion_multiclass():
    y_true = np.array([0,0,1,1,2,2])
    y_pred = np.array([0,1,1,2,2,0])
    M, labs = confusion_matrix(y_true, y_pred)
    assert M.shape == (3,3)
    assert list(labs) == [0,1,2]
    # diagonal has at least one correct per class
    assert (np.diag(M) >= 1).all()
