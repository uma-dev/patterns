import numpy as np
from knn import KNNClassifier, StandardScaler, Pipeline
from knn.cv import KFold, LeaveOneOut, cross_validate
from knn.metrics import balanced_accuracy

def pipe_factory():
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", KNNClassifier(k=3))])

def test_kfold_runs():
    X = np.array([[0,0],[0,1],[1,0],[10,10],[10,9],[9,10]], dtype=float)
    y = np.array([0,0,0,1,1,1])
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    res = cross_validate(pipe_factory, X, y, kf, {"bal_acc": balanced_accuracy})
    assert "folds" in res and "summary" in res
    assert len(res["folds"]) == 3

def test_loo_runs_small():
    X = np.array([[0,0],[0,1],[1,0],[10,10]], dtype=float)
    y = np.array([0,0,0,1])
    loo = LeaveOneOut()
    res = cross_validate(pipe_factory, X, y, loo, {"bal_acc": balanced_accuracy})
    assert len(res["folds"]) == len(X)
