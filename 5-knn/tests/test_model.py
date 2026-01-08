import numpy as np
import pytest
from knn.model import KNNClassifier, NotFittedError

def test_predict_simple_two_clusters():
    X = np.array([[0,0],[0,1],[1,0],[10,10],[10,9],[9,10]], dtype=float)
    y = np.array([0,0,0,1,1,1])
    clf = KNNClassifier(k=3).fit(X, y)
    preds = clf.predict(np.array([[0.2,0.1],[9.3,9.1]]))
    assert (preds == np.array([0,1])).all()

def test_not_fitted_raises():
    clf = KNNClassifier(k=3)
    with pytest.raises(NotFittedError):
        clf.predict(np.array([[0.0, 0.0]]))
