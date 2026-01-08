import numpy as np
from knn.neighbors import kneighbors
from knn.distances import l2

def test_kneighbors_basic():
    X = np.array([[0,0],[0,1],[1,0],[10,10]], dtype=float)
    x = np.array([0.1, 0.2], dtype=float)
    idxs = kneighbors(X, x, k=2, metric=l2)
    # nearest two should come from the cluster around origin
    assert set(idxs.tolist()) <= {0,1,2}
    assert len(idxs) == 2

