import numpy as np
from knn.distances import l1, l2, cosine

def test_l2_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert l2(a, a) == 0.0

def test_l1_simple():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    assert l1(a, b) == 3.0

def test_cosine_bounds():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    c = np.array([1.0, 0.0])
    assert abs(cosine(a, b) - 1.0) < 1e-12   # orthogonal → distance ~ 1
    assert abs(cosine(a, c) - 0.0) < 1e-12   # identical  → distance 0
