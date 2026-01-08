import numpy as np
from dtree.criteria import gini_impurity


def test_gini_pure():
    y = np.array([1, 1, 1, 1])
    assert gini_impurity(y) == 0.0


def test_gini_balanced_binary():
    y = np.array([0, 0, 1, 1])
    # p0=0.5, p1=0.5 => 1 - (0.25+0.25)=0.5
    assert abs(gini_impurity(y) - 0.5) < 1e-12
