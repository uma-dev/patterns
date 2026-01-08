import numpy as np
from dtree.model import DecisionTreeBinary


def test_tree_fits_and_predicts_easy():
    X = np.array([[0.0], [0.2], [0.8], [1.0]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeBinary(max_depth=2).fit(X, y)
    preds = clf.predict(X)
    assert np.array_equal(preds, y)


def test_tree_requires_binary_labels():
    X = np.array([[0.0], [0.2], [0.8], [1.0]])
    y = np.array([0, 1, 2, 2])
    clf = DecisionTreeBinary(max_depth=2)
    try:
        clf.fit(X, y)
        assert False, "Expected ValueError for non-binary labels"
    except ValueError:
        assert True
