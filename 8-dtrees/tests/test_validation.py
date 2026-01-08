import numpy as np

from dtree.validation import stratified_train_test_split, HoldoutConfig, evaluate_holdout_accuracy
from dtree.model import DecisionTreeBinary


def test_stratified_split_coverage():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.array([0] * 5 + [1] * 5)

    Xtr, Xte, ytr, yte = stratified_train_test_split(
        X, y, test_size=0.3, seed=42)

    # total coverage
    assert Xtr.shape[0] + Xte.shape[0] == X.shape[0]
    assert ytr.shape[0] + yte.shape[0] == y.shape[0]

    # should keep both classes in train if possible
    assert set(np.unique(ytr)) == {0, 1}


def test_evaluate_holdout_accuracy_runs():
    X = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    acc = evaluate_holdout_accuracy(
        X, y,
        config=HoldoutConfig(test_size=0.33, seed=1),
        model_factory=lambda: DecisionTreeBinary(max_depth=2),
    )
    assert 0.0 <= acc <= 1.0
