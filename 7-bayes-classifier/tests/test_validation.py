import numpy as np

from gnb.validation import CVConfig, stratified_kfold_indices, run_kfold_cv, run_leave_one_out


def test_stratified_kfold_sizes_and_coverage():
    # 12 samples, 2 classes, perfectly balanced
    y = np.array([0, 1] * 6)
    folds = stratified_kfold_indices(y, n_splits=3, seed=42)

    # Coverage: each index appears exactly once in test sets
    seen = np.concatenate([test for _, test in folds])
    assert np.array_equal(np.sort(seen), np.arange(len(y)))

    # Each fold should have 4 test samples
    for _, test_idx in folds:
        assert test_idx.shape[0] == 4


def test_kfold_and_loo_run():
    # Simple separable dataset
    X = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    cv_res = run_kfold_cv(X, y, config=CVConfig(n_splits=3, seed=1))
    loo_res = run_leave_one_out(X, y)

    # Should be strong on this easy dataset
    assert cv_res.balanced_accuracy >= 0.8
    assert loo_res.balanced_accuracy >= 0.8
