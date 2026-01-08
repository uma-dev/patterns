from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .model import GaussianNB
from .metrics import MetricResult, classification_metrics_macro_ovr


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 10
    seed: int = 42


def stratified_kfold_indices(y: np.ndarray, *, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Explicit Stratified K-Fold splitter implemented from scratch.

    Returns a list of (train_idx, test_idx) tuples of length n_splits.

    Strategy:
    - For each class, shuffle its indices with the same RNG.
    - Split per-class indices into n_splits chunks as evenly as possible.
    - Fold i test indices = concat of chunk_i from each class.
    - Train indices = all other indices.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    n = y.shape[0]
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if n_splits > n:
        raise ValueError("n_splits cannot be greater than number of samples.")

    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    # Build per-class shuffled indices
    per_class_chunks: Dict[object, List[np.ndarray]] = {}
    for c in classes:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)

        # Split into n_splits nearly equal chunks
        chunks = np.array_split(idx, n_splits)
        per_class_chunks[c] = chunks

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(n)

    for i in range(n_splits):
        test_parts = [per_class_chunks[c][i] for c in classes]
        test_idx = np.concatenate(
            test_parts) if test_parts else np.array([], dtype=int)

        # Keep test indices in a stable order (optional; helps reproducibility)
        test_idx = np.sort(test_idx)

        # Train = all - test
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        train_idx = all_idx[mask]

        folds.append((train_idx, test_idx))

    return folds


def run_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: CVConfig,
    model_factory: Callable[[], GaussianNB] = GaussianNB,
) -> MetricResult:
    """
    Run explicit stratified K-fold CV and return macro-OVR averaged metrics.

    We aggregate predictions across all folds (like evaluating on the full dataset once),
    then compute metrics once on the full y_true/y_pred (common educational approach).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y length mismatch.")

    folds = stratified_kfold_indices(
        y, n_splits=config.n_splits, seed=config.seed)

    y_pred_all = np.empty_like(y)
    for train_idx, test_idx in folds:
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred_all[test_idx] = model.predict(X[test_idx])

    return classification_metrics_macro_ovr(y, y_pred_all)


def run_leave_one_out(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_factory: Callable[[], GaussianNB] = GaussianNB,
) -> MetricResult:
    """
    Leave-One-Out validation: train on N-1, test on 1, for each sample.
    Aggregate y_pred, then compute macro-OVR metrics once.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y length mismatch.")

    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples for LOO.")

    y_pred = np.empty_like(y)

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        model = model_factory()
        model.fit(X[train_mask], y[train_mask])
        # X[i] is 1D, predict returns shape (1,)
        y_pred[i] = model.predict(X[i])[0]

    return classification_metrics_macro_ovr(y, y_pred)
