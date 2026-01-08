from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GaussianNB:
    """
    Gaussian Naive Bayes (from scratch).

    Fit:
      - class priors pi[c]
      - per-class per-feature mean mu[c, j]
      - per-class per-feature variance var[c, j] + eps

    Predict:
      argmax_c [ log pi[c] + sum_j log N(x_j | mu[c,j], var[c,j]) ]
    """

    eps: float = 1e-9

    classes_: Optional[np.ndarray] = None          # shape: (C,)
    class_log_prior_: Optional[np.ndarray] = None  # shape: (C,)
    theta_: Optional[np.ndarray] = None            # means, shape: (C, D)
    var_: Optional[np.ndarray] = None              # variances, shape: (C, D)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D (n_samples, n_features). Got shape={X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (n_samples,). Got shape={y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}")
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples to fit.")

        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        n_samples, n_features = X.shape
        n_classes = classes.shape[0]

        # priors
        priors = counts / n_samples
        # avoid log(0) if anything weird happens
        priors = np.clip(priors, self.eps, 1.0)
        self.class_log_prior_ = np.log(priors)

        # means and variances per class
        theta = np.zeros((n_classes, n_features), dtype=float)
        var = np.zeros((n_classes, n_features), dtype=float)

        for idx, c in enumerate(classes):
            Xc = X[y == c]
            if Xc.shape[0] == 0:
                # Should not happen with np.unique counts, but keep safe.
                raise ValueError(f"Class {c} has no samples.")
            theta[idx] = Xc.mean(axis=0)

            # population variance (ddof=0) like most educational implementations
            v = Xc.var(axis=0, ddof=0)
            # smoothing to avoid division by zero / log(0)
            var[idx] = v + self.eps

        self.theta_ = theta
        self.var_ = var
        return self

    def _check_is_fitted(self) -> None:
        if (
            self.classes_ is None
            or self.class_log_prior_ is None
            or self.theta_ is None
            or self.var_ is None
        ):
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Returns matrix of shape (n_samples, n_classes) with:
          log P(c) + sum_j log N(x_j | mu[c,j], var[c,j])
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape={X.shape}")

        # Shapes:
        # X: (N, D)
        # theta_: (C, D)
        # var_: (C, D)
        # We compute per class using broadcasting.
        theta = self.theta_
        var = self.var_
        log_prior = self.class_log_prior_

        # log Gaussian probability per feature:
        # -0.5 * [ log(2π*var) + (x-mu)^2 / var ]
        # Sum over features.
        log_2pi = np.log(2.0 * np.pi)

        # Expand to (N, C, D)
        diff2 = (X[:, None, :] - theta[None, :, :]) ** 2
        log_prob = -0.5 * \
            (log_2pi + np.log(var[None, :, :]) + diff2 / var[None, :, :])
        # sum over features -> (N, C)
        jll = log_prior[None, :] + np.sum(log_prob, axis=2)
        return jll

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        jll = self._joint_log_likelihood(X)  # (N, C)
        class_idx = np.argmax(jll, axis=1)
        return self.classes_[class_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns normalized probabilities (softmax over joint log-likelihood).
        """
        self._check_is_fitted()
        jll = self._joint_log_likelihood(X)  # (N, C)
        # stable softmax
        jll = jll - np.max(jll, axis=1, keepdims=True)
        probs = np.exp(jll)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs
