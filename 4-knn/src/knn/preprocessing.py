from __future__ import annotations
import numpy as np
from typing import Any, Protocol, List, Tuple

Array = np.ndarray

class Transformer(Protocol):
    def fit(self, X: Array, y: Array | None = None) -> "Transformer": ...
    def transform(self, X: Array) -> Array: ...

class StandardScaler:
    """Zero-mean, unit-variance per feature; guards zero std."""
    def __init__(self) -> None:
        self.mean_: Array | None = None
        self.std_: Array | None = None

    def fit(self, X: Array, y: Array | None = None) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std[std == 0.0] = 1.0
        self.std_ = std
        return self

    def transform(self, X: Array) -> Array:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler is not fitted.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

class Pipeline:
    """
    Minimal pipeline: sequence of (name, step), where last step must implement fit/predict.
    All previous steps must implement fit/transform.
    """
    def __init__(self, steps: List[Tuple[str, Any]]):
        if not steps:
            raise ValueError("Pipeline requires at least one step.")
        self.steps = steps

    def fit(self, X: Array, y: Array):
        Xt = X
        # fit/transform all but last
        for name, step in self.steps[:-1]:
            Xt = step.fit(Xt, y).transform(Xt)
        # fit last estimator
        last_name, last = self.steps[-1]
        last.fit(Xt, y)
        return self

    def predict(self, X: Array):
        Xt = X
        for name, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        _, last = self.steps[-1]
        return last.predict(Xt)

    def predict_proba(self, X: Array):
        Xt = X
        for name, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        _, last = self.steps[-1]
        if not hasattr(last, "predict_proba"):
            raise AttributeError("Last step does not implement predict_proba.")
        return last.predict_proba(Xt)
