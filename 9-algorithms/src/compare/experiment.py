from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .datasets import DatasetSpec, load_openml_binary
from .metrics import FoldMetrics, compute_metrics_binary
from .models import get_models
from .preprocess import build_preprocessor
from .validation import CVConfig, stratified_kfold_splits


@dataclass(frozen=True)
class AvgMetrics:
    sensitivity: float
    specificity: float
    balanced_acc: float
    f1: float


def _to_dense_if_sparse(X):
    # GaussianNB needs dense after OneHotEncoder.
    return X.toarray() if hasattr(X, "toarray") else X


def evaluate_dataset(spec: DatasetSpec, cfg: CVConfig) -> Dict[str, AvgMetrics]:
    X, y = load_openml_binary(spec.openml_name)

    pre = build_preprocessor()
    models = get_models()

    results: Dict[str, AvgMetrics] = {}

    for model_name, model in models.items():
        fold_metrics: list[FoldMetrics] = []

        # GaussianNB needs dense. Others can stay sparse.
        if model_name == "GaussianNB":
            pipe = Pipeline(
                steps=[
                    ("pre", pre),
                    ("dense", FunctionTransformer(
                        _to_dense_if_sparse, accept_sparse=True)),
                    ("clf", model),
                ]
            )
        else:
            pipe = Pipeline(steps=[("pre", pre), ("clf", model)])

        for train_idx, test_idx in stratified_kfold_splits(X, y, cfg):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            fold_metrics.append(compute_metrics_binary(y_test, y_pred))

        results[model_name] = AvgMetrics(
            sensitivity=float(np.mean([m.sensitivity for m in fold_metrics])),
            specificity=float(np.mean([m.specificity for m in fold_metrics])),
            balanced_acc=float(
                np.mean([m.balanced_acc for m in fold_metrics])),
            f1=float(np.mean([m.f1 for m in fold_metrics])),
        )

    return results
