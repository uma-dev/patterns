from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Any

def cross_validate(
    pipeline_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    splitter,
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> dict:
    """
    Runs CV with a fresh pipeline per fold.
    Returns:
      {
        'folds': [{'idx': i, 'n_train': ..., 'n_test': ..., 'metrics': {...}}, ...],
        'summary': {'metric': {'mean': ..., 'std': ...}, ...}
      }
    """
    folds = []
    for i, (tr_idx, te_idx) in enumerate(splitter.split(X, y)):
        pipe = pipeline_factory()      # fresh pipeline per fold
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        mvals = {name: float(func(yte, yhat)) for name, func in metrics.items()}
        folds.append({"idx": i, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)), "metrics": mvals})

    # aggregate
    summary: dict[str, dict[str, float]] = {}
    for name in metrics.keys():
        values = [f["metrics"][name] for f in folds]
        summary[name] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}

    return {"folds": folds, "summary": summary}
