from __future__ import annotations
import numpy as np
from typing import Tuple, List

def load_iris(source: str = "sklearn") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns: X (float32, shape [n,4]), y (int64, shape [n]), class_names (list[str])
    Tries scikit-learn first; if unavailable, uses a tiny bundled fallback.
    """
    if source == "sklearn":
        try:
            from sklearn.datasets import load_iris as _load
            ds = _load()
            X = ds.data.astype(np.float32)
            y = ds.target.astype(np.int64)
            class_names: List[str] = [str(c) for c in ds.target_names]
            return X, y, class_names
        except Exception:
            pass  # fall through to bundled

    # Minimal bundled subset (for no-sklearn environments). Not full iris.
    X = np.array([
        [5.1,3.5,1.4,0.2],
        [4.9,3.0,1.4,0.2],
        [6.3,3.3,6.0,2.5],
        [5.8,2.7,5.1,1.9],
        [5.7,2.8,4.1,1.3],
        [6.0,2.7,5.1,1.6],
        [6.7,3.1,4.7,1.5],
        [5.6,2.5,3.9,1.1],
        [6.3,2.9,5.6,1.8],
        [6.5,3.0,5.5,1.8],
    ], dtype=np.float32)
    y = np.array([0,0,2,2,1,2,1,1,2,2], dtype=np.int64)
    class_names = ["setosa","versicolor","virginica"]
    return X, y, class_names
