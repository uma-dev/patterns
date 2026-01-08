from __future__ import annotations
import numpy as np

Array = np.ndarray

def l2(a: Array, b: Array) -> float:
    """Euclidean distance."""
    diff = a - b
    # sqrt of sum of squares; cast to float for JSON/printing friendliness
    return float(np.sqrt(np.dot(diff, diff)))

def l1(a: Array, b: Array) -> float:
    """Manhattan distance."""
    return float(np.abs(a - b).sum())

def cosine(a: Array, b: Array) -> float:
    """Cosine distance (1 - cosine similarity). Range ~[0,2]."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        # define max distance if one vector is zero to avoid NaN
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))
