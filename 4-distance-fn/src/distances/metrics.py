from __future__ import annotations

import numpy as np
from .utils import to_1d_float_array, validate_same_shape, ArrayLike


def chebyshev_distance(a: ArrayLike, b: ArrayLike) -> float:
    """
    Chebyshev distance (L-infinity): max_i |a_i - b_i|
    """
    va = to_1d_float_array(a, name="a")
    vb = to_1d_float_array(b, name="b")
    validate_same_shape(va, vb)

    return float(np.max(np.abs(va - vb)))


def hamming_distance(a: ArrayLike, b: ArrayLike) -> int:
    """
    Hamming distance: number of positions where a_i != b_i.
    Works best for binary/categorical vectors, but supports any numeric values.
    """
    va = np.asarray(a)
    vb = np.asarray(b)
    if va.ndim != 1 or vb.ndim != 1:
        raise ValueError(
            f"a and b must be 1D vectors. Got shapes {va.shape} and {vb.shape}")
    validate_same_shape(va, vb)

    return int(np.sum(va != vb))


def cosine_distance(a: ArrayLike, b: ArrayLike) -> float:
    """
    Cosine distance: 1 - (a·b / (||a|| * ||b||))

    Zero-vector behavior 
    - if both are zero vectors -> distance = 0.0
    - if only one is zero -> distance = 1.0
    """
    va = to_1d_float_array(a, name="a")
    vb = to_1d_float_array(b, name="b")
    validate_same_shape(va, vb)

    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))

    if na == 0.0 and nb == 0.0:
        return 0.0
    if na == 0.0 or nb == 0.0:
        return 1.0

    cos_sim = float(np.dot(va, vb) / (na * nb))
    # Numerical safety: keep within [-1, 1]
    cos_sim = max(-1.0, min(1.0, cos_sim))

    return 1.0 - cos_sim
