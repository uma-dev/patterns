from __future__ import annotations

from typing import Iterable, Union
import numpy as np

ArrayLike = Union[Iterable[float], np.ndarray]


def to_1d_float_array(x: ArrayLike, *, name: str) -> np.ndarray:
    """
    Convert input to a 1D float numpy array. Raises ValueError on invalid shape.
    """
    arr = np.asarray(x, dtype=float)

    if arr.ndim != 1:
        # We keep it strict and simple (vectors only).
        raise ValueError(f"{name} must be a 1D vector. Got shape={arr.shape}")

    return arr


def validate_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Vectors must have the same shape. Got {a.shape} vs {b.shape}")


def parse_csv_vector(s: str) -> np.ndarray:
    """
    Parse a comma-separated vector string like "1,2,3" into a 1D numpy float array.
    """
    if s is None:
        raise ValueError("Vector string is required")

    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("Vector string is empty")

    try:
        values = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"Invalid number in vector: {s}") from e

    return np.asarray(values, dtype=float)
