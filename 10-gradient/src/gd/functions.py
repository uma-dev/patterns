from __future__ import annotations

import numpy as np


def f_xy(x: float, y: float) -> float:
    """
    Convex quadratic:
      f(x,y) = (x-3)^2 + 2(y+1)^2
    """
    return (x - 3.0) ** 2 + 2.0 * (y + 1.0) ** 2


def grad_xy(x: float, y: float) -> tuple[float, float]:
    """
    Gradient:
      df/dx = 2(x-3)
      df/dy = 4(y+1)
    """
    return 2.0 * (x - 3.0), 4.0 * (y + 1.0)


def grad_norm(gx: float, gy: float) -> float:
    return float(np.sqrt(gx * gx + gy * gy))
