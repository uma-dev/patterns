from __future__ import annotations

from typing import List, Tuple

from .functions import f_xy, grad_xy, grad_norm
from .types import GDConfig, GDStep


def gradient_descent(
    x0: float,
    y0: float,
    cfg: GDConfig,
) -> Tuple[float, float, List[GDStep]]:
    """
    Plain gradient descent:
      (x,y) <- (x,y) - lr * grad(x,y)

    Stops when:
      - epoch reaches cfg.epochs
      - OR grad_norm < cfg.tol
    """
    x, y = float(x0), float(y0)
    history: List[GDStep] = []

    for epoch in range(cfg.epochs):
        gx, gy = grad_xy(x, y)
        gnorm = grad_norm(gx, gy)
        fx = f_xy(x, y)

        history.append(GDStep(epoch=epoch, x=x, y=y, f=fx, grad_norm=gnorm))

        if gnorm < cfg.tol:
            break

        x = x - cfg.lr * gx
        y = y - cfg.lr * gy

    return x, y, history
