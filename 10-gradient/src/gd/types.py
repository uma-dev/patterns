from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GDConfig:
    lr: float = 0.1
    epochs: int = 200
    tol: float = 1e-6


@dataclass(frozen=True)
class GDStep:
    epoch: int
    x: float
    y: float
    f: float
    grad_norm: float
