from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 10
    seed: int = 42


def stratified_kfold_splits(
    X: pd.DataFrame, y: np.ndarray, cfg: CVConfig
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=cfg.n_splits,
                          shuffle=True, random_state=cfg.seed)
    yield from skf.split(X, y)
