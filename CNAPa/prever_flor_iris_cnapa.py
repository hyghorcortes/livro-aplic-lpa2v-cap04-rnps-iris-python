from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .classificar_amostra_iris_cnapa import classificar_amostra_iris_cnapa
except ImportError:
    from classificar_amostra_iris_cnapa import classificar_amostra_iris_cnapa


def prever_flor_iris_cnapa(
    amostra: Sequence[float],
    mu_train: np.ndarray,
    sigma_train: np.ndarray,
    prototipos: np.ndarray,
    feature_range: np.ndarray,
    class_names: List[str],
    ftc: float,
    ftct: float,
    memoria_X: np.ndarray | None = None,
    memoria_y: np.ndarray | None = None,
) -> Tuple[str, pd.DataFrame]:
    x = np.asarray(amostra, dtype=float).reshape(1, -1)
    if x.shape[1] != 4:
        raise ValueError('A amostra deve ter exatamente 4 atributos.')
    x_norm = (x - mu_train.reshape(1, -1)) / sigma_train.reshape(1, -1)
    classe_prevista, _, tabela = classificar_amostra_iris_cnapa(
        x_norm.ravel(),
        prototipos,
        feature_range,
        class_names,
        ftc,
        ftct,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )
    return classe_prevista, tabela
