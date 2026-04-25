from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .classificar_amostra_iris_cnapca import classificar_amostra_iris_cnapca
except ImportError:
    from classificar_amostra_iris_cnapca import classificar_amostra_iris_cnapca


def prever_flor_iris_cnapca(
    amostra: Sequence[float],
    xmin: np.ndarray,
    xrange: np.ndarray,
    prototipos: np.ndarray,
    class_names: List[str],
    ftce: float,
    ftct: float,
    memoria_X: np.ndarray | None = None,
    memoria_y: np.ndarray | None = None,
) -> Tuple[str, pd.DataFrame]:
    x = np.asarray(amostra, dtype=float).reshape(1, -1)
    if x.shape[1] != 4:
        raise ValueError('A amostra deve ter exatamente 4 atributos.')
    x01 = np.clip((x - xmin.reshape(1, -1)) / xrange.reshape(1, -1), 0.0, 1.0)
    classe_prevista, _, tabela = classificar_amostra_iris_cnapca(
        x01.ravel(),
        prototipos,
        class_names,
        ftce,
        ftct,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )
    return classe_prevista, tabela
