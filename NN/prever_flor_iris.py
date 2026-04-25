from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from NN.common import classificar_por_memoria_exemplares


def prever_flor_iris(
    net,
    mu: np.ndarray,
    sigma: np.ndarray,
    amostra: Sequence[float],
    class_names: List[str],
    memoria_X: np.ndarray | None = None,
    memoria_y: np.ndarray | None = None,
) -> Tuple[str, np.ndarray]:
    """Realiza a previsao da classe para uma nova amostra Iris."""
    x = np.asarray(amostra, dtype=float).reshape(1, -1)
    if x.shape[1] != 4:
        raise ValueError('A amostra deve ter exatamente 4 atributos.')

    xn = (x - mu.reshape(1, -1)) / sigma.reshape(1, -1)
    if memoria_X is not None and memoria_y is not None:
        y_mem, _, _, probabilidades = classificar_por_memoria_exemplares(
            xn,
            memoria_X,
            memoria_y,
            num_classes=len(class_names),
        )
        idx = int(y_mem[0] - 1)
        classe_prevista = class_names[idx]
        return classe_prevista, np.asarray(probabilidades[0], dtype=float)

    probabilidades = np.asarray(net.predict_proba(xn)[0], dtype=float)
    idx = int(np.argmax(probabilidades))
    classe_prevista = class_names[idx]
    return classe_prevista, probabilidades
