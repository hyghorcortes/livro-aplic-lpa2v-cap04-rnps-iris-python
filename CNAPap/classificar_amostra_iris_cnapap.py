from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from .common import classificar_por_memoria_exemplares
except ImportError:
    from common import classificar_por_memoria_exemplares


def classificar_amostra_iris_cnapap(
    x01: np.ndarray,
    prototipos: np.ndarray,
    class_names: List[str],
    memoria_X: np.ndarray | None = None,
    memoria_y: np.ndarray | None = None,
) -> Tuple[str, int, pd.DataFrame]:
    x01 = np.clip(np.asarray(x01, dtype=float).reshape(-1), 0.0, 1.0)
    num_classes = prototipos.shape[0]
    sim_por_classe = np.zeros((num_classes, prototipos.shape[1]), dtype=float)
    evidencia = np.zeros(num_classes, dtype=float)

    for c in range(num_classes):
        sim = 1.0 - np.abs(x01 - prototipos[c, :])
        sim = np.clip(sim, 0.0, 1.0)
        sim_por_classe[c, :] = sim
        evidencia[c] = float(sim.mean())

    idx_prev_0 = int(np.argmax(evidencia))
    classe_prevista = class_names[idx_prev_0]
    classe_memoria = ''
    dist_memoria = np.nan
    prob_final = np.full(num_classes, np.nan, dtype=float)

    if memoria_X is not None and memoria_y is not None:
        y_mem, _, dist_mem, proba_mem = classificar_por_memoria_exemplares(
            x01,
            memoria_X,
            memoria_y,
            num_classes=num_classes,
        )
        idx_prev_0 = int(y_mem[0]) - 1
        classe_prevista = class_names[idx_prev_0]
        classe_memoria = classe_prevista
        dist_memoria = float(dist_mem[0])
        prob_final = np.asarray(proba_mem[0], dtype=float)

    tabela = pd.DataFrame({
        'Classe': class_names,
        'Mem_SepComp': prototipos[:, 0],
        'Mem_SepLarg': prototipos[:, 1],
        'Mem_PetComp': prototipos[:, 2],
        'Mem_PetLarg': prototipos[:, 3],
        'Sim_SepComp': sim_por_classe[:, 0],
        'Sim_SepLarg': sim_por_classe[:, 1],
        'Sim_PetComp': sim_por_classe[:, 2],
        'Sim_PetLarg': sim_por_classe[:, 3],
        'Evidencia_Media': evidencia,
    })
    if memoria_X is not None and memoria_y is not None:
        tabela['Probabilidade_Final'] = prob_final
        tabela['Classe_Memoria'] = classe_memoria
        tabela['Dist_Memoria'] = dist_memoria
        tabela['Predicao_Final'] = classe_prevista
    return classe_prevista, idx_prev_0 + 1, tabela
