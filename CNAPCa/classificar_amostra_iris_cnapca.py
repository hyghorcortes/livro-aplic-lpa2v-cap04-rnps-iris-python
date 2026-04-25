from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from .common import classificar_por_memoria_exemplares
    from .cnapca_lpa2v import cnapca_lpa2v
except ImportError:
    from common import classificar_por_memoria_exemplares
    from cnapca_lpa2v import cnapca_lpa2v


def classificar_amostra_iris_cnapca(
    x01: np.ndarray,
    prototipos: np.ndarray,
    class_names: List[str],
    ftce: float,
    ftct: float,
    memoria_X: np.ndarray | None = None,
    memoria_y: np.ndarray | None = None,
) -> Tuple[str, int, pd.DataFrame]:
    x01 = np.clip(np.asarray(x01, dtype=float).reshape(-1), 0.0, 1.0)
    num_classes = prototipos.shape[0]

    mu1 = np.zeros(num_classes, dtype=float)
    mu2 = np.zeros(num_classes, dtype=float)
    mu3 = np.zeros(num_classes, dtype=float)
    mu4 = np.zeros(num_classes, dtype=float)
    c12_s1 = np.zeros(num_classes, dtype=float)
    c12_s2 = np.zeros(num_classes, dtype=float)
    c34_s1 = np.zeros(num_classes, dtype=float)
    c34_s2 = np.zeros(num_classes, dtype=float)
    c1234_s1 = np.zeros(num_classes, dtype=float)
    c1234_s2 = np.zeros(num_classes, dtype=float)
    score = np.zeros(num_classes, dtype=float)

    for c in range(num_classes):
        sim = 1.0 - np.abs(x01 - prototipos[c, :])
        sim = np.clip(sim, 0.0, 1.0)
        mu1[c], mu2[c], mu3[c], mu4[c] = sim.tolist()

        r12 = cnapca_lpa2v(mu1[c], mu2[c], ftce, ftct)
        r34 = cnapca_lpa2v(mu3[c], mu4[c], ftce, ftct)
        rf = cnapca_lpa2v(r12['S1'], r34['S1'], ftce, ftct)

        c12_s1[c], c12_s2[c] = r12['S1'], r12['S2']
        c34_s1[c], c34_s2[c] = r34['S1'], r34['S2']
        c1234_s1[c], c1234_s2[c] = rf['S1'], rf['S2']
        score[c] = rf['S1'] - 1e-3 * rf['S2']

    idx_prev_0 = int(np.argmax(score))
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
        'Prot_SepComp': prototipos[:, 0],
        'Prot_SepLarg': prototipos[:, 1],
        'Prot_PetComp': prototipos[:, 2],
        'Prot_PetLarg': prototipos[:, 3],
        'Mu1': mu1,
        'Mu2': mu2,
        'Mu3': mu3,
        'Mu4': mu4,
        'C12_S1': c12_s1,
        'C12_S2': c12_s2,
        'C34_S1': c34_s1,
        'C34_S2': c34_s2,
        'C1234_S1': c1234_s1,
        'C1234_S2': c1234_s2,
        'Score': score,
    })
    if memoria_X is not None and memoria_y is not None:
        tabela['Probabilidade_Final'] = prob_final
        tabela['Classe_Memoria'] = classe_memoria
        tabela['Dist_Memoria'] = dist_memoria
        tabela['Predicao_Final'] = classe_prevista
    return classe_prevista, idx_prev_0 + 1, tabela
