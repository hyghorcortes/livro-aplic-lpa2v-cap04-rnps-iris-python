from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from .common import classificar_por_memoria_exemplares
    from .cnapa_lpa2v import cnapa_lpa2v
except ImportError:
    from common import classificar_por_memoria_exemplares
    from cnapa_lpa2v import cnapa_lpa2v


def classificar_amostra_iris_cnapa(
    x_norm: np.ndarray,
    prototipos: np.ndarray,
    feature_range: np.ndarray,
    class_names: List[str],
    ftc: float,
    ftct: float,
    memoria_X: np.ndarray | None = None,
    memoria_y: np.ndarray | None = None,
) -> Tuple[str, int, pd.DataFrame]:
    x_norm = np.asarray(x_norm, dtype=float).reshape(-1)
    num_classes = prototipos.shape[0]

    mu_fav = np.zeros(num_classes, dtype=float)
    for k in range(num_classes):
        similaridade = 1.0 - np.abs(x_norm - prototipos[k, :]) / (feature_range + np.finfo(float).eps)
        similaridade = np.clip(similaridade, 0.0, 1.0)
        mu_fav[k] = float(similaridade.mean())

    lambda_contr = np.zeros(num_classes, dtype=float)
    gc = np.zeros(num_classes, dtype=float)
    gct = np.zeros(num_classes, dtype=float)
    mu_e = np.zeros(num_classes, dtype=float)
    phi_e = np.zeros(num_classes, dtype=float)
    s1 = np.zeros(num_classes, dtype=float)
    s2 = np.zeros(num_classes, dtype=float)
    score = np.zeros(num_classes, dtype=float)

    for k in range(num_classes):
        idx_outros = [i for i in range(num_classes) if i != k]
        lambda_contr[k] = float(np.max(mu_fav[idx_outros]))
        r = cnapa_lpa2v(mu_fav[k], lambda_contr[k], ftc, ftct)
        gc[k] = r['Gc']
        gct[k] = r['Gct']
        mu_e[k] = r['muE']
        phi_e[k] = r['phiE']
        s1[k] = r['S1']
        s2[k] = r['S2']
        score[k] = s1[k] + 1e-3 * gc[k]

    idx_previsto_0 = int(np.argmax(score))
    classe_prevista = class_names[idx_previsto_0]
    classe_memoria = ''
    dist_memoria = np.nan
    prob_final = np.full(num_classes, np.nan, dtype=float)

    if memoria_X is not None and memoria_y is not None:
        y_mem, _, dist_mem, proba_mem = classificar_por_memoria_exemplares(
            x_norm,
            memoria_X,
            memoria_y,
            num_classes=num_classes,
        )
        idx_previsto_0 = int(y_mem[0]) - 1
        classe_prevista = class_names[idx_previsto_0]
        classe_memoria = classe_prevista
        dist_memoria = float(dist_mem[0])
        prob_final = np.asarray(proba_mem[0], dtype=float)

    tabela = pd.DataFrame({
        'Classe': class_names,
        'Mu_Favoravel': mu_fav,
        'Lambda_Contraria': lambda_contr,
        'Gc': gc,
        'Gct': gct,
        'MuE': mu_e,
        'PhiE': phi_e,
        'S1': s1,
        'S2': s2,
        'Score': score,
    })
    if memoria_X is not None and memoria_y is not None:
        tabela['Probabilidade_Final'] = prob_final
        tabela['Classe_Memoria'] = classe_memoria
        tabela['Dist_Memoria'] = dist_memoria
        tabela['Predicao_Final'] = classe_prevista
    return classe_prevista, idx_previsto_0 + 1, tabela
