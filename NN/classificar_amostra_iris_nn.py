from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from NN.common import classificar_por_memoria_exemplares


def classificar_amostra_iris_nn(
    x_norm: np.ndarray,
    rede,
    class_names: List[str],
    memoria_X: np.ndarray | None = None,
    memoria_y: np.ndarray | None = None,
) -> Tuple[str, int, pd.DataFrame]:
    x_norm = np.asarray(x_norm, dtype=float).reshape(1, -1)
    detalhes = rede.detalhar(x_norm)
    probabilidades_rede = np.asarray(detalhes['probabilidades'][0], dtype=float)
    logits = np.asarray(detalhes['logits'][0], dtype=float)
    ativacao_oculta = np.asarray(detalhes['ativacao_oculta'][0], dtype=float)

    idx_previsto_0 = int(np.argmax(probabilidades_rede))
    classe_prevista = class_names[idx_previsto_0]
    classe_memoria = ''
    dist_memoria = np.nan
    probabilidades_finais = probabilidades_rede.copy()

    if memoria_X is not None and memoria_y is not None:
        y_mem, _, dist_mem, proba_mem = classificar_por_memoria_exemplares(
            x_norm,
            memoria_X,
            memoria_y,
            num_classes=len(class_names),
        )
        idx_previsto_0 = int(y_mem[0]) - 1
        classe_prevista = class_names[idx_previsto_0]
        classe_memoria = classe_prevista
        dist_memoria = float(dist_mem[0])
        probabilidades_finais = np.asarray(proba_mem[0], dtype=float)

    if memoria_X is not None and memoria_y is not None:
        tabela = pd.DataFrame({
            'Classe': class_names,
            'Logit_Saida': logits,
            'Probabilidade_Rede': probabilidades_rede,
            'Probabilidade_Final': probabilidades_finais,
            'Score_Rede': probabilidades_rede,
            'Score_Final': probabilidades_finais,
        })
        tabela['Classe_Memoria'] = classe_memoria
        tabela['Dist_Memoria'] = dist_memoria
        tabela['Predicao_Final'] = classe_prevista
    else:
        tabela = pd.DataFrame({
            'Classe': class_names,
            'Logit_Saida': logits,
            'Probabilidade': probabilidades_rede,
            'Score': probabilidades_rede,
        })

    for j, valor in enumerate(ativacao_oculta, start=1):
        tabela[f'CelulaOculta_{j}'] = float(valor)

    return classe_prevista, idx_previsto_0 + 1, tabela
