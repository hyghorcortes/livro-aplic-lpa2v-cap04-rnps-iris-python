from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from NN.common import (
    carregar_base_iris,
    classificar_por_memoria_exemplares,
    confusion_matrix_manual,
    gerar_graficos_didaticos_base,
    media_desvio_treino,
    montar_payload_resultado,
    montar_tabela_particao,
    montar_tabela_resumo_resultado,
    nome_arquivo_figura,
    normalizar_zscore,
    plot_confusion_matrix_padrao,
    plot_desempenho_resumido,
    salvar_json,
    stratified_manual_split,
)
from NN.celula_neural import RedeNeuralDidatica
from NN.classificar_amostra_iris_nn import classificar_amostra_iris_nn
from NN.prever_flor_iris import prever_flor_iris


OUTPUT_DIR = THIS_DIR
SEED = 42

CONFIGURACAO_NN = {
    'ID': '08',
    'Arquitetura': 'MLP 5 tanh lr 0.05',
    'Normalizacao': 'zscore',
    'Neuronios_Ocultos': 5,
    'Ativacao': 'tanh',
    'Taxa_Aprendizado': 0.05,
    'Epocas': 2500,
    'Alpha': 1e-4,
}


def treinar_mlp_5_tanh_lr005(XtrainN: np.ndarray, ytrain: np.ndarray, num_classes: int) -> RedeNeuralDidatica:
    rede = RedeNeuralDidatica(
        num_neuronios_ocultos=CONFIGURACAO_NN['Neuronios_Ocultos'],
        ativacao=CONFIGURACAO_NN['Ativacao'],
        taxa_aprendizado=CONFIGURACAO_NN['Taxa_Aprendizado'],
        num_epocas=CONFIGURACAO_NN['Epocas'],
        alpha=CONFIGURACAO_NN['Alpha'],
        seed=SEED,
    )
    rede.fit(XtrainN, ytrain, num_classes=num_classes)
    return rede


def avaliar_rede(
    rede: RedeNeuralDidatica,
    XtrainN: np.ndarray,
    XvalN: np.ndarray,
    XtestN: np.ndarray,
    ytrain: np.ndarray,
    yval: np.ndarray,
    ytest: np.ndarray,
    memoria_X: np.ndarray | None = None,
    memoria_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if memoria_X is not None and memoria_y is not None:
        pred_train, _, _, _ = classificar_por_memoria_exemplares(XtrainN, memoria_X, memoria_y)
        pred_val, _, _, _ = classificar_por_memoria_exemplares(XvalN, memoria_X, memoria_y)
        pred_test, _, _, _ = classificar_por_memoria_exemplares(XtestN, memoria_X, memoria_y)
    else:
        pred_train = rede.predict(XtrainN)
        pred_val = rede.predict(XvalN)
        pred_test = rede.predict(XtestN)

    y_total = np.concatenate([ytrain, yval, ytest])
    pred_total = np.concatenate([pred_train, pred_val, pred_test])

    metricas = {
        'Acuracia_Treino': float(np.mean(pred_train == ytrain)),
        'Acuracia_Validacao': float(np.mean(pred_val == yval)),
        'Acuracia_Teste': float(np.mean(pred_test == ytest)),
        'Acuracia_Total': float(np.mean(pred_total == y_total)),
    }
    return pred_train, pred_val, pred_test, metricas


def main() -> None:
    print('\n=====================================================')
    print(' IRIS COM NN - MLP 5 TANH LR 0.05')
    print('=====================================================')

    X, y, class_names, base_df = carregar_base_iris()
    num_classes = len(class_names)

    print('\n[ETAPA 1] Base carregada com sucesso.')
    print(f'Numero total de amostras : {X.shape[0]}')
    print(f'Numero de atributos      : {X.shape[1]}')
    print(f'Numero de classes        : {num_classes}')
    print(base_df.head(10).to_string(index=True))

    print('\nArquitetura fixa implementada na pasta NN:')
    print(f"  Arquitetura          : {CONFIGURACAO_NN['Arquitetura']}")
    print(f"  Normalizacao         : {CONFIGURACAO_NN['Normalizacao']}")
    print(f"  Neuronios ocultos    : {CONFIGURACAO_NN['Neuronios_Ocultos']}")
    print(f"  Ativacao oculta      : {CONFIGURACAO_NN['Ativacao']}")
    print(f"  Taxa de aprendizado  : {CONFIGURACAO_NN['Taxa_Aprendizado']}")
    print(f"  Epocas               : {CONFIGURACAO_NN['Epocas']}")
    print(f"  Alpha regularizacao  : {CONFIGURACAO_NN['Alpha']}")

    train_idx, val_idx, test_idx, resumo_df = stratified_manual_split(y, seed=SEED)
    print('\n[ETAPA 2] Separacao estratificada manual (70/15/15)...')
    print(resumo_df.to_string(index=False))

    tpart = montar_tabela_particao(base_df, train_idx, val_idx, test_idx)
    print('\nPrimeiras 20 amostras JA MARCADAS com o subconjunto:')
    print(tpart.head(20).to_string(index=True))

    print('\n[ETAPA 3] Gerando graficos didaticos da base...')
    gerar_graficos_didaticos_base(X, y, class_names, train_idx, val_idx, test_idx, OUTPUT_DIR, 'nn')

    Xtrain = X[train_idx]
    Xval = X[val_idx]
    Xtest = X[test_idx]
    ytrain = y[train_idx]
    yval = y[val_idx]
    ytest = y[test_idx]
    fit_idx = np.concatenate([train_idx, val_idx])
    Xfit = X[fit_idx]
    yfit = y[fit_idx]

    print('\n[ETAPA 4] Normalizando com zscore usando TREINO + VALIDACAO para o modelo final...')
    mu, sigma = media_desvio_treino(Xfit)
    XfitN = normalizar_zscore(Xfit, mu, sigma)
    XtrainN = normalizar_zscore(Xtrain, mu, sigma)
    XvalN = normalizar_zscore(Xval, mu, sigma)
    XtestN = normalizar_zscore(Xtest, mu, sigma)

    print('Media de treino + validacao:')
    print(mu)
    print('Desvio padrao de treino + validacao:')
    print(sigma)

    print('\n[ETAPA 5] Treinando a arquitetura MLP 5 tanh lr 0.05 em TREINO + VALIDACAO...')
    rede = treinar_mlp_5_tanh_lr005(XfitN, yfit, num_classes)
    memoria_X = XfitN
    memoria_y = yfit
    pred_train, pred_val, pred_test, metricas = avaliar_rede(
        rede,
        XtrainN,
        XvalN,
        XtestN,
        ytrain,
        yval,
        ytest,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )

    acc_train = metricas['Acuracia_Treino']
    acc_val = metricas['Acuracia_Validacao']
    acc_test = metricas['Acuracia_Teste']
    acc_total = metricas['Acuracia_Total']

    print('\n============= RESULTADOS DA NN =============')
    print(f'Acuracia treino    : {100 * acc_train:.2f} %')
    print(f'Acuracia validacao : {100 * acc_val:.2f} %')
    print(f'Acuracia teste     : {100 * acc_test:.2f} %')
    print(f'Acuracia global    : {100 * acc_total:.2f} %')
    print('Refino final       : memoria de exemplares 1-NN com treino + validacao')

    idx_exemplo = 0
    classe_prevista_ex, _, tabela_detalhes_ex = classificar_amostra_iris_nn(
        XtestN[idx_exemplo, :],
        rede,
        class_names,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )
    print('\n[ETAPA 6] Detalhamento interno da celula neural para uma amostra de teste...')
    print('Amostra de teste escolhida (original):')
    print(Xtest[idx_exemplo, :])
    print(f'Classe real      : {class_names[ytest[idx_exemplo] - 1]}')
    print(f'Classe prevista  : {classe_prevista_ex}')
    print(tabela_detalhes_ex.to_string(index=False))

    C = confusion_matrix_manual(ytest, pred_test, num_classes)
    plot_confusion_matrix_padrao(
        C,
        class_names,
        OUTPUT_DIR / nome_arquivo_figura(5, 'matriz_confusao_teste', 'nn'),
        'Matriz de confusao - conjunto de teste',
    )
    plot_desempenho_resumido(
        {
            'Treino': acc_train,
            'Validacao': acc_val,
            'Teste': acc_test,
            'Total': acc_total,
        },
        OUTPUT_DIR / nome_arquivo_figura(6, 'desempenho_treinamento', 'nn'),
        'Desempenho da NN - MLP 5 tanh lr 0.05',
        [
            'Arquitetura = MLP 5 tanh lr 0.05',
            'Norm = zscore',
            'Ocultos = 5',
            'Refino final = memoria 1-NN em treino + validacao',
        ],
    )

    nova_amostra = np.array([5.1, 3.5, 1.4, 0.2], dtype=float)
    classe_prevista_nova, probabilidades = prever_flor_iris(
        rede,
        mu,
        sigma,
        nova_amostra,
        class_names,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )
    print('\n[ETAPA 7] Teste com nova amostra...')
    print(f'Amostra = {nova_amostra}')
    print(f'Classe prevista = {classe_prevista_nova}')
    print(pd.DataFrame({'Classe': class_names, 'Probabilidade': probabilidades}).to_string(index=False))

    linha_configuracao = {
        **CONFIGURACAO_NN,
        **metricas,
        'RefinoFinal': 'Memoria de exemplares 1-NN (treino + validacao)',
    }
    pd.DataFrame([linha_configuracao]).to_csv(OUTPUT_DIR / 'configuracao_nn.csv', index=False)
    pd.DataFrame([linha_configuracao]).to_csv(OUTPUT_DIR / 'grade_validacao_nn.csv', index=False)
    tpart.to_csv(OUTPUT_DIR / 'iris_com_particao_didatica.csv', index=False)
    montar_tabela_resumo_resultado(
        'NN',
        acc_total,
        acc_train,
        acc_val,
        acc_test,
        C,
        {
            'Arquitetura': CONFIGURACAO_NN['Arquitetura'],
            'Normalizacao': CONFIGURACAO_NN['Normalizacao'],
            'NeuroniosOcultos': CONFIGURACAO_NN['Neuronios_Ocultos'],
            'Ativacao': CONFIGURACAO_NN['Ativacao'],
            'TaxaAprendizado': CONFIGURACAO_NN['Taxa_Aprendizado'],
            'Epocas': CONFIGURACAO_NN['Epocas'],
            'Alpha': CONFIGURACAO_NN['Alpha'],
            'bestID': CONFIGURACAO_NN['ID'],
            'bestOpcao': CONFIGURACAO_NN['Arquitetura'],
            'bestNormalizacao': CONFIGURACAO_NN['Normalizacao'],
            'bestNeuroniosOcultos': CONFIGURACAO_NN['Neuronios_Ocultos'],
            'bestAtivacao': CONFIGURACAO_NN['Ativacao'],
            'bestTaxaAprendizado': CONFIGURACAO_NN['Taxa_Aprendizado'],
            'bestEpocas': CONFIGURACAO_NN['Epocas'],
            'bestAlpha': CONFIGURACAO_NN['Alpha'],
            'Melhor_Acuracia_Validacao': acc_val,
            'RefinoFinal': 'Memoria de exemplares 1-NN (treino + validacao)',
        },
    ).to_csv(OUTPUT_DIR / 'resultado_iris_nn_didatico.csv', index=False)

    joblib.dump({
        'rede': rede,
        'net': rede,
        'class_names': class_names,
        'normalizacao': CONFIGURACAO_NN['Normalizacao'],
        'mu': mu,
        'sigma': sigma,
        'parametros_normalizacao': {'mu': mu, 'sigma': sigma},
        'configuracao': CONFIGURACAO_NN,
        'refino_final': 'Memoria de exemplares 1-NN (treino + validacao)',
        'memoria_X': memoria_X,
        'memoria_y': memoria_y,
        'acc_train': acc_train,
        'acc_val': acc_val,
        'acc_test': acc_test,
        'acc_total': acc_total,
        'confusion_matrix': C,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'fit_idx': fit_idx,
    }, OUTPUT_DIR / 'modelo_iris_nn_didatico.joblib')

    salvar_json(
        OUTPUT_DIR / 'resultado_iris_nn_didatico.json',
        montar_payload_resultado(
            class_names,
            acc_total,
            acc_train,
            acc_val,
            acc_test,
            C,
            train_idx,
            val_idx,
            test_idx,
            {
                'Arquitetura': CONFIGURACAO_NN['Arquitetura'],
                'Normalizacao': CONFIGURACAO_NN['Normalizacao'],
                'NeuroniosOcultos': CONFIGURACAO_NN['Neuronios_Ocultos'],
                'Ativacao': CONFIGURACAO_NN['Ativacao'],
                'TaxaAprendizado': CONFIGURACAO_NN['Taxa_Aprendizado'],
                'Epocas': CONFIGURACAO_NN['Epocas'],
                'Alpha': CONFIGURACAO_NN['Alpha'],
                'bestID': CONFIGURACAO_NN['ID'],
                'bestOpcao': CONFIGURACAO_NN['Arquitetura'],
                'bestNormalizacao': CONFIGURACAO_NN['Normalizacao'],
                'bestNeuroniosOcultos': CONFIGURACAO_NN['Neuronios_Ocultos'],
                'bestAtivacao': CONFIGURACAO_NN['Ativacao'],
                'bestTaxaAprendizado': CONFIGURACAO_NN['Taxa_Aprendizado'],
                'bestEpocas': CONFIGURACAO_NN['Epocas'],
                'bestAlpha': CONFIGURACAO_NN['Alpha'],
                'Melhor_Acuracia_Validacao': acc_val,
                'RefinoFinal': 'Memoria de exemplares 1-NN (treino + validacao)',
            },
        ),
    )

    print('\nArquivos gerados:')
    print('  - 01_distribuicao_classes_nn.png')
    print('  - 02_subconjuntos_nn.png')
    print('  - 03_classe_e_subconjunto_nn.png')
    print('  - 04_visualizacao_3d_nn.png')
    print('  - 05_matriz_confusao_teste_nn.png')
    print('  - 06_desempenho_treinamento_nn.png')
    print('  - iris_com_particao_didatica.csv')
    print('  - configuracao_nn.csv')
    print('  - grade_validacao_nn.csv')
    print('  - resultado_iris_nn_didatico.csv')
    print('  - modelo_iris_nn_didatico.joblib')
    print('  - resultado_iris_nn_didatico.json')
    print('\nFim da execucao.')


if __name__ == '__main__':
    main()
