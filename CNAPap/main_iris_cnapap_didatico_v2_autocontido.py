from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from .common import (
        NOMES_VARIAVEIS,
        carregar_base_iris,
        confusion_matrix_manual,
        gerar_graficos_didaticos_base,
        minmax_treino,
        montar_payload_resultado,
        montar_tabela_particao,
        montar_tabela_resumo_resultado,
        nome_arquivo_figura,
        normalizar_minmax,
        plot_confusion_matrix_padrao,
        plot_desempenho_resumido,
        salvar_json,
        stratified_manual_split,
    )
    from .classificar_amostra_iris_cnapap import classificar_amostra_iris_cnapap
    from .cnapap_lpa2v import cnapap_lpa2v
    from .prever_flor_iris_cnapap import prever_flor_iris_cnapap
except ImportError:
    from common import (
        NOMES_VARIAVEIS,
        carregar_base_iris,
        confusion_matrix_manual,
        gerar_graficos_didaticos_base,
        minmax_treino,
        montar_payload_resultado,
        montar_tabela_particao,
        montar_tabela_resumo_resultado,
        nome_arquivo_figura,
        normalizar_minmax,
        plot_confusion_matrix_padrao,
        plot_desempenho_resumido,
        salvar_json,
        stratified_manual_split,
    )
    from classificar_amostra_iris_cnapap import classificar_amostra_iris_cnapap
    from cnapap_lpa2v import cnapap_lpa2v
    from prever_flor_iris_cnapap import prever_flor_iris_cnapap

SEED = 42
OUTPUT_DIR = THIS_DIR


def treinar_cnapap_prototipos(Xtrain01, ytrain, num_classes, fl, num_epocas, seed=SEED):
    rng = np.random.default_rng(seed)
    num_features = Xtrain01.shape[1]
    prototipos = 0.5 * np.ones((num_classes, num_features), dtype=float)
    history = np.zeros((num_epocas, num_classes, num_features), dtype=float)

    for ep in range(num_epocas):
        for c in range(1, num_classes + 1):
            idx = np.flatnonzero(ytrain == c)
            idx = rng.permutation(idx)
            for ii in idx:
                x = Xtrain01[ii, :]
                for j in range(num_features):
                    prototipos[c - 1, j] = cnapap_lpa2v(x[j], prototipos[c - 1, j], fl)
        history[ep, :, :] = prototipos
    return prototipos, history


def main() -> None:
    print('\n=====================================================')
    print(' IRIS COM CNAPap (LPA2v) - VERSAO DIDATICA')
    print('=====================================================')

    X, y, class_names, base_df = carregar_base_iris()
    num_classes = len(class_names)
    num_features = X.shape[1]

    train_idx, val_idx, test_idx, _ = stratified_manual_split(y, seed=SEED)
    tpart = montar_tabela_particao(base_df, train_idx, val_idx, test_idx)
    gerar_graficos_didaticos_base(X, y, class_names, train_idx, val_idx, test_idx, OUTPUT_DIR, 'cnapap')

    Xtrain = X[train_idx]
    Xval = X[val_idx]
    Xtest = X[test_idx]
    ytrain = y[train_idx]
    yval = y[val_idx]
    ytest = y[test_idx]

    xmin, xmax, xrange = minmax_treino(Xtrain)
    Xtrain01 = normalizar_minmax(Xtrain, xmin, xrange)
    Xval01 = normalizar_minmax(Xval, xmin, xrange)
    Xtest01 = normalizar_minmax(Xtest, xmin, xrange)

    print('\n[ETAPA 7] Ajustando FL e numero de epocas pela validacao...')
    fl_grid = np.array([0.20, 0.40, 0.60, 0.80, 1.00], dtype=float)
    epocas_grid = np.array([5, 10, 15, 20, 30], dtype=int)
    acc_grid = np.zeros((len(epocas_grid), len(fl_grid)), dtype=float)

    best_acc = -np.inf
    best_fl = np.nan
    best_epocas = np.nan
    best_proto = None
    best_history = None

    for iE, num_epocas in enumerate(epocas_grid):
        for iF, fl in enumerate(fl_grid):
            prototipos, history = treinar_cnapap_prototipos(Xtrain01, ytrain, num_classes, float(fl), int(num_epocas), seed=SEED)
            pred_val = np.zeros(Xval01.shape[0], dtype=int)
            for i in range(Xval01.shape[0]):
                _, idx_prev, _ = classificar_amostra_iris_cnapap(Xval01[i, :], prototipos, class_names)
                pred_val[i] = idx_prev
            acc_val = float(np.mean(pred_val == yval))
            acc_grid[iE, iF] = acc_val
            if acc_val > best_acc:
                best_acc = acc_val
                best_fl = float(fl)
                best_epocas = int(num_epocas)
                best_proto = prototipos.copy()
                best_history = history.copy()

    print('Melhor configuracao encontrada:')
    print(f'  FL          = {best_fl:.2f}')
    print(f'  Num. epocas = {best_epocas}')
    print(f'  Acuracia na validacao = {100*best_acc:.2f} %')

    print('\n[ETAPA 8] Reajustando normalizacao, prototipos e memoria com TREINO + VALIDACAO...')
    fit_idx = np.concatenate([train_idx, val_idx])
    Xfit = X[fit_idx]
    yfit = y[fit_idx]
    xmin, xmax, xrange = minmax_treino(Xfit)
    Xfit01 = normalizar_minmax(Xfit, xmin, xrange)
    Xtrain01 = normalizar_minmax(Xtrain, xmin, xrange)
    Xval01 = normalizar_minmax(Xval, xmin, xrange)
    Xtest01 = normalizar_minmax(Xtest, xmin, xrange)
    best_proto, best_history = treinar_cnapap_prototipos(Xfit01, yfit, num_classes, best_fl, best_epocas, seed=SEED)
    tabela_prot = pd.DataFrame(best_proto, columns=NOMES_VARIAVEIS)
    tabela_prot.insert(0, 'Classe', class_names)
    memoria_X = Xfit01
    memoria_y = yfit

    pred_train = np.zeros(Xtrain01.shape[0], dtype=int)
    pred_val = np.zeros(Xval01.shape[0], dtype=int)
    pred_test = np.zeros(Xtest01.shape[0], dtype=int)

    for i in range(Xtrain01.shape[0]):
        _, idx_prev, _ = classificar_amostra_iris_cnapap(
            Xtrain01[i, :],
            best_proto,
            class_names,
            memoria_X=memoria_X,
            memoria_y=memoria_y,
        )
        pred_train[i] = idx_prev
    for i in range(Xval01.shape[0]):
        _, idx_prev, _ = classificar_amostra_iris_cnapap(
            Xval01[i, :],
            best_proto,
            class_names,
            memoria_X=memoria_X,
            memoria_y=memoria_y,
        )
        pred_val[i] = idx_prev
    for i in range(Xtest01.shape[0]):
        _, idx_prev, _ = classificar_amostra_iris_cnapap(
            Xtest01[i, :],
            best_proto,
            class_names,
            memoria_X=memoria_X,
            memoria_y=memoria_y,
        )
        pred_test[i] = idx_prev

    acc_train = float(np.mean(pred_train == ytrain))
    acc_val = float(np.mean(pred_val == yval))
    acc_test = float(np.mean(pred_test == ytest))
    acc_total = float(np.mean(np.concatenate([pred_train, pred_val, pred_test]) == np.concatenate([ytrain, yval, ytest])))

    print('\n============= RESULTADOS =============')
    print(f'Acuracia treino    : {100*acc_train:.2f} %')
    print(f'Acuracia validacao : {100*acc_val:.2f} %')
    print(f'Acuracia teste     : {100*acc_test:.2f} %')
    print(f'Acuracia global    : {100*acc_total:.2f} %')
    print('Refino final       : memoria de exemplares 1-NN com treino + validacao')

    idx_exemplo = 0
    classe_prevista_ex, idx_prev_ex, tabela_detalhes_ex = classificar_amostra_iris_cnapap(
        Xtest01[idx_exemplo, :],
        best_proto,
        class_names,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )
    print('\n[ETAPA 11] Detalhamento interno para uma amostra de teste...')
    print(tabela_detalhes_ex.to_string(index=False))

    C = confusion_matrix_manual(ytest, pred_test, num_classes)
    plot_confusion_matrix_padrao(
        C,
        class_names,
        OUTPUT_DIR / nome_arquivo_figura(5, 'matriz_confusao_teste', 'cnapap'),
        'Matriz de confusao - conjunto de teste',
    )

    plot_desempenho_resumido(
        {
            'Treino': acc_train,
            'Validacao': acc_val,
            'Teste': acc_test,
            'Total': acc_total,
        },
        OUTPUT_DIR / nome_arquivo_figura(6, 'desempenho_treinamento', 'cnapap'),
        'Desempenho do CNAPap',
        [
            f'FL = {best_fl:.2f}',
            f'Epocas = {best_epocas}',
            f'Melhor validacao = {100 * best_acc:.2f}%',
            'Refino final = memoria 1-NN em treino + validacao',
        ],
    )

    nova_amostra = np.array([5.1, 3.5, 1.4, 0.2], dtype=float)
    classe_prevista_nova, tabela_nova = prever_flor_iris_cnapap(
        nova_amostra,
        xmin,
        xrange,
        best_proto,
        class_names,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )
    print('\n[ETAPA 12] Teste com nova amostra...')
    print(f'Classe prevista = {classe_prevista_nova}')
    print(tabela_nova.to_string(index=False))

    tpart.to_csv(OUTPUT_DIR / 'iris_com_particao_cnapap.csv', index=False)
    tabela_prot.to_csv(OUTPUT_DIR / 'prototipos_cnapap.csv', index=False)
    gridE, gridF = np.meshgrid(epocas_grid, fl_grid, indexing='ij')
    pd.DataFrame({
        'Epocas': gridE.ravel(),
        'FL': gridF.ravel(),
        'Acuracia_Validacao': acc_grid.ravel(),
    }).to_csv(OUTPUT_DIR / 'grade_validacao_cnapap.csv', index=False)
    montar_tabela_resumo_resultado(
        'CNAPap',
        acc_total,
        acc_train,
        acc_val,
        acc_test,
        C,
        {
            'bestFL': best_fl,
            'bestEpocas': best_epocas,
            'Melhor_Acuracia_Validacao': best_acc,
            'RefinoFinal': 'Memoria de exemplares 1-NN (treino + validacao)',
        },
    ).to_csv(OUTPUT_DIR / 'resultado_iris_cnapap_didatico.csv', index=False)

    joblib.dump({
        'class_names': class_names,
        'bestFL': best_fl,
        'bestEpocas': best_epocas,
        'acc_train': acc_train,
        'acc_val': acc_val,
        'acc_test': acc_test,
        'acc_total': acc_total,
        'xmin': xmin,
        'xmax': xmax,
        'xrange': xrange,
        'prototipos': best_proto,
        'history': best_history,
        'refino_final': 'Memoria de exemplares 1-NN (treino + validacao)',
        'memoria_X': memoria_X,
        'memoria_y': memoria_y,
        'confusion_matrix': C,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'fit_idx': fit_idx,
    }, OUTPUT_DIR / 'modelo_iris_cnapap_didatico.joblib')

    salvar_json(
        OUTPUT_DIR / 'resultado_iris_cnapap_didatico.json',
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
                'bestFL': best_fl,
                'bestEpocas': best_epocas,
                'Melhor_Acuracia_Validacao': best_acc,
                'RefinoFinal': 'Memoria de exemplares 1-NN (treino + validacao)',
            },
        ),
    )

    print('\nArquivos gerados:')
    print('  - 01_distribuicao_classes_cnapap.png')
    print('  - 02_subconjuntos_cnapap.png')
    print('  - 03_classe_e_subconjunto_cnapap.png')
    print('  - 04_visualizacao_3d_cnapap.png')
    print('  - 05_matriz_confusao_teste_cnapap.png')
    print('  - 06_desempenho_treinamento_cnapap.png')
    print('  - iris_com_particao_cnapap.csv')
    print('  - prototipos_cnapap.csv')
    print('  - grade_validacao_cnapap.csv')
    print('  - resultado_iris_cnapap_didatico.csv')
    print('  - modelo_iris_cnapap_didatico.joblib')
    print('  - resultado_iris_cnapap_didatico.json')
    print('\nFim da execucao.')


if __name__ == '__main__':
    main()
