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
    from .classificar_amostra_iris_cnapca import classificar_amostra_iris_cnapca
    from .prever_flor_iris_cnapca import prever_flor_iris_cnapca
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
    from classificar_amostra_iris_cnapca import classificar_amostra_iris_cnapca
    from prever_flor_iris_cnapca import prever_flor_iris_cnapca

SEED = 42
OUTPUT_DIR = THIS_DIR


def main() -> None:
    print('\n=====================================================')
    print(' IRIS COM CNAPCa (LPA2v) - VERSAO DIDATICA')
    print('=====================================================')

    X, y, class_names, base_df = carregar_base_iris()
    num_classes = len(class_names)

    train_idx, val_idx, test_idx, _ = stratified_manual_split(y, seed=SEED)
    tpart = montar_tabela_particao(base_df, train_idx, val_idx, test_idx)
    gerar_graficos_didaticos_base(X, y, class_names, train_idx, val_idx, test_idx, OUTPUT_DIR, 'cnapca')

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

    prototipos = np.zeros((num_classes, Xtrain01.shape[1]), dtype=float)
    for c in range(1, num_classes + 1):
        prototipos[c - 1, :] = Xtrain01[ytrain == c, :].mean(axis=0)

    tabela_prot = pd.DataFrame(prototipos, columns=NOMES_VARIAVEIS)
    tabela_prot.insert(0, 'Classe', class_names)

    ftce_grid = np.round(np.arange(0.0, 1.01, 0.1), 2)
    ftct_grid = np.round(np.arange(0.0, 1.01, 0.1), 2)
    acc_grid = np.zeros((len(ftct_grid), len(ftce_grid)), dtype=float)

    best_acc = -np.inf
    best_ftce = np.nan
    best_ftct = np.nan

    for iCt, ftct in enumerate(ftct_grid):
        for iCe, ftce in enumerate(ftce_grid):
            pred_val = np.zeros(Xval01.shape[0], dtype=int)
            for i in range(Xval01.shape[0]):
                _, idx_prev, _ = classificar_amostra_iris_cnapca(Xval01[i, :], prototipos, class_names, float(ftce), float(ftct))
                pred_val[i] = idx_prev
            acc_val = float(np.mean(pred_val == yval))
            acc_grid[iCt, iCe] = acc_val
            if acc_val > best_acc:
                best_acc = acc_val
                best_ftce = float(ftce)
                best_ftct = float(ftct)

    print('Melhor configuracao:')
    print(f'  Ftce = {best_ftce:.2f}')
    print(f'  Ftct = {best_ftct:.2f}')
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

    prototipos = np.zeros((num_classes, Xfit01.shape[1]), dtype=float)
    for c in range(1, num_classes + 1):
        prototipos[c - 1, :] = Xfit01[yfit == c, :].mean(axis=0)

    tabela_prot = pd.DataFrame(prototipos, columns=NOMES_VARIAVEIS)
    tabela_prot.insert(0, 'Classe', class_names)
    memoria_X = Xfit01
    memoria_y = yfit

    pred_train = np.zeros(Xtrain01.shape[0], dtype=int)
    pred_val = np.zeros(Xval01.shape[0], dtype=int)
    pred_test = np.zeros(Xtest01.shape[0], dtype=int)

    for i in range(Xtrain01.shape[0]):
        _, idx_prev, _ = classificar_amostra_iris_cnapca(
            Xtrain01[i, :],
            prototipos,
            class_names,
            best_ftce,
            best_ftct,
            memoria_X=memoria_X,
            memoria_y=memoria_y,
        )
        pred_train[i] = idx_prev
    for i in range(Xval01.shape[0]):
        _, idx_prev, _ = classificar_amostra_iris_cnapca(
            Xval01[i, :],
            prototipos,
            class_names,
            best_ftce,
            best_ftct,
            memoria_X=memoria_X,
            memoria_y=memoria_y,
        )
        pred_val[i] = idx_prev
    for i in range(Xtest01.shape[0]):
        _, idx_prev, _ = classificar_amostra_iris_cnapca(
            Xtest01[i, :],
            prototipos,
            class_names,
            best_ftce,
            best_ftct,
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
    classe_prevista_ex, idx_prev_ex, tabela_detalhes_ex = classificar_amostra_iris_cnapca(
        Xtest01[idx_exemplo, :],
        prototipos,
        class_names,
        best_ftce,
        best_ftct,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )
    print('\n[ETAPA 10] Mostrando o processamento CNAPCa de uma amostra de teste...')
    print(tabela_detalhes_ex.to_string(index=False))

    C = confusion_matrix_manual(ytest, pred_test, num_classes)
    plot_confusion_matrix_padrao(
        C,
        class_names,
        OUTPUT_DIR / nome_arquivo_figura(5, 'matriz_confusao_teste', 'cnapca'),
        'Matriz de confusao - conjunto de teste',
    )

    plot_desempenho_resumido(
        {
            'Treino': acc_train,
            'Validacao': acc_val,
            'Teste': acc_test,
            'Total': acc_total,
        },
        OUTPUT_DIR / nome_arquivo_figura(6, 'desempenho_treinamento', 'cnapca'),
        'Desempenho do CNAPCa',
        [
            f'Ftce = {best_ftce:.2f}',
            f'Ftct = {best_ftct:.2f}',
            f'Melhor validacao = {100 * best_acc:.2f}%',
            'Refino final = memoria 1-NN em treino + validacao',
        ],
    )

    nova_amostra = np.array([5.1, 3.5, 1.4, 0.2], dtype=float)
    classe_prevista_nova, tabela_nova = prever_flor_iris_cnapca(
        nova_amostra,
        xmin,
        xrange,
        prototipos,
        class_names,
        best_ftce,
        best_ftct,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )
    print('\n[ETAPA 11] Teste com nova amostra...')
    print(f'Classe prevista = {classe_prevista_nova}')
    print(tabela_nova.to_string(index=False))

    tpart.to_csv(OUTPUT_DIR / 'iris_com_particao_cnapca.csv', index=False)
    tabela_prot.to_csv(OUTPUT_DIR / 'prototipos_cnapca.csv', index=False)
    gridCt, gridCe = np.meshgrid(ftct_grid, ftce_grid, indexing='ij')
    pd.DataFrame({
        'Ftct': gridCt.ravel(),
        'Ftce': gridCe.ravel(),
        'Acuracia_Validacao': acc_grid.ravel(),
    }).to_csv(OUTPUT_DIR / 'grade_validacao_cnapca.csv', index=False)
    montar_tabela_resumo_resultado(
        'CNAPCa',
        acc_total,
        acc_train,
        acc_val,
        acc_test,
        C,
        {
            'bestFtce': best_ftce,
            'bestFtct': best_ftct,
            'Melhor_Acuracia_Validacao': best_acc,
            'RefinoFinal': 'Memoria de exemplares 1-NN (treino + validacao)',
        },
    ).to_csv(OUTPUT_DIR / 'resultado_iris_cnapca_didatico.csv', index=False)

    joblib.dump({
        'class_names': class_names,
        'bestFtce': best_ftce,
        'bestFtct': best_ftct,
        'acc_train': acc_train,
        'acc_val': acc_val,
        'acc_test': acc_test,
        'acc_total': acc_total,
        'xmin': xmin,
        'xmax': xmax,
        'xrange': xrange,
        'prototipos': prototipos,
        'refino_final': 'Memoria de exemplares 1-NN (treino + validacao)',
        'memoria_X': memoria_X,
        'memoria_y': memoria_y,
        'confusion_matrix': C,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'fit_idx': fit_idx,
    }, OUTPUT_DIR / 'modelo_iris_cnapca_didatico.joblib')

    salvar_json(
        OUTPUT_DIR / 'resultado_iris_cnapca_didatico.json',
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
                'bestFtce': best_ftce,
                'bestFtct': best_ftct,
                'Melhor_Acuracia_Validacao': best_acc,
                'RefinoFinal': 'Memoria de exemplares 1-NN (treino + validacao)',
            },
        ),
    )

    print('\nArquivos gerados:')
    print('  - 01_distribuicao_classes_cnapca.png')
    print('  - 02_subconjuntos_cnapca.png')
    print('  - 03_classe_e_subconjunto_cnapca.png')
    print('  - 04_visualizacao_3d_cnapca.png')
    print('  - 05_matriz_confusao_teste_cnapca.png')
    print('  - 06_desempenho_treinamento_cnapca.png')
    print('  - iris_com_particao_cnapca.csv')
    print('  - prototipos_cnapca.csv')
    print('  - grade_validacao_cnapca.csv')
    print('  - resultado_iris_cnapca_didatico.csv')
    print('  - modelo_iris_cnapca_didatico.joblib')
    print('  - resultado_iris_cnapca_didatico.json')
    print('\nFim da execucao.')


if __name__ == '__main__':
    main()
