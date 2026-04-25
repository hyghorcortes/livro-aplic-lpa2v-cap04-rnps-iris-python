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
        media_desvio_treino,
        montar_tabela_didatica_50_itens_por_classe,
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
    from .classificar_amostra_iris_cnapa import classificar_amostra_iris_cnapa
    from .prever_flor_iris_cnapa import prever_flor_iris_cnapa
except ImportError:
    from common import (
        NOMES_VARIAVEIS,
        carregar_base_iris,
        confusion_matrix_manual,
        gerar_graficos_didaticos_base,
        media_desvio_treino,
        montar_tabela_didatica_50_itens_por_classe,
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
    from classificar_amostra_iris_cnapa import classificar_amostra_iris_cnapa
    from prever_flor_iris_cnapa import prever_flor_iris_cnapa

SEED = 42
OUTPUT_DIR = THIS_DIR


def avaliar_conjunto(XN, prototipos, feature_range, class_names, ftc, ftct, memoria_X=None, memoria_y=None):
    pred = np.zeros(XN.shape[0], dtype=int)
    detalhes = []
    for ii in range(XN.shape[0]):
        _, idx_prev, tabela = classificar_amostra_iris_cnapa(
            XN[ii, :],
            prototipos,
            feature_range,
            class_names,
            ftc,
            ftct,
            memoria_X=memoria_X,
            memoria_y=memoria_y,
        )
        pred[ii] = idx_prev
        detalhes.append(tabela)
    return pred, detalhes


def main() -> None:
    print('\n=====================================================')
    print(' IRIS COM CNAPa (LPA2v) - VERSAO DIDATICA')
    print('=====================================================')

    X, y, class_names, base_df = carregar_base_iris()
    num_classes = len(class_names)

    print('\n[ETAPA 1] Base carregada.')
    print(base_df.head(10).to_string(index=True))

    train_idx, val_idx, test_idx, resumo_df = stratified_manual_split(y, seed=SEED)
    print('\n[ETAPA 4] Separacao estratificada manual (70/15/15)...')
    print(resumo_df.to_string(index=False))

    tpart = montar_tabela_particao(base_df, train_idx, val_idx, test_idx)
    tabela_50_itens = montar_tabela_didatica_50_itens_por_classe(tpart, class_names)
    gerar_graficos_didaticos_base(X, y, class_names, train_idx, val_idx, test_idx, OUTPUT_DIR, 'cnapa')
    Xtrain = X[train_idx]
    Xval = X[val_idx]
    Xtest = X[test_idx]

    print('\n[ETAPA 5] Tabela didatica com 50 itens e escala percentual...')
    print(tabela_50_itens.head(10).to_string(index=False))

    ytrain = y[train_idx]
    yval = y[val_idx]
    ytest = y[test_idx]

    print('\n[ETAPA 6] Normalizando com estatisticas do TREINO...')
    mu_train, sigma_train = media_desvio_treino(Xtrain)
    XtrainN = normalizar_zscore(Xtrain, mu_train, sigma_train)
    XvalN = normalizar_zscore(Xval, mu_train, sigma_train)
    XtestN = normalizar_zscore(Xtest, mu_train, sigma_train)

    feature_range = XtrainN.max(axis=0) - XtrainN.min(axis=0)
    feature_range[feature_range == 0] = 1.0

    print('\n[ETAPA 7] Formando prototipos das classes com os dados de treino...')
    prototipos = np.zeros((num_classes, XtrainN.shape[1]), dtype=float)
    for k in range(1, num_classes + 1):
        prototipos[k - 1, :] = XtrainN[ytrain == k, :].mean(axis=0)

    tabela_prototipos = pd.DataFrame(prototipos, columns=NOMES_VARIAVEIS)
    tabela_prototipos.insert(0, 'Classe', class_names)
    print(tabela_prototipos.to_string(index=False))

    print('\n[ETAPA 8] Ajustando Ftc e Ftct no conjunto de validacao...')
    grid_ftc = np.round(np.arange(0.0, 1.01, 0.1), 2)
    grid_ftct = np.round(np.arange(0.0, 1.01, 0.1), 2)
    acc_grid = np.zeros((len(grid_ftct), len(grid_ftc)), dtype=float)

    for i_ftct, ftct in enumerate(grid_ftct):
        for i_ftc, ftc in enumerate(grid_ftc):
            pred_val = np.zeros(len(val_idx), dtype=int)
            for i in range(len(val_idx)):
                _, idx_prev, _ = classificar_amostra_iris_cnapa(XvalN[i, :], prototipos, feature_range, class_names, float(ftc), float(ftct))
                pred_val[i] = idx_prev
            acc_grid[i_ftct, i_ftc] = float(np.mean(pred_val == yval))

    idx_best = np.unravel_index(np.argmax(acc_grid), acc_grid.shape)
    ftct_best = float(grid_ftct[idx_best[0]])
    ftc_best = float(grid_ftc[idx_best[1]])
    max_acc_val = float(acc_grid[idx_best])

    print('Melhor validacao encontrada:')
    print(f'  Ftc  = {ftc_best:.2f}')
    print(f'  Ftct = {ftct_best:.2f}')
    print(f'  Acuracia validacao = {100 * max_acc_val:.2f} %')

    print('\n[ETAPA 9] Reajustando normalizacao, prototipos e memoria com TREINO + VALIDACAO...')
    fit_idx = np.concatenate([train_idx, val_idx])
    Xfit = X[fit_idx]
    yfit = y[fit_idx]
    mu_train, sigma_train = media_desvio_treino(Xfit)
    XfitN = normalizar_zscore(Xfit, mu_train, sigma_train)
    XtrainN = normalizar_zscore(Xtrain, mu_train, sigma_train)
    XvalN = normalizar_zscore(Xval, mu_train, sigma_train)
    XtestN = normalizar_zscore(Xtest, mu_train, sigma_train)
    feature_range = XfitN.max(axis=0) - XfitN.min(axis=0)
    feature_range[feature_range == 0] = 1.0

    prototipos = np.zeros((num_classes, XfitN.shape[1]), dtype=float)
    for k in range(1, num_classes + 1):
        prototipos[k - 1, :] = XfitN[yfit == k, :].mean(axis=0)

    tabela_prototipos = pd.DataFrame(prototipos, columns=NOMES_VARIAVEIS)
    tabela_prototipos.insert(0, 'Classe', class_names)

    memoria_X = XfitN
    memoria_y = yfit
    pred_train, detalhes_train = avaliar_conjunto(
        XtrainN,
        prototipos,
        feature_range,
        class_names,
        ftc_best,
        ftct_best,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )
    pred_val, detalhes_val = avaliar_conjunto(
        XvalN,
        prototipos,
        feature_range,
        class_names,
        ftc_best,
        ftct_best,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )
    pred_test, detalhes_test = avaliar_conjunto(
        XtestN,
        prototipos,
        feature_range,
        class_names,
        ftc_best,
        ftct_best,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )

    acc_train = float(np.mean(pred_train == ytrain))
    acc_val = float(np.mean(pred_val == yval))
    acc_test = float(np.mean(pred_test == ytest))
    acc_total = float(np.mean(np.concatenate([pred_train, pred_val, pred_test]) == np.concatenate([ytrain, yval, ytest])))

    print('\n============= RESULTADOS =============')
    print(f'Acuracia treino    : {100 * acc_train:.2f} %')
    print(f'Acuracia validacao : {100 * acc_val:.2f} %')
    print(f'Acuracia teste     : {100 * acc_test:.2f} %')
    print(f'Acuracia global    : {100 * acc_total:.2f} %')
    print('Refino final       : memoria de exemplares 1-NN com treino + validacao')

    idx_exemplo = 0
    classe_prevista_ex, idx_prev_ex, tabela_detalhes_ex = classificar_amostra_iris_cnapa(
        XtestN[idx_exemplo, :],
        prototipos,
        feature_range,
        class_names,
        ftc_best,
        ftct_best,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )
    print('\n[ETAPA 10] Detalhamento CNAPa de uma amostra de teste...')
    print('Amostra de teste escolhida (original):')
    print(Xtest[idx_exemplo, :])
    print(f'Classe real      : {class_names[ytest[idx_exemplo] - 1]}')
    print(f'Classe prevista  : {classe_prevista_ex}')
    print(tabela_detalhes_ex.to_string(index=False))

    C = confusion_matrix_manual(ytest, pred_test, num_classes)
    plot_confusion_matrix_padrao(
        C,
        class_names,
        OUTPUT_DIR / nome_arquivo_figura(5, 'matriz_confusao_teste', 'cnapa'),
        'Matriz de confusao - conjunto de teste',
    )

    plot_desempenho_resumido(
        {
            'Treino': acc_train,
            'Validacao': acc_val,
            'Teste': acc_test,
            'Total': acc_total,
        },
        OUTPUT_DIR / nome_arquivo_figura(6, 'desempenho_treinamento', 'cnapa'),
        'Desempenho do CNAPa',
        [
            f'Ftc = {ftc_best:.2f}',
            f'Ftct = {ftct_best:.2f}',
            f'Melhor validacao = {100 * max_acc_val:.2f}%',
            'Refino final = memoria 1-NN em treino + validacao',
        ],
    )

    nova_amostra = np.array([5.1, 3.5, 1.4, 0.2], dtype=float)
    classe_prevista_nova, tabela_nova = prever_flor_iris_cnapa(
        nova_amostra,
        mu_train,
        sigma_train,
        prototipos,
        feature_range,
        class_names,
        ftc_best,
        ftct_best,
        memoria_X=memoria_X,
        memoria_y=memoria_y,
    )
    print('\n[ETAPA 11] Teste com nova amostra...')
    print(f'Classe prevista = {classe_prevista_nova}')
    print(tabela_nova.to_string(index=False))

    tpart.to_csv(OUTPUT_DIR / 'iris_com_particao_cnapa.csv', index=False)
    tabela_50_itens.to_csv(OUTPUT_DIR / 'tabela_50_itens_cnapa.csv', index=False)
    tabela_prototipos.to_csv(OUTPUT_DIR / 'prototipos_cnapa.csv', index=False)
    lin, col = np.meshgrid(grid_ftct, grid_ftc, indexing='ij')
    tgrid = pd.DataFrame({
        'Ftct': lin.ravel(),
        'Ftc': col.ravel(),
        'Acuracia_Validacao': acc_grid.ravel(),
    })
    tgrid.to_csv(OUTPUT_DIR / 'grade_validacao_cnapa.csv', index=False)
    montar_tabela_resumo_resultado(
        'CNAPa',
        acc_total,
        acc_train,
        acc_val,
        acc_test,
        C,
        {
            'FtcBest': ftc_best,
            'FtctBest': ftct_best,
            'Melhor_Acuracia_Validacao': max_acc_val,
            'RefinoFinal': 'Memoria de exemplares 1-NN (treino + validacao)',
        },
    ).to_csv(OUTPUT_DIR / 'resultado_iris_cnapa_didatico.csv', index=False)

    joblib.dump({
        'class_names': class_names,
        'mu_train': mu_train,
        'sigma_train': sigma_train,
        'feature_range': feature_range,
        'prototipos': prototipos,
        'FtcBest': ftc_best,
        'FtctBest': ftct_best,
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
    }, OUTPUT_DIR / 'modelo_iris_cnapa_didatico.joblib')

    salvar_json(
        OUTPUT_DIR / 'resultado_iris_cnapa_didatico.json',
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
                'FtcBest': ftc_best,
                'FtctBest': ftct_best,
                'Melhor_Acuracia_Validacao': max_acc_val,
                'RefinoFinal': 'Memoria de exemplares 1-NN (treino + validacao)',
            },
        ),
    )

    print('\nArquivos gerados:')
    print('  - 01_distribuicao_classes_cnapa.png')
    print('  - 02_subconjuntos_cnapa.png')
    print('  - 03_classe_e_subconjunto_cnapa.png')
    print('  - 04_visualizacao_3d_cnapa.png')
    print('  - 05_matriz_confusao_teste_cnapa.png')
    print('  - 06_desempenho_treinamento_cnapa.png')
    print('  - iris_com_particao_cnapa.csv')
    print('  - tabela_50_itens_cnapa.csv')
    print('  - prototipos_cnapa.csv')
    print('  - grade_validacao_cnapa.csv')
    print('  - resultado_iris_cnapa_didatico.csv')
    print('  - modelo_iris_cnapa_didatico.joblib')
    print('  - resultado_iris_cnapa_didatico.json')
    print('\nFim da execucao.')


if __name__ == '__main__':
    main()
