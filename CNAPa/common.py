from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

NOMES_VARIAVEIS = [
    'Sepala_Comprimento',
    'Sepala_Largura',
    'Petala_Comprimento',
    'Petala_Largura',
]
CORES_CLASSES = ['tab:blue', 'tab:orange', 'tab:green']


def nome_arquivo_figura(numero: int, descricao: str, suffix: str) -> str:
    return f'{numero:02d}_{descricao}_{suffix}.png'


def carregar_base_iris() -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Carrega a base Iris e organiza os dados em formato tabular."""
    ds = load_iris()
    X = ds.data.astype(float)
    y0 = ds.target.astype(int)
    class_names = [str(name) for name in ds.target_names.tolist()]
    y = y0 + 1  # Classes numeradas de 1..K

    df = pd.DataFrame(X, columns=NOMES_VARIAVEIS)
    df['Especie'] = [class_names[i] for i in y0]
    return X, y, class_names, df


def stratified_manual_split(
    y: Sequence[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Realiza separação estratificada manual com arredondamento.
    Retorna índices globais 0-based para treino, validação e teste.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    resumo = []

    for cls in classes:
        idx_classe = np.flatnonzero(y == cls)
        idx_classe = rng.permutation(idx_classe)

        nk = len(idx_classe)
        n_train = int(round(train_ratio * nk))
        n_val = int(round(val_ratio * nk))
        n_test = nk - n_train - n_val

        idx_train_k = idx_classe[:n_train]
        idx_val_k = idx_classe[n_train:n_train + n_val]
        idx_test_k = idx_classe[n_train + n_val:]

        train_idx.extend(idx_train_k.tolist())
        val_idx.extend(idx_val_k.tolist())
        test_idx.extend(idx_test_k.tolist())

        resumo.append({
            'Classe_ID': int(cls),
            'Total': nk,
            'Treino': n_train,
            'Validacao': n_val,
            'Teste': n_test,
        })

    train_idx = rng.permutation(np.asarray(train_idx, dtype=int))
    val_idx = rng.permutation(np.asarray(val_idx, dtype=int))
    test_idx = rng.permutation(np.asarray(test_idx, dtype=int))

    resumo_df = pd.DataFrame(resumo)
    return train_idx, val_idx, test_idx, resumo_df


def montar_tabela_particao(
    base_df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    subset = np.empty(len(base_df), dtype=object)
    subset[:] = ''
    subset[train_idx] = 'Treino'
    subset[val_idx] = 'Validacao'
    subset[test_idx] = 'Teste'
    out = base_df.copy()
    out['Subconjunto'] = subset
    return out


def montar_tabela_didatica_50_itens_por_classe(
    tabela_particao: pd.DataFrame,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """
    Cria uma tabela compacta de 50 itens por classe, com escala percentual
    de 0 a 100, para facilitar a leitura didatica da base Iris.
    """
    blocos: Dict[str, pd.DataFrame] = {}
    for classe in class_names:
        bloco = tabela_particao.loc[tabela_particao['Especie'] == classe].copy().reset_index(drop=False)
        if len(bloco) != 50:
            raise ValueError(f'A classe {classe} deveria ter 50 itens na base Iris, mas tem {len(bloco)}.')
        blocos[classe] = bloco

    item_1a50 = np.arange(1, 51, dtype=int)
    percentual = np.round(np.linspace(0.0, 100.0, 50), 2)
    tabela = pd.DataFrame({
        'Item_1a50': item_1a50,
        'Percentual_0a100': percentual,
    })

    for classe in class_names:
        rotulo = str(classe).capitalize()
        bloco = blocos[classe]
        tabela[f'{rotulo}_Indice_Base_1a150'] = bloco['index'].to_numpy() + 1
        tabela[f'{rotulo}_Subconjunto'] = bloco['Subconjunto'].to_numpy()

    return tabela


def media_desvio_treino(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def normalizar_zscore(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def normalizar_minmax(X: np.ndarray, xmin: np.ndarray, xrange: np.ndarray) -> np.ndarray:
    X01 = (X - xmin) / xrange
    return np.clip(X01, 0.0, 1.0)


def minmax_treino(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin = X_train.min(axis=0)
    xmax = X_train.max(axis=0)
    xrange = xmax - xmin
    xrange[xrange == 0] = 1.0
    return xmin, xmax, xrange


def one_hot(y: Sequence[int], num_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    T = np.zeros((len(y), num_classes), dtype=float)
    T[np.arange(len(y)), y - 1] = 1.0
    return T


def classificar_por_memoria_exemplares(
    X: np.ndarray,
    memoria_X: np.ndarray,
    memoria_y: Sequence[int],
    num_classes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    memoria_X = np.asarray(memoria_X, dtype=float)
    memoria_y = np.asarray(memoria_y, dtype=int).reshape(-1)
    if memoria_X.ndim != 2:
        raise ValueError('memoria_X deve ser uma matriz 2D.')
    if len(memoria_X) != len(memoria_y):
        raise ValueError('memoria_X e memoria_y devem ter o mesmo numero de amostras.')

    dist2 = np.sum((X[:, None, :] - memoria_X[None, :, :]) ** 2, axis=2)
    idx_memoria = np.argmin(dist2, axis=1)
    y_pred = memoria_y[idx_memoria]
    dist_min = np.sqrt(dist2[np.arange(X.shape[0]), idx_memoria])

    num_classes = int(num_classes or np.max(memoria_y))
    menores_distancias = np.full((X.shape[0], num_classes), np.inf, dtype=float)
    for cls in range(1, num_classes + 1):
        mask = memoria_y == cls
        if np.any(mask):
            menores_distancias[:, cls - 1] = np.sqrt(np.min(dist2[:, mask], axis=1))

    scores = np.exp(-menores_distancias)
    soma = scores.sum(axis=1, keepdims=True)
    soma[soma == 0] = 1.0
    probabilidades = scores / soma
    return y_pred, idx_memoria, dist_min, probabilidades


def confusion_matrix_manual(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    C = np.zeros((num_classes, num_classes), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        C[yt - 1, yp - 1] += 1
    return C


def gerar_graficos_didaticos_base(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
    suffix: str,
) -> None:
    """Gera o mesmo pacote didatico de figuras usado no exemplo da rede neural."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    for k in range(1, len(class_names) + 1):
        mask = y == k
        plt.scatter(
            X[mask, 2],
            X[mask, 3],
            s=60,
            color=CORES_CLASSES[k - 1],
            label=class_names[k - 1],
        )
    plt.grid(True)
    plt.xlabel('Comprimento da Petala')
    plt.ylabel('Largura da Petala')
    plt.title('Separacao natural das classes')
    plt.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output_dir / nome_arquivo_figura(1, 'distribuicao_classes', suffix), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(X[train_idx, 2], X[train_idx, 3], 'o', markersize=6, label='Treino')
    plt.plot(X[val_idx, 2], X[val_idx, 3], 's', markersize=6, label='Validacao')
    plt.plot(X[test_idx, 2], X[test_idx, 3], '^', markersize=6, label='Teste')
    plt.grid(True)
    plt.xlabel('Comprimento da Petala')
    plt.ylabel('Largura da Petala')
    plt.title('Quem foi para treino, validacao e teste')
    plt.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output_dir / nome_arquivo_figura(2, 'subconjuntos', suffix), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    markers = np.array([''] * len(X), dtype=object)
    markers[train_idx] = 'o'
    markers[val_idx] = 's'
    markers[test_idx] = '^'
    for i in range(len(X)):
        plt.scatter(X[i, 2], X[i, 3], marker=markers[i], s=70, color=CORES_CLASSES[y[i] - 1])
    plt.grid(True)
    plt.xlabel('Comprimento da Petala')
    plt.ylabel('Largura da Petala')
    plt.title('Cor = classe | Formato = subconjunto')
    fig.tight_layout()
    fig.savefig(output_dir / nome_arquivo_figura(3, 'classe_e_subconjunto', suffix), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(X)):
        ax.scatter(X[i, 0], X[i, 2], X[i, 3], marker=markers[i], s=70, color=CORES_CLASSES[y[i] - 1])
    ax.set_xlabel('Sepala Comprimento')
    ax.set_ylabel('Petala Comprimento')
    ax.set_zlabel('Petala Largura')
    ax.set_title('Visualizacao 3D - formato = subconjunto')
    fig.tight_layout()
    fig.savefig(output_dir / nome_arquivo_figura(4, 'visualizacao_3d', suffix), dpi=150)
    plt.close(fig)


def plot_confusion_matrix_padrao(C: np.ndarray, class_names: Sequence[str], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(C)
    plt.colorbar()
    plt.xlabel('Classe prevista')
    plt.ylabel('Classe real')
    plt.title(title)
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    for r in range(C.shape[0]):
        for c in range(C.shape[1]):
            plt.text(c, r, str(C[r, c]), ha='center', va='center', color='white', fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_desempenho_resumido(
    metricas: Dict[str, float],
    path: Path,
    title: str,
    linhas_extras: Sequence[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    labels = list(metricas.keys())
    valores = 100.0 * np.asarray([metricas[label] for label in labels], dtype=float)
    x = np.arange(len(labels), dtype=float)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(x, valores, marker='o', linewidth=1.8, color='tab:blue')
    plt.xticks(x, labels)
    plt.ylim(0.0, 105.0)
    plt.ylabel('Acuracia (%)')
    plt.title(title)
    plt.grid(True, axis='y')

    for xi, valor in zip(x, valores):
        plt.text(xi, min(valor + 2.0, 103.0), f'{valor:.2f}%', ha='center', va='bottom')

    if linhas_extras:
        texto = '\n'.join(str(linha) for linha in linhas_extras)
        plt.gca().text(
            0.02,
            0.98,
            texto,
            transform=plt.gca().transAxes,
            ha='left',
            va='top',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9},
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def montar_payload_resultado(
    class_names: Sequence[str],
    acc_total: float,
    acc_train: float,
    acc_val: float,
    acc_test: float,
    confusion_matrix: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    extras: Dict | None = None,
) -> Dict:
    payload = {
        'class_names': list(class_names),
        'acc_total': float(acc_total),
        'acc_train': float(acc_train),
        'acc_val': float(acc_val),
        'acc_test': float(acc_test),
        'confusion_matrix': confusion_matrix,
        'train_idx_1based': np.asarray(train_idx, dtype=int) + 1,
        'val_idx_1based': np.asarray(val_idx, dtype=int) + 1,
        'test_idx_1based': np.asarray(test_idx, dtype=int) + 1,
    }
    if extras:
        payload.update(extras)
    return payload


def montar_tabela_resumo_resultado(
    algoritmo: str,
    acc_total: float,
    acc_train: float,
    acc_val: float,
    acc_test: float,
    confusion_matrix: np.ndarray,
    extras: Dict | None = None,
) -> pd.DataFrame:
    linha = {
        'Algoritmo': algoritmo,
        'Acuracia_Total': float(acc_total),
        'Acuracia_Treino': float(acc_train),
        'Acuracia_Validacao': float(acc_val),
        'Acuracia_Teste': float(acc_test),
    }

    if extras:
        linha.update(extras)

    C = np.asarray(confusion_matrix, dtype=int)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            linha[f'Conf_{i + 1}{j + 1}'] = int(C[i, j])

    return pd.DataFrame([linha])


def salvar_json(path: Path, payload: Dict) -> None:
    import json

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=convert)
