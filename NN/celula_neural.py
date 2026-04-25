from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=int).reshape(-1)
    T = np.zeros((len(y), num_classes), dtype=float)
    T[np.arange(len(y)), y - 1] = 1.0
    return T


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def _ativacao(z: np.ndarray, nome: str) -> np.ndarray:
    if nome == 'tanh':
        return np.tanh(z)
    if nome == 'logistic':
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
    if nome == 'relu':
        return np.maximum(0.0, z)
    if nome == 'linear':
        return z
    raise ValueError(f'Ativacao desconhecida: {nome}')


def _derivada_ativacao(a: np.ndarray, z: np.ndarray, nome: str) -> np.ndarray:
    if nome == 'tanh':
        return 1.0 - a * a
    if nome == 'logistic':
        return a * (1.0 - a)
    if nome == 'relu':
        return (z > 0.0).astype(float)
    if nome == 'linear':
        return np.ones_like(a)
    raise ValueError(f'Ativacao desconhecida: {nome}')


@dataclass
class RedeNeuralDidatica:
    """Rede neural simples: celula oculta + camada softmax de saida."""

    num_neuronios_ocultos: int
    ativacao: str
    taxa_aprendizado: float
    num_epocas: int
    alpha: float = 1e-4
    seed: int = 42

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int | None = None) -> 'RedeNeuralDidatica':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        if X.ndim != 2:
            raise ValueError('X deve ser uma matriz 2D.')
        if len(X) != len(y):
            raise ValueError('X e y devem ter o mesmo numero de amostras.')

        self.num_features_ = X.shape[1]
        self.num_classes_ = int(num_classes or np.max(y))
        self.classes_ = np.arange(1, self.num_classes_ + 1, dtype=int)
        self.loss_curve_ = []

        rng = np.random.default_rng(self.seed)
        T = _one_hot(y, self.num_classes_)

        if self.num_neuronios_ocultos == 0:
            self.W1_ = None
            self.b1_ = None
            self.W2_ = rng.normal(0.0, 0.1, size=(self.num_features_, self.num_classes_))
            self.b2_ = np.zeros(self.num_classes_, dtype=float)
        else:
            escala_entrada = np.sqrt(2.0 / (self.num_features_ + self.num_neuronios_ocultos))
            escala_saida = np.sqrt(2.0 / (self.num_neuronios_ocultos + self.num_classes_))
            self.W1_ = rng.normal(0.0, escala_entrada, size=(self.num_features_, self.num_neuronios_ocultos))
            self.b1_ = np.zeros(self.num_neuronios_ocultos, dtype=float)
            self.W2_ = rng.normal(0.0, escala_saida, size=(self.num_neuronios_ocultos, self.num_classes_))
            self.b2_ = np.zeros(self.num_classes_, dtype=float)

        for _ in range(int(self.num_epocas)):
            cache = self._forward(X)
            P = cache['probabilidades']
            loss = self._loss(P, T)
            self.loss_curve_.append(loss)

            n = X.shape[0]
            dZ2 = (P - T) / n

            if self.num_neuronios_ocultos == 0:
                dW2 = X.T @ dZ2 + self.alpha * self.W2_
                db2 = dZ2.sum(axis=0)
                self.W2_ -= self.taxa_aprendizado * dW2
                self.b2_ -= self.taxa_aprendizado * db2
            else:
                A1 = cache['ativacao_oculta']
                Z1 = cache['entrada_oculta']
                dW2 = A1.T @ dZ2 + self.alpha * self.W2_
                db2 = dZ2.sum(axis=0)
                dA1 = dZ2 @ self.W2_.T
                dZ1 = dA1 * _derivada_ativacao(A1, Z1, self.ativacao)
                dW1 = X.T @ dZ1 + self.alpha * self.W1_
                db1 = dZ1.sum(axis=0)

                self.W2_ -= self.taxa_aprendizado * dW2
                self.b2_ -= self.taxa_aprendizado * db2
                self.W1_ -= self.taxa_aprendizado * dW1
                self.b1_ -= self.taxa_aprendizado * db1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._as_2d(X)
        return self._forward(X)['probabilidades']

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1).astype(int) + 1

    def detalhar(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        X = self._as_2d(X)
        return self._forward(X)

    def parametros(self) -> Dict[str, float | int | str]:
        return {
            'num_neuronios_ocultos': int(self.num_neuronios_ocultos),
            'ativacao': self.ativacao,
            'taxa_aprendizado': float(self.taxa_aprendizado),
            'num_epocas': int(self.num_epocas),
            'alpha': float(self.alpha),
            'seed': int(self.seed),
        }

    def _forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if self.num_neuronios_ocultos == 0:
            logits = X @ self.W2_ + self.b2_
            return {
                'entrada_oculta': np.empty((X.shape[0], 0), dtype=float),
                'ativacao_oculta': np.empty((X.shape[0], 0), dtype=float),
                'logits': logits,
                'probabilidades': _softmax(logits),
            }

        entrada_oculta = X @ self.W1_ + self.b1_
        ativacao_oculta = _ativacao(entrada_oculta, self.ativacao)
        logits = ativacao_oculta @ self.W2_ + self.b2_
        return {
            'entrada_oculta': entrada_oculta,
            'ativacao_oculta': ativacao_oculta,
            'logits': logits,
            'probabilidades': _softmax(logits),
        }

    def _loss(self, P: np.ndarray, T: np.ndarray) -> float:
        eps = np.finfo(float).eps
        ce = -np.sum(T * np.log(P + eps)) / T.shape[0]
        reg = 0.5 * self.alpha * np.sum(self.W2_ * self.W2_)
        if self.W1_ is not None:
            reg += 0.5 * self.alpha * np.sum(self.W1_ * self.W1_)
        return float(ce + reg)

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return X.reshape(1, -1)
        if X.ndim == 2:
            return X
        raise ValueError('A entrada deve ser vetor 1D ou matriz 2D.')
