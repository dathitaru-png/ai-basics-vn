"""Perceptron and a tiny multi-layer perceptron — from scratch.

Companion code for lesson 08 (``lessons/08-mang-no-ron-nho.md``).

- :class:`Perceptron` is a single artificial neuron: a weighted sum plus
  a step function. It learns by correcting its weights whenever it
  misclassifies a point. It can only learn linearly separable problems
  (AND, OR, ... — but not XOR).
- :class:`TinyMLP` adds one hidden layer of tanh neurons trained with
  backpropagation, which is enough to learn XOR — the classic example
  of why neural networks need more than one layer.
"""

from __future__ import annotations

import math
import random


class Perceptron:
    """A single neuron for binary classification (labels 0 and 1)."""

    def __init__(self, learning_rate: float = 1.0, epochs: int = 100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] = []
        self.bias: float = 0.0

    def _activate(self, row: list[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, row)) + self.bias

    def fit(self, X: list[list[float]], y: list[int]) -> "Perceptron":
        if not X or len(X) != len(y):
            raise ValueError("X and y must be non-empty and equally long")
        self.weights = [0.0] * len(X[0])
        self.bias = 0.0
        for _ in range(self.epochs):
            errors = 0
            for row, target in zip(X, y):
                pred = 1 if self._activate(row) >= 0 else 0
                update = self.learning_rate * (target - pred)
                if update != 0:
                    errors += 1
                    self.weights = [w + update * x for w, x in zip(self.weights, row)]
                    self.bias += update
            if errors == 0:
                break
        return self

    def predict_one(self, row: list[float]) -> int:
        if not self.weights:
            raise RuntimeError("call fit() before predict()")
        return 1 if self._activate(row) >= 0 else 0

    def predict(self, X: list[list[float]]) -> list[int]:
        return [self.predict_one(row) for row in X]


def _tanh(x: float) -> float:
    return math.tanh(x)


def _tanh_derivative(x: float) -> float:
    t = math.tanh(x)
    return 1.0 - t * t


class TinyMLP:
    """One-hidden-layer network (tanh) trained with backpropagation.

    Sized for toy problems such as XOR: a handful of inputs, a few
    hidden units, one output in (0, 1) via a sigmoid.
    """

    def __init__(self, hidden_dim: int = 4, learning_rate: float = 0.5,
                 epochs: int = 5000, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.loss_history: list[float] = []

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _sigmoid(x: float) -> float:
        if x < -60:
            return 0.0
        if x > 60:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def _forward(self, row):
        hidden_pre = [
            sum(w * x for w, x in zip(self._W1[h], row)) + self._b1[h]
            for h in range(self.hidden_dim)
        ]
        hidden = [_tanh(v) for v in hidden_pre]
        out_pre = sum(w * h for w, h in zip(self._W2, hidden)) + self._b2
        out = self._sigmoid(out_pre)
        return hidden_pre, hidden, out

    # -- training ------------------------------------------------------
    def fit(self, X: list[list[float]], y: list[int]) -> "TinyMLP":
        if not X or len(X) != len(y):
            raise ValueError("X and y must be non-empty and equally long")
        rng = random.Random(self.seed)
        d = len(X[0])
        self._W1 = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(self.hidden_dim)]
        self._b1 = [0.0] * self.hidden_dim
        self._W2 = [rng.uniform(-1, 1) for _ in range(self.hidden_dim)]
        self._b2 = 0.0
        self.loss_history = []
        for _ in range(self.epochs):
            total_loss = 0.0
            for row, target in zip(X, y):
                hidden_pre, hidden, out = self._forward(row)
                total_loss += (out - target) ** 2
                # Backpropagation (gradient of squared error).
                d_out = 2 * (out - target) * out * (1 - out)
                d_hidden = [
                    d_out * self._W2[h] * _tanh_derivative(hidden_pre[h])
                    for h in range(self.hidden_dim)
                ]
                for h in range(self.hidden_dim):
                    self._W2[h] -= self.learning_rate * d_out * hidden[h]
                    for j in range(d):
                        self._W1[h][j] -= self.learning_rate * d_hidden[h] * row[j]
                    self._b1[h] -= self.learning_rate * d_hidden[h]
                self._b2 -= self.learning_rate * d_out
            self.loss_history.append(total_loss / len(X))
        return self

    # -- inference -----------------------------------------------------
    def predict_proba_one(self, row: list[float]) -> float:
        if not hasattr(self, "_W1"):
            raise RuntimeError("call fit() before predict()")
        return self._forward(row)[2]

    def predict_one(self, row: list[float]) -> int:
        return 1 if self.predict_proba_one(row) >= 0.5 else 0

    def predict(self, X: list[list[float]]) -> list[int]:
        return [self.predict_one(row) for row in X]
