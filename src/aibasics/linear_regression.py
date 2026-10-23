"""Linear regression trained with gradient descent — from scratch.

Companion code for lesson 03 (``lessons/03-hoi-quy-tuyen-tinh.md``).

The model learns ``y ≈ w·x + b`` by repeatedly nudging ``w`` and ``b``
against the gradient of the mean-squared-error loss. Features are
standardised internally (zero mean, unit variance) so a single learning
rate works reasonably for data on any scale; the learned coefficients
are converted back to the original scale for inspection.
"""

from __future__ import annotations

import math


class LinearRegression:
    """Multiple linear regression fitted by batch gradient descent."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] = []
        self.bias: float = 0.0
        self.loss_history: list[float] = []
        self._mean: list[float] = []
        self._std: list[float] = []

    # -- training ------------------------------------------------------
    def fit(self, X: list[list[float]], y: list[float]) -> "LinearRegression":
        if not X or len(X) != len(y):
            raise ValueError("X and y must be non-empty and equally long")
        n = len(X)
        d = len(X[0])

        # Standardise each feature column.
        self._mean = [sum(row[j] for row in X) / n for j in range(d)]
        self._std = []
        for j in range(d):
            var = sum((row[j] - self._mean[j]) ** 2 for row in X) / n
            self._std.append(math.sqrt(var) or 1.0)
        Z = [[(row[j] - self._mean[j]) / self._std[j] for j in range(d)] for row in X]

        w = [0.0] * d
        b = 0.0
        self.loss_history = []
        for _ in range(self.epochs):
            preds = [sum(wj * zj for wj, zj in zip(w, row)) + b for row in Z]
            errors = [p - t for p, t in zip(preds, y)]
            self.loss_history.append(sum(e * e for e in errors) / n)
            grad_w = [sum(errors[i] * Z[i][j] for i in range(n)) * 2 / n for j in range(d)]
            grad_b = sum(errors) * 2 / n
            w = [wj - self.learning_rate * g for wj, g in zip(w, grad_w)]
            b -= self.learning_rate * grad_b

        self._w_std, self._b_std = w, b
        # Convert coefficients back to the original feature scale.
        self.weights = [w[j] / self._std[j] for j in range(d)]
        self.bias = b - sum(w[j] * self._mean[j] / self._std[j] for j in range(d))
        return self

    # -- inference -----------------------------------------------------
    def predict_one(self, row: list[float]) -> float:
        if not self.weights:
            raise RuntimeError("call fit() before predict()")
        return sum(w * x for w, x in zip(self.weights, row)) + self.bias

    def predict(self, X: list[list[float]]) -> list[float]:
        return [self.predict_one(row) for row in X]
