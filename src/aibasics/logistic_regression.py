"""Logistic regression trained with gradient descent — from scratch.

Companion code for lesson 11 (``lessons/11-hoi-quy-logistic.md``).

Logistic regression keeps the familiar linear score ``z = w·x + b`` from
lesson 03, but instead of predicting a number it squeezes ``z`` through
the sigmoid function into a probability between 0 and 1. It is trained
by minimising binary cross-entropy loss (``log loss``) with gradient
descent, exactly the "measure the slope, step downhill" idea from
lesson 03 — only the loss function changed.

Features are standardised internally (zero mean, unit variance) so a
single learning rate works reasonably for data on any scale; the
learned coefficients are converted back to the original scale for
inspection.
"""

from __future__ import annotations

import math


def sigmoid(z: float) -> float:
    """Squash any real number into the range (0, 1).

    The two branches are mathematically identical; the split only
    avoids ``math.exp`` overflowing on large negative inputs.
    """
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


class LogisticRegression:
    """Binary classifier fitted by batch gradient descent.

    Labels must be 0 or 1. :meth:`predict_proba` returns P(y = 1) for
    each row; :meth:`predict` thresholds those probabilities at 0.5.
    """

    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] = []
        self.bias: float = 0.0
        self.loss_history: list[float] = []
        self._mean: list[float] = []
        self._std: list[float] = []

    # -- training ------------------------------------------------------
    def fit(self, X: list[list[float]], y: list[int]) -> "LogisticRegression":
        if not X or len(X) != len(y):
            raise ValueError("X and y must be non-empty and equally long")
        if any(label not in (0, 1) for label in y):
            raise ValueError("labels must be 0 or 1")
        n = len(X)
        d = len(X[0])

        # Standardise each feature column (same trick as lesson 03).
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
            probs = [sigmoid(sum(wj * zj for wj, zj in zip(w, row)) + b) for row in Z]
            # Binary cross-entropy, with probabilities clipped away from
            # exactly 0 or 1 so log() never sees an impossible value.
            clipped = [min(max(p, 1e-15), 1.0 - 1e-15) for p in probs]
            loss = -sum(
                t * math.log(p) + (1 - t) * math.log(1 - p)
                for t, p in zip(y, clipped)
            ) / n
            self.loss_history.append(loss)
            errors = [p - t for p, t in zip(probs, y)]
            grad_w = [sum(errors[i] * Z[i][j] for i in range(n)) / n for j in range(d)]
            grad_b = sum(errors) / n
            w = [wj - self.learning_rate * g for wj, g in zip(w, grad_w)]
            b -= self.learning_rate * grad_b

        # Convert coefficients back to the original feature scale.
        self.weights = [w[j] / self._std[j] for j in range(d)]
        self.bias = b - sum(w[j] * self._mean[j] / self._std[j] for j in range(d))
        return self

    # -- inference -----------------------------------------------------
    def predict_proba_one(self, row: list[float]) -> float:
        if not self.weights:
            raise RuntimeError("call fit() before predict()")
        z = sum(w * x for w, x in zip(self.weights, row)) + self.bias
        return sigmoid(z)

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [self.predict_proba_one(row) for row in X]

    def predict_one(self, row: list[float]) -> int:
        return 1 if self.predict_proba_one(row) >= 0.5 else 0

    def predict(self, X: list[list[float]]) -> list[int]:
        return [self.predict_one(row) for row in X]
