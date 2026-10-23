"""k-Nearest Neighbours classifier — from scratch.

Companion code for lesson 04 (``lessons/04-knn-phan-loai.md``).

k-NN does not really "train": it memorises the training points. To label
a new point it measures the distance to every memorised point, takes the
``k`` closest ones, and lets them vote.
"""

from __future__ import annotations

import math
from collections import Counter


def euclidean(a: list[float], b: list[float]) -> float:
    """Straight-line distance between two points."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class KNeighborsClassifier:
    """Classify points by majority vote of their ``k`` nearest neighbours."""

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self._X: list[list[float]] = []
        self._y: list = []

    def fit(self, X: list[list[float]], y: list) -> "KNeighborsClassifier":
        if not X or len(X) != len(y):
            raise ValueError("X and y must be non-empty and equally long")
        self._X = [list(row) for row in X]
        self._y = list(y)
        return self

    def predict_one(self, row: list[float]):
        if not self._X:
            raise RuntimeError("call fit() before predict()")
        neighbours = sorted(
            ((euclidean(row, train_row), label)
             for train_row, label in zip(self._X, self._y)),
            key=lambda pair: pair[0],
        )[: self.k]
        votes = Counter(label for _, label in neighbours)
        return votes.most_common(1)[0][0]

    def predict(self, X: list[list[float]]) -> list:
        return [self.predict_one(row) for row in X]
