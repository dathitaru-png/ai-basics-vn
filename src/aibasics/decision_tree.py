"""A small decision-tree classifier (CART-style) — from scratch.

Companion code for lesson 05 (``lessons/05-cay-quyet-dinh.md``).

The tree repeatedly asks yes/no questions of the form
"is feature j <= threshold?" and picks the question that makes the two
resulting groups as "pure" as possible, measured with Gini impurity.
Growth stops at ``max_depth`` or when a node has too few samples.
"""

from __future__ import annotations

from collections import Counter


def _gini(labels: list) -> float:
    """Gini impurity: 0.0 when all labels are the same."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _majority(labels: list):
    return Counter(labels).most_common(1)[0][0]


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "label")

    def __init__(self, label, feature=None, threshold=None, left=None, right=None):
        self.label = label          # prediction if this is a leaf
        self.feature = feature      # feature index to split on (None = leaf)
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTreeClassifier:
    """Binary-split decision tree for numeric features."""

    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: _Node | None = None

    # -- training ------------------------------------------------------
    def fit(self, X: list[list[float]], y: list) -> "DecisionTreeClassifier":
        if not X or len(X) != len(y):
            raise ValueError("X and y must be non-empty and equally long")
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth: int) -> _Node:
        label = _majority(y)
        if (
            depth >= self.max_depth
            or len(set(y)) == 1
            or len(y) < self.min_samples_split
        ):
            return _Node(label)
        split = self._best_split(X, y)
        if split is None:
            return _Node(label)
        feature, threshold, left_idx, right_idx = split
        left = self._build([X[i] for i in left_idx], [y[i] for i in left_idx], depth + 1)
        right = self._build([X[i] for i in right_idx], [y[i] for i in right_idx], depth + 1)
        return _Node(label, feature, threshold, left, right)

    def _best_split(self, X, y):
        n = len(y)
        n_features = len(X[0])
        best = None  # (weighted impurity, feature, threshold, left_idx, right_idx)
        for j in range(n_features):
            values = sorted({row[j] for row in X})
            for a, b in zip(values, values[1:]):
                threshold = (a + b) / 2
                left_idx = [i for i in range(n) if X[i][j] <= threshold]
                right_idx = [i for i in range(n) if X[i][j] > threshold]
                if not left_idx or not right_idx:
                    continue
                impurity = (
                    len(left_idx) * _gini([y[i] for i in left_idx])
                    + len(right_idx) * _gini([y[i] for i in right_idx])
                ) / n
                if best is None or impurity < best[0]:
                    best = (impurity, j, threshold, left_idx, right_idx)
        if best is None:
            return None
        _, feature, threshold, left_idx, right_idx = best
        return feature, threshold, left_idx, right_idx

    # -- inference -----------------------------------------------------
    def predict_one(self, row: list[float]):
        if self.root is None:
            raise RuntimeError("call fit() before predict()")
        node = self.root
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.label

    def predict(self, X: list[list[float]]) -> list:
        return [self.predict_one(row) for row in X]
