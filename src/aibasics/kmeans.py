"""k-Means clustering — from scratch.

Companion code for lesson 07 (``lessons/07-kmeans-phan-cum.md``).

k-Means groups points without any labels: start with ``k`` centres,
assign every point to its nearest centre, move each centre to the
average of its points, and repeat until things settle. The initial
centres are chosen with the k-means++ strategy (spread out, weighted
by distance) using a seeded random generator, so results are
reproducible for a given ``seed``.
"""

from __future__ import annotations

import random

from .knn import euclidean


def _init_plus_plus(X, k: int, rng: random.Random) -> list[list[float]]:
    """k-means++ seeding: pick centres that start out spread apart."""
    centroids = [list(rng.choice(X))]
    while len(centroids) < k:
        weights = [
            min(euclidean(row, c) ** 2 for c in centroids) for row in X
        ]
        total = sum(weights)
        if total == 0:  # all points coincide with chosen centres
            centroids.append(list(rng.choice(X)))
            continue
        pick = rng.uniform(0, total)
        cumulative = 0.0
        for row, w in zip(X, weights):
            cumulative += w
            if cumulative >= pick:
                centroids.append(list(row))
                break
    return centroids


class KMeans:
    """Group points into ``k`` clusters by nearest-centre assignment."""

    def __init__(self, k: int = 2, max_iter: int = 100, seed: int = 42):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centroids_: list[list[float]] = []
        self.labels_: list[int] = []
        self.inertia_: float = 0.0  # sum of squared distances to centres
        self.n_iter_: int = 0

    def fit(self, X: list[list[float]]) -> "KMeans":
        if not X:
            raise ValueError("X must be non-empty")
        if self.k > len(X):
            raise ValueError("k cannot exceed the number of points")
        rng = random.Random(self.seed)
        centroids = _init_plus_plus(X, self.k, rng)
        labels = [0] * len(X)
        for iteration in range(1, self.max_iter + 1):
            # Assign each point to its nearest centre.
            labels = [
                min(range(self.k), key=lambda c: euclidean(row, centroids[c]))
                for row in X
            ]
            # Move each centre to the mean of its assigned points.
            new_centroids = []
            for c in range(self.k):
                members = [row for row, lab in zip(X, labels) if lab == c]
                if members:
                    dim = len(members[0])
                    new_centroids.append(
                        [sum(row[j] for row in members) / len(members) for j in range(dim)]
                    )
                else:  # empty cluster: keep the old centre
                    new_centroids.append(centroids[c])
            self.n_iter_ = iteration
            if new_centroids == centroids:
                break
            centroids = new_centroids
        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = sum(
            euclidean(row, centroids[lab]) ** 2 for row, lab in zip(X, labels)
        )
        return self

    def predict(self, X: list[list[float]]) -> list[int]:
        if not self.centroids_:
            raise RuntimeError("call fit() before predict()")
        return [
            min(range(self.k), key=lambda c: euclidean(row, self.centroids_[c]))
            for row in X
        ]
