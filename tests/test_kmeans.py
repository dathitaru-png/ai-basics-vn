"""Tests for KMeans (lesson 07)."""

from aibasics import KMeans, make_blobs
from aibasics.knn import euclidean


def _match_score(labels, truth):
    """Best agreement between cluster ids and true labels (ids may swap)."""
    direct = sum(1 for a, b in zip(labels, truth) if a == b)
    swapped = sum(1 for a, b in zip(labels, truth) if a != b)
    return max(direct, swapped) / len(truth)


def test_kmeans_recovers_two_clusters():
    centers = ((0.0, 0.0), (8.0, 8.0))
    X, y = make_blobs(n_per_cluster=25, centers=centers, seed=42)
    model = KMeans(k=2, seed=42).fit(X)
    assert _match_score(model.labels_, y) >= 0.95
    found = sorted(model.centroids_)
    for got, expected in zip(found, centers):
        assert euclidean(list(got), list(expected)) < 1.0


def test_kmeans_recovers_three_clusters():
    centers = ((0.0, 0.0), (8.0, 0.0), (0.0, 8.0))
    X, y = make_blobs(n_per_cluster=20, centers=centers, seed=7)
    model = KMeans(k=3, seed=3).fit(X)
    # Every true centre has a found centre nearby.
    for expected in centers:
        assert min(euclidean(list(expected), c) for c in model.centroids_) < 1.0


def test_kmeans_predicts_consistently_with_fit():
    X, _ = make_blobs(n_per_cluster=15, seed=2)
    model = KMeans(k=2, seed=42).fit(X)
    assert model.predict(X) == model.labels_
    assert model.inertia_ > 0
