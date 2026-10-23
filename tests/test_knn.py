"""Tests for KNeighborsClassifier (lesson 04)."""

from aibasics import KNeighborsClassifier, accuracy_score, make_blobs, train_test_split


def test_knn_classifies_obvious_clusters():
    X, y = make_blobs(n_per_cluster=25, centers=((0.0, 0.0), (10.0, 10.0)), seed=42)
    model = KNeighborsClassifier(k=3).fit(X, y)
    assert accuracy_score(y, model.predict(X)) == 1.0


def test_knn_generalises_to_held_out_points():
    X, y = make_blobs(n_per_cluster=30, seed=11)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=5)
    model = KNeighborsClassifier(k=3).fit(X_train, y_train)
    assert accuracy_score(y_test, model.predict(X_test)) >= 0.95


def test_knn_predicts_nearby_point_label():
    X, y = make_blobs(n_per_cluster=10, centers=((0.0, 0.0), (10.0, 0.0)), seed=1)
    model = KNeighborsClassifier(k=1).fit(X, y)
    assert model.predict_one([9.5, 0.5]) == 1
    assert model.predict_one([0.5, -0.5]) == 0
