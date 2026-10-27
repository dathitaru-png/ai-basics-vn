"""Tests for LogisticRegression (lesson 11)."""

import pytest

from aibasics import LogisticRegression, accuracy_score, make_blobs, train_test_split


def test_logistic_loss_decreases():
    X = [[float(h)] for h in range(1, 9)]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    model = LogisticRegression(learning_rate=0.1, epochs=1000).fit(X, y)
    assert model.loss_history[-1] < model.loss_history[0]
    assert model.loss_history[0] == 0.6931471805599453  # starts at 50/50 guessing
    assert model.loss_history[-1] < 0.1


def test_logistic_learns_separable_problem():
    X, y = make_blobs(n_per_cluster=25, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=1)
    model = LogisticRegression(learning_rate=0.1, epochs=1000).fit(X_train, y_train)
    assert accuracy_score(y_test, model.predict(X_test)) >= 0.95


def test_logistic_probabilities_are_valid():
    X = [[float(h)] for h in range(1, 9)]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    model = LogisticRegression(learning_rate=0.1, epochs=1000).fit(X, y)
    probs = model.predict_proba(X)
    assert all(0.0 <= p <= 1.0 for p in probs)
    # More study hours -> higher pass probability (monotone here).
    assert probs == sorted(probs)
    assert model.predict_proba_one([4.0]) < 0.5 < model.predict_proba_one([6.0])


def test_logistic_rejects_bad_labels():
    with pytest.raises(ValueError):
        LogisticRegression().fit([[1.0], [2.0]], [0, 2])


def test_logistic_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LogisticRegression().predict_one([1.0])
