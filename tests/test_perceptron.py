"""Tests for Perceptron and TinyMLP (lesson 08)."""

from aibasics import Perceptron, TinyMLP, accuracy_score

GATES = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]


def test_perceptron_learns_and():
    y = [0, 0, 0, 1]
    model = Perceptron(epochs=50).fit(GATES, y)
    assert model.predict(GATES) == y


def test_perceptron_learns_or():
    y = [0, 1, 1, 1]
    model = Perceptron(epochs=50).fit(GATES, y)
    assert accuracy_score(y, model.predict(GATES)) == 1.0


def test_mlp_learns_xor():
    y = [0, 1, 1, 0]
    model = TinyMLP(hidden_dim=4, learning_rate=0.5, epochs=4000, seed=42).fit(GATES, y)
    assert model.predict(GATES) == y
    assert model.loss_history[-1] < model.loss_history[0]
