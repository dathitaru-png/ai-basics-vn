"""Tests for LinearRegression (lesson 03)."""

from aibasics import LinearRegression, make_regression_data, mean_squared_error


def test_regression_loss_decreases():
    X, y = make_regression_data(n=60, seed=42)
    model = LinearRegression(learning_rate=0.1, epochs=800).fit(X, y)
    assert model.loss_history[-1] < model.loss_history[0]
    assert model.loss_history[-1] < 20.0  # ends near the noise level (sigma=1.5)


def test_regression_recovers_true_line():
    X, y = make_regression_data(n=80, seed=3, noise=0.5)
    model = LinearRegression(learning_rate=0.1, epochs=1000).fit(X, y)
    assert abs(model.weights[0] - 3.0) < 0.3
    assert abs(model.bias - 2.0) < 0.5


def test_regression_predicts_near_true_values():
    X, y = make_regression_data(n=60, seed=42, noise=0.5)
    model = LinearRegression(learning_rate=0.1, epochs=1000).fit(X, y)
    for x, expected in ((0.0, 2.0), (5.0, 17.0), (10.0, 32.0)):
        assert abs(model.predict_one([x]) - expected) < 1.0
    assert mean_squared_error(y, model.predict(X)) < 1.0
