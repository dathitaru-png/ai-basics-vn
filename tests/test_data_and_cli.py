"""Tests for data helpers and the CLI."""

from aibasics import cli, make_blobs, train_test_split


def test_train_test_split_sizes_and_coverage():
    X, y = make_blobs(n_per_cluster=20, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.25, seed=42)
    assert len(X_test) == len(y_test) == 10
    assert len(X_train) == len(y_train) == 30
    assert len(X_train) + len(X_test) == len(X)


def test_cli_demos_run(capsys):
    for name in ("regression", "knn", "tree", "bayes", "kmeans", "perceptron", "mlp"):
        assert cli.main(["demo", name]) == 0
    out = capsys.readouterr().out
    assert "Hồi quy tuyến tính" in out
    assert "Naive Bayes" in out
