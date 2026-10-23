"""Tests for DecisionTreeClassifier (lesson 05)."""


from aibasics import DecisionTreeClassifier, accuracy_score


def test_tree_learns_threshold_rule():
    X = [[float(s)] for s in range(0, 11)]
    y = [1 if s >= 5 else 0 for s in range(0, 11)]
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert accuracy_score(y, tree.predict(X)) == 1.0


def test_tree_learns_two_feature_rule():
    # Label is 1 only when BOTH features are large (an AND-like rule).
    X = [
        [3.0, 3.0], [4.0, 2.5], [3.5, 4.0], [5.0, 5.0],
        [0.0, 3.0], [1.0, 4.0], [3.0, 0.5], [4.0, 1.0], [0.5, 0.5],
    ]
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0]
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert accuracy_score(y, tree.predict(X)) == 1.0
    assert tree.predict_one([4.5, 3.5]) == 1
    assert tree.predict_one([4.5, 0.0]) == 0
