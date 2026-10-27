"""Tests for the metrics module (lesson 12)."""

import pytest

from aibasics.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def test_confusion_matrix_matches_hand_computed_values():
    # tp: positions 0, 4; fp: position 3; fn: position 1; tn: 2, 5.
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 1, 0]
    assert confusion_matrix(y_true, y_pred) == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}


def test_precision_recall_f1_match_hand_computed_values():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 1, 0]
    assert precision_score(y_true, y_pred) == 2 / 3
    assert recall_score(y_true, y_pred) == 2 / 3
    assert f1_score(y_true, y_pred) == 2 / 3
    assert accuracy_score(y_true, y_pred) == 4 / 6


def test_all_correct_scores_are_one():
    y_true = [1, 0, 1, 0]
    assert confusion_matrix(y_true, y_true) == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}
    assert precision_score(y_true, y_true) == 1.0
    assert recall_score(y_true, y_true) == 1.0
    assert f1_score(y_true, y_true) == 1.0
    assert accuracy_score(y_true, y_true) == 1.0


def test_zero_denominators_return_zero_not_crash():
    # Model never predicts positive: precision denominator is 0.
    y_true = [1, 0, 1]
    y_pred = [0, 0, 0]
    assert precision_score(y_true, y_pred) == 0.0
    assert recall_score(y_true, y_pred) == 0.0
    assert f1_score(y_true, y_pred) == 0.0


def test_string_labels_with_explicit_positive_label():
    # Same logic as the spam filter in lesson 06.
    y_true = ["spam", "ham", "spam", "ham"]
    y_pred = ["spam", "spam", "ham", "ham"]
    cm = confusion_matrix(y_true, y_pred, positive_label="spam")
    assert cm == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert precision_score(y_true, y_pred, positive_label="spam") == 0.5
    assert recall_score(y_true, y_pred, positive_label="spam") == 0.5


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        confusion_matrix([1, 0], [1])
