"""Classifier evaluation beyond accuracy — from scratch.

Companion code for lesson 12 (``lessons/12-danh-gia-classifier.md``).

Everything here is computed from one small table, the confusion matrix:
for a chosen "positive" label, count how many predictions were

- ``tp`` — true positives:  predicted positive, actually positive
- ``fp`` — false positives: predicted positive, actually negative
- ``tn`` — true negatives: predicted negative, actually negative
- ``fn`` — false negatives: predicted negative, actually positive

Precision, recall and F1 are just different ratios of those four
counts. When a ratio has a zero denominator (for example precision
when the model never predicts positive), the function returns 0.0
instead of crashing — a model that never says "positive" has found
no positives, so its precision is zero, not undefined.

``accuracy_score`` is re-exported from :mod:`aibasics.data` so all
evaluation helpers can be imported from this one module.
"""

from __future__ import annotations

from .data import accuracy_score

__all__ = [
    "accuracy_score",
    "confusion_matrix",
    "f1_score",
    "precision_score",
    "recall_score",
]


def confusion_matrix(y_true, y_pred, positive_label=1) -> dict:
    """Count tp / fp / tn / fn for ``positive_label`` (default 1)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = fp = tn = fn = 0
    for actual, predicted in zip(y_true, y_pred):
        actual_pos = actual == positive_label
        predicted_pos = predicted == positive_label
        if predicted_pos and actual_pos:
            tp += 1
        elif predicted_pos and not actual_pos:
            fp += 1
        elif not predicted_pos and not actual_pos:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_score(y_true, y_pred, positive_label=1) -> float:
    """Of the items predicted positive, how many really are positive?"""
    cm = confusion_matrix(y_true, y_pred, positive_label)
    if cm["tp"] + cm["fp"] == 0:
        return 0.0
    return cm["tp"] / (cm["tp"] + cm["fp"])


def recall_score(y_true, y_pred, positive_label=1) -> float:
    """Of the truly positive items, how many did the model catch?"""
    cm = confusion_matrix(y_true, y_pred, positive_label)
    if cm["tp"] + cm["fn"] == 0:
        return 0.0
    return cm["tp"] / (cm["tp"] + cm["fn"])


def f1_score(y_true, y_pred, positive_label=1) -> float:
    """Harmonic mean of precision and recall — high only if both are."""
    precision = precision_score(y_true, y_pred, positive_label)
    recall = recall_score(y_true, y_pred, positive_label)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
