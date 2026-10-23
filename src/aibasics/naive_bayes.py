"""Multinomial Naive Bayes for text — from scratch.

Companion code for lesson 06 (``lessons/06-naive-bayes-van-ban.md``).

Naive Bayes scores a message for each class with Bayes' rule while
(naively) pretending the words are independent:

    log P(class) + sum over words of log P(word | class)

Word probabilities are smoothed (Laplace, ``alpha``) so a word that
never appeared in training does not zero out a whole class.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens (keeps Vietnamese letters)."""
    return _TOKEN_RE.findall(text.lower())


class MultinomialNaiveBayes:
    """Naive Bayes over word counts, for short text classification."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: list = []
        self._log_prior: dict = {}
        self._word_counts: dict = {}
        self._total_words: dict = {}
        self._vocab: set = set()

    def fit(self, texts: list[str], labels: list) -> "MultinomialNaiveBayes":
        if not texts or len(texts) != len(labels):
            raise ValueError("texts and labels must be non-empty and equally long")
        class_counts = Counter(labels)
        self.classes_ = sorted(class_counts, key=str)
        n = len(texts)
        self._log_prior = {c: math.log(class_counts[c] / n) for c in self.classes_}
        self._word_counts = {c: Counter() for c in self.classes_}
        self._total_words = {c: 0 for c in self.classes_}
        self._vocab = set()
        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            self._word_counts[label].update(tokens)
            self._total_words[label] += len(tokens)
            self._vocab.update(tokens)
        return self

    def _log_prob_word(self, word: str, cls) -> float:
        count = self._word_counts[cls].get(word, 0)
        denom = self._total_words[cls] + self.alpha * len(self._vocab)
        return math.log((count + self.alpha) / denom)

    def predict_log_proba(self, text: str) -> dict:
        """Unnormalised log score per class (higher = more likely)."""
        if not self.classes_:
            raise RuntimeError("call fit() before predict()")
        tokens = tokenize(text)
        return {
            cls: self._log_prior[cls] + sum(self._log_prob_word(w, cls) for w in tokens)
            for cls in self.classes_
        }

    def predict_proba(self, text: str) -> dict:
        """Normalised probability per class (values sum to 1)."""
        scores = self.predict_log_proba(text)
        peak = max(scores.values())
        exps = {c: math.exp(s - peak) for c, s in scores.items()}
        total = sum(exps.values())
        return {c: v / total for c, v in exps.items()}

    def predict_one(self, text: str):
        scores = self.predict_log_proba(text)
        return max(scores, key=scores.get)

    def predict(self, texts: list[str]) -> list:
        return [self.predict_one(t) for t in texts]
