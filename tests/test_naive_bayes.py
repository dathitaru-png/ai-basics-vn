"""Tests for MultinomialNaiveBayes (lesson 06)."""

from aibasics import MultinomialNaiveBayes, accuracy_score, make_text_dataset, tokenize


def _trained():
    texts, labels = make_text_dataset()
    return MultinomialNaiveBayes().fit(texts, labels), texts, labels


def test_naive_bayes_fits_toy_dataset():
    model, texts, labels = _trained()
    assert accuracy_score(labels, model.predict(texts)) >= 0.9


def test_naive_bayes_separates_obvious_spam_and_ham():
    model, _, _ = _trained()
    assert model.predict_one("Trúng thưởng lớn, nhận quà miễn phí ngay!") == "spam"
    assert model.predict_one("Chiều mai họp nhóm ở thư viện nhé bạn.") == "ham"


def test_naive_bayes_probabilities_sum_to_one():
    model, _, _ = _trained()
    proba = model.predict_proba("Khuyến mãi giảm giá sốc hôm nay")
    assert abs(sum(proba.values()) - 1.0) < 1e-9
    assert proba["spam"] > proba["ham"]


def test_tokenize_keeps_vietnamese_letters():
    assert tokenize("Xin chào, bạn ơi!") == ["xin", "chào", "bạn", "ơi"]
