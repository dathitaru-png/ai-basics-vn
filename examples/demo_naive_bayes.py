"""Demo bài 06 — Naive Bayes phân loại tin nhắn rác.

Run:  python examples/demo_naive_bayes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aibasics import MultinomialNaiveBayes, accuracy_score, make_text_dataset


def main() -> None:
    texts, labels = make_text_dataset()  # toàn bộ tin nhắn đều hư cấu
    model = MultinomialNaiveBayes().fit(texts, labels)
    print(f"Độ chính xác trên tập học: {accuracy_score(labels, model.predict(texts)):.0%}")
    for text in [
        "Nhận quà miễn phí khi trúng thưởng lớn hôm nay!",
        "Ngày mai lớp mình họp phụ huynh lúc 8 giờ sáng.",
    ]:
        proba = model.predict_proba(text)
        print(f"'{text}'")
        print(f"  -> {model.predict_one(text)} (spam {proba['spam']:.0%}, ham {proba['ham']:.0%})")


if __name__ == "__main__":
    main()
