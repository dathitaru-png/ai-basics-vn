"""Synthetic data helpers and small evaluation utilities.

Every dataset produced here is **fictional and randomly generated** so
beginners can learn and experiment without touching anyone's real data.
Never present this data as real-world measurements.

Data layout used across the whole package:

- ``X``: a list of rows, each row a list of floats (the features).
- ``y``: a list of labels (floats for regression, ints/strings for
  classification), one per row.
"""

from __future__ import annotations

import random


def train_test_split(X, y, test_ratio: float = 0.25, seed: int = 42):
    """Shuffle and split ``X``/``y`` into (X_train, X_test, y_train, y_test)."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    indices = list(range(len(X)))
    random.Random(seed).shuffle(indices)
    n_test = max(1, int(round(len(X) * test_ratio)))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return (
        [X[i] for i in train_idx],
        [X[i] for i in test_idx],
        [y[i] for i in train_idx],
        [y[i] for i in test_idx],
    )


def accuracy_score(y_true, y_pred) -> float:
    """Fraction of predictions that match the true labels."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("cannot score an empty dataset")
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def mean_squared_error(y_true, y_pred) -> float:
    """Average squared difference between true and predicted values."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("cannot score an empty dataset")
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def make_regression_data(n: int = 60, seed: int = 42, noise: float = 1.5):
    """Fictional data following ``y = 3*x + 2 + noise`` (one feature)."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x = rng.uniform(0, 10)
        X.append([x])
        y.append(3.0 * x + 2.0 + rng.gauss(0, noise))
    return X, y


def make_blobs(n_per_cluster: int = 25, centers=((0.0, 0.0), (6.0, 6.0)),
               seed: int = 42, spread: float = 0.6):
    """Fictional 2-D point clusters, one integer label per center."""
    rng = random.Random(seed)
    X, y = [], []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            X.append([rng.gauss(cx, spread), rng.gauss(cy, spread)])
            y.append(label)
    return X, y


#: Toy text messages for the Naive Bayes lesson. Every message is invented
#: for teaching; none of it is copied from any real inbox.
TOY_MESSAGES = [
    # --- "spam" (tin rác) ---
    ("Trúng thưởng lớn! Nhận ngay quà miễn phí, bấm vào link này!", "spam"),
    ("Khuyến mãi sốc giảm giá 90% chỉ hôm nay, mua ngay kẻo hết!", "spam"),
    ("Bạn đã thắng giải thưởng tiền mặt, hãy gửi phí để nhận quà!", "spam"),
    ("Vay tiền nhanh không cần thế chấp, lãi suất thấp, duyệt ngay!", "spam"),
    ("Cơ hội đầu tư sinh lời gấp đôi trong một tuần, đừng bỏ lỡ!", "spam"),
    ("Miễn phí 100%! Nhận quà tặng giá trị lớn khi đăng ký hôm nay!", "spam"),
    ("Chúc mừng bạn trúng xe máy, liên hệ ngay để nhận thưởng!", "spam"),
    ("Giảm giá cực sốc, hàng hiệu giá rẻ bất ngờ, số lượng có hạn!", "spam"),
    # --- "ham" (tin bình thường) ---
    ("Chiều nay họp nhóm lúc 3 giờ ở phòng cũ nhé bạn.", "ham"),
    ("Mẹ ơi, con về nhà ăn cơm lúc 7 giờ tối nay.", "ham"),
    ("Bài tập tuần này nộp vào thứ sáu, nhớ làm đầy đủ nhé.", "ham"),
    ("Cuối tuần này cả nhà mình đi siêu thị mua đồ ăn không?", "ham"),
    ("Thầy gửi tài liệu ôn tập qua email, bạn kiểm tra giúp mình.", "ham"),
    ("Hẹn gặp bạn ở thư viện sáng mai để ôn bài cùng nhau.", "ham"),
    ("Hoá đơn tiền điện tháng này đã có, nhớ đóng trước ngày 20.", "ham"),
    ("Cảm ơn bạn đã gửi file, mình nhận được rồi.", "ham"),
]


def make_text_dataset():
    """Return the toy (fictional) spam/ham messages as (texts, labels)."""
    texts = [text for text, _ in TOY_MESSAGES]
    labels = [label for _, label in TOY_MESSAGES]
    return texts, labels
