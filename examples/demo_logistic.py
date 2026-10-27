"""Demo bài 11 — Hồi quy logistic (gradient descent).

Run:  python examples/demo_logistic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aibasics import LogisticRegression, accuracy_score, make_blobs, train_test_split


def main() -> None:
    # Dữ liệu hư cấu: giờ ôn -> đỗ (1) / trượt (0), đỗ khi ôn >= 5 giờ
    X = [[float(h)] for h in range(1, 9)]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    model = LogisticRegression(learning_rate=0.1, epochs=1000).fit(X, y)
    print(f"w học được: {model.weights[0]:.2f}, b: {model.bias:.2f}")
    print(f"Loss vòng đầu: {model.loss_history[0]:.3f} "
          f"-> vòng cuối: {model.loss_history[-1]:.3f}")
    for hours in (4.0, 4.5, 6.0):
        print(f"Ôn {hours} giờ: P(đỗ) = {model.predict_proba_one([hours]):.2f}")

    # Hai cụm điểm hư cấu: xác suất + độ chính xác trên tập kiểm tra
    Xb, yb = make_blobs(n_per_cluster=25, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(Xb, yb, seed=1)
    blob_model = LogisticRegression(learning_rate=0.1, epochs=1000).fit(X_train, y_train)
    print(f"Độ chính xác trên 2 cụm điểm: "
          f"{accuracy_score(y_test, blob_model.predict(X_test)):.0%}")


if __name__ == "__main__":
    main()
