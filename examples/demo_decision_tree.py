"""Demo bài 05 — Cây quyết định.

Run:  python examples/demo_decision_tree.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aibasics import DecisionTreeClassifier, accuracy_score


def main() -> None:
    # Ví dụ hư cấu: dựa vào giờ học mỗi ngày và số buổi nghỉ, đoán qua môn hay không.
    # X = [giờ học/ngày, số buổi nghỉ], y = 1 (qua môn) / 0 (trượt)
    X = [
        [3.0, 0.0], [2.5, 1.0], [4.0, 0.0], [3.5, 2.0], [2.0, 0.0],
        [0.5, 4.0], [1.0, 5.0], [0.0, 3.0], [1.5, 6.0], [0.5, 2.0],
    ]
    y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    print(f"Độ chính xác trên tập học: {accuracy_score(y, model.predict(X)):.0%}")
    print(f"Học sinh học 3h/ngày, nghỉ 1 buổi -> {'qua môn' if model.predict_one([3.0, 1.0]) else 'trượt'}")
    print(f"Học sinh học 0.5h/ngày, nghỉ 5 buổi -> {'qua môn' if model.predict_one([0.5, 5.0]) else 'trượt'}")


if __name__ == "__main__":
    main()
