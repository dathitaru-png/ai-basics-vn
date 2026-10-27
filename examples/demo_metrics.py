"""Demo bài 12 — Đánh giá classifier bằng ma trận nhầm lẫn.

Run:  python examples/demo_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aibasics import accuracy_score
from aibasics.metrics import confusion_matrix, f1_score, precision_score, recall_score


def main() -> None:
    # 100 email hư cấu: 10 rác thật, 90 thường; máy gắn cờ 8 email là rác
    y_true = [1] * 10 + [0] * 90
    y_pred = [1] * 6 + [0] * 4 + [1] * 2 + [0] * 88
    print(f"Ma trận nhầm lẫn: {confusion_matrix(y_true, y_pred)}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.0%} — nghe rất tốt...")
    print(f"Precision: {precision_score(y_true, y_pred):.0%}")
    print(f"Recall:    {recall_score(y_true, y_pred):.0%} — ...nhưng bỏ lọt 40% số rác")
    print(f"F1:        {f1_score(y_true, y_pred):.2f}")


if __name__ == "__main__":
    main()
