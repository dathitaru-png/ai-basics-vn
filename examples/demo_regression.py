"""Demo bài 03 — Hồi quy tuyến tính (gradient descent).

Run:  python examples/demo_regression.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aibasics import LinearRegression, make_regression_data, mean_squared_error


def main() -> None:
    # Dữ liệu hư cấu: y = 3x + 2 + nhiễu nhỏ
    X, y = make_regression_data(n=60, seed=42)
    model = LinearRegression(learning_rate=0.1, epochs=800).fit(X, y)
    print(f"w học được: {model.weights[0]:.2f} (giá trị thật: 3)")
    print(f"b học được: {model.bias:.2f} (giá trị thật: 2)")
    print(f"MSE: {mean_squared_error(y, model.predict(X)):.2f}")
    print(f"Loss vòng đầu: {model.loss_history[0]:.1f} -> vòng cuối: {model.loss_history[-1]:.2f}")


if __name__ == "__main__":
    main()
