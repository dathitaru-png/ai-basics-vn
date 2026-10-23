"""Demo bài 08 — Perceptron học AND, MLP nhỏ học XOR.

Run:  python examples/demo_perceptron.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aibasics import Perceptron, TinyMLP

GATES = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]


def main() -> None:
    and_y = [0, 0, 0, 1]
    perceptron = Perceptron(epochs=50).fit(GATES, and_y)
    print(f"Perceptron học AND: {perceptron.predict(GATES)} (đúng: {and_y})")

    xor_y = [0, 1, 1, 0]
    perceptron_xor = Perceptron(epochs=50).fit(GATES, xor_y)
    print(f"Perceptron học XOR: {perceptron_xor.predict(GATES)} (đúng: {xor_y}) <- không học nổi!")

    mlp = TinyMLP(hidden_dim=4, epochs=4000, seed=42).fit(GATES, xor_y)
    print(f"MLP nhỏ học XOR:    {mlp.predict(GATES)} (đúng: {xor_y}) <- thêm 1 lớp ẩn là xong")


if __name__ == "__main__":
    main()
