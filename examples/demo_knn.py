"""Demo bài 04 — k-NN phân loại.

Run:  python examples/demo_knn.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aibasics import KNeighborsClassifier, accuracy_score, make_blobs, train_test_split


def main() -> None:
    X, y = make_blobs(n_per_cluster=25, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=1)
    for k in (1, 3, 5):
        model = KNeighborsClassifier(k=k).fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"k = {k}: độ chính xác trên tập kiểm tra = {acc:.0%}")


if __name__ == "__main__":
    main()
