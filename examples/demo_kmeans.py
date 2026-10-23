"""Demo bài 07 — k-Means phân cụm.

Run:  python examples/demo_kmeans.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aibasics import KMeans, make_blobs


def main() -> None:
    centers = ((0.0, 0.0), (6.0, 6.0), (0.0, 8.0))
    X, y = make_blobs(n_per_cluster=20, centers=centers, seed=7)
    model = KMeans(k=3, seed=42).fit(X)
    print("Tâm cụm thật:   ", centers)
    print("Tâm cụm tìm được:", [tuple(round(v, 1) for v in c) for c in model.centroids_])
    print(f"Số vòng lặp: {model.n_iter_}, inertia: {model.inertia_:.1f}")


if __name__ == "__main__":
    main()
