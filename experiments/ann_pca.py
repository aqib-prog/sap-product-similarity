"""Shorten the vectors with PCA, then run every search method on the shorter vectors.

Run from the project folder:  python experiments/ann_pca.py
Same rule as ann_benchmark.py: the same 1,000 random products (seed 2020, no locked or
development product), one thread, one query at a time, fastest of 5 runs, median reported.
PCA is trained on all the vectors and keeps 256, 128 or 64 numbers. Exact search, FAISS HNSW,
FAISS IVF and Annoy then run on the shorter vectors. Recall@10 is measured against the exact
top 10 on the full vectors, so it shows what PCA and the index lose together. The extra time
to project a query vector with PCA is measured too. Results: results/ann_pca.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ann_benchmark import K, make_recall, run_space
from ann_data import load_spaces

from product_similarity.image_features import normalise

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (256, 128, 64)


def projection_ms(pca, matrix: np.ndarray, positions: np.ndarray) -> float:
    """Median time to turn one full query vector into its short version."""
    times = []
    for q in positions:
        best = np.inf
        for _ in range(5):
            t0 = time.perf_counter()
            normalise(pca.apply(matrix[int(q) : int(q) + 1]))
            best = min(best, time.perf_counter() - t0)
        times.append(best * 1000)
    return float(np.median(times))


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    spaces, query_rows, _ = load_spaces()
    saved = np.load(ROOT / "data" / "embeddings" / "ann_exact_top10.npz")
    report = {"k": K, "threads": 1, "targets": list(TARGETS), "spaces": {}}
    for name, space in spaces.items():
        positions = np.array([space.position[int(r)] for r in query_rows["random1000"]])
        recall = make_recall(space.matrix, positions, saved[f"{name}_random1000"])
        report["spaces"][name] = {
            "products": int(space.matrix.shape[0]),
            "dimensions": int(space.matrix.shape[1]),
            "by_dimensions": {},
        }
        for target in TARGETS:
            print(f"== {name}: PCA to {target} numbers", flush=True)
            pca = faiss.PCAMatrix(space.matrix.shape[1], target)
            pca.train(space.matrix)
            small = normalise(pca.apply(space.matrix))
            rows = run_space(SimpleNamespace(matrix=small), positions, recall)
            for row in rows:
                row["dimensions"] = target
            report["spaces"][name]["by_dimensions"][str(target)] = {
                "projection_ms": round(projection_ms(pca, space.matrix, positions), 4),
                "vectors_mb": round(small.nbytes / 1e6, 1),
                "indexes": rows,
            }
    out = ROOT / "results" / "ann_pca.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
