"""Time the exact search on the picture and sentence vectors, and save its answers.

Run from the project folder:  python experiments/ann_exact_baseline.py
The one timing rule (the same as in ann_benchmark.py): 1,000 random products, each searched
5 times, fastest run kept, median reported.
For each of the two vector spaces, every query product is compared with all the other
products by cosine similarity. The time per query and the memory are measured, and the
exact top 10 are saved as the right answers for the fast methods to be compared with.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
from ann_data import SEED, load_spaces

from product_similarity.vector_search import exact_top_k

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
K = 10
REPEATS = 5


def main() -> None:
    spaces, query_rows, products = load_spaces()
    report: dict = {
        "k": K,
        "machine": platform.platform(),
        "numpy": np.__version__,
        "spaces": {},
    }
    answers: dict[str, np.ndarray] = {}
    for name, space in spaces.items():
        position = space.position
        n, dim = space.matrix.shape
        info = {
            "products": n,
            "dimensions": dim,
            "memory_mb": round(space.matrix.nbytes / 1e6, 1),
            "queries": {},
        }
        for set_name, rows in query_rows.items():
            positions = np.array([position[int(r)] for r in rows])
            top = exact_top_k(space.matrix, positions, K)
            answers[f"{name}_{set_name}"] = top
            single = []
            for q in positions:
                best = np.inf
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    exact_top_k(space.matrix, np.array([q]), K)
                    best = min(best, time.perf_counter() - t0)
                single.append(best * 1000)
            t0 = time.perf_counter()
            exact_top_k(space.matrix, positions, K)
            batch = (time.perf_counter() - t0) * 1000 / len(positions)
            info["queries"][set_name] = {
                "queries": len(positions),
                "one_query_median_ms": round(float(np.median(single)), 2),
                "one_query_p95_ms": round(float(np.percentile(single, 95)), 2),
                "batch_ms_per_query": round(batch, 2),
            }
            print(name, set_name, info["queries"][set_name], flush=True)
        report["spaces"][name] = info
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ann_exact_baseline.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "data" / "embeddings" / "ann_exact_top10.npz",
        **answers,
        **{f"{n}_rows": s.rows for n, s in spaces.items()},
        **{f"queries_{k}": v for k, v in query_rows.items()},
    )
    print(json.dumps({k: v for k, v in report.items() if k != "spaces"}))
    print("seed", SEED, "products", len(products))


if __name__ == "__main__":
    main()
