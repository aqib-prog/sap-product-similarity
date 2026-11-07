"""Score the final search on the 20 locked final products, with my ratings. Run this once.

Run from the project folder:  python experiments/score_final_test.py
Needs evaluation/judgments/final_grades.csv (made with run_final_test_rating.py). My ratings are
the truth: for each mode (exact, and fast with the 100 nearest products from each index) it takes
the top 10 of each of the 20 locked products and scores them with nDCG@10 and precision@10, and
counts the results I rated 2, 1 and 0. It also times the whole search of each mode: 5 runs per
product, the fastest kept, the median of the 20 products, one thread. Nothing is tuned here.
Results: results/final_test_scores.json.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from product_similarity import evaluation as ev
from product_similarity.hybrid import HybridEngine
from product_similarity.service import baseline_engine
from product_similarity.vector_files import load_vector_artifact

ROOT = Path(__file__).resolve().parents[1]
NEAREST = 100
REPEATS = 5


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    base = baseline_engine()
    vectors = load_vector_artifact(base.products, ROOT / "artifacts" / "vectors")
    assert vectors is not None, "the vector files are missing"
    queries = json.loads((ROOT / "evaluation" / "queries.json").read_text(encoding="utf-8"))
    locked = {q["query_no"]: q["id"] for q in queries if q["split"] == "final"}
    assert len(locked) == 20
    grades = ev.load_grades(ROOT / "evaluation" / "judgments" / "final_grades.csv")
    engines = {
        "exact": HybridEngine(base.products, base.id_map, vectors, "exact"),
        "fast": HybridEngine(base.products, base.id_map, vectors, "fast", NEAREST),
    }
    report: dict = {"products": len(locked), "ratings": "my ratings, 204 results", "modes": {}}
    per_product: dict[str, dict] = {}
    for name, engine in engines.items():
        top = {no: engine.find_similar(pid, 10) for no, pid in locked.items()}
        scored = ev.score_method(top, grades)
        summary = ev.summarise(scored)
        counts = Counter(
            grades.get(no, {}).get(uid, "unrated") for no, ids in top.items() for uid in ids
        )
        times = []
        for pid in locked.values():
            best = np.inf
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                engine.find_similar(pid, 10)
                best = min(best, time.perf_counter() - t0)
            times.append(best * 1000)
        report["modes"][name] = {
            "ndcg_at_10": round(summary["ndcg_at_10"], 4),
            "precision_at_10": round(summary["precision_at_10"], 4),
            "results": 10 * len(locked),
            "rated_2": counts.get(2, 0),
            "rated_1": counts.get(1, 0),
            "rated_0": counts.get(0, 0),
            "unrated": counts.get("unrated", 0),
            "products_with_something_to_find": int(summary["products_scored"]),
            "ms_per_search": round(float(np.median(times)), 3),
        }
        for no in locked:
            per_product.setdefault(no, {})[name] = {
                "ndcg": scored[no]["ndcg"],
                "precision": scored[no]["precision"],
            }
        print(name, report["modes"][name], flush=True)
    same = lower = higher = 0
    differing = {}
    for no, v in per_product.items():
        e, f = v["exact"]["ndcg"], v["fast"]["ndcg"]
        if e is None or f is None:
            continue
        if abs(e - f) < 1e-9:
            same += 1
        else:
            lower += f < e
            higher += f > e
            differing[no] = {"exact": round(e, 4), "fast": round(f, 4)}
    report["per_product_ndcg"] = {
        "same_in_both_modes": same,
        "fast_lower": int(lower),
        "fast_higher": int(higher),
        "differing": differing,
    }
    report["per_product"] = {
        no: {
            mode: {k: (None if x is None else round(x, 4)) for k, x in vals.items()}
            for mode, vals in v.items()
        }
        for no, v in per_product.items()
    }
    (ROOT / "results" / "final_test_scores.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("saved results/final_test_scores.json")


if __name__ == "__main__":
    main()
