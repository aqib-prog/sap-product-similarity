"""Compare the exact mode and the fast mode of the final search with my ratings only.

Run from the project folder:  python experiments/rating_comparison.py
My ratings are the truth. For each of the 30 development products it takes the top 10 of the
exact mode and of the fast mode (100 nearest products from each index) and counts how many
results I rated 2, 1 and 0, and scores each product with nDCG@10 and precision@10. The 20
locked products are hidden. Results: results/final_search_ratings.json.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from hybrid_setup import build_evaluator

from product_similarity import evaluation as ev
from product_similarity.hybrid import HybridEngine, load_final_vectors

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NEAREST = 100  # from each of the two indexes


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    setup = build_evaluator()
    base, grades, queries = setup.engine, setup.grades, setup.queries
    vectors = load_final_vectors(
        base.products,
        ROOT / "data" / "embeddings",
        ROOT / "data" / "images" / "shared" / "excluded.json",
    )
    engines = {
        "exact": HybridEngine(
            base.products, base.id_map, vectors, "exact", hidden_ids=setup.locked
        ),
        "fast": HybridEngine(
            base.products, base.id_map, vectors, "fast", NEAREST, hidden_ids=setup.locked
        ),
    }
    out: dict = {"products": len(queries), "nearest_from_each_index": NEAREST, "modes": {}}
    per_product: dict[str, dict] = {}
    for name, engine in engines.items():
        top = {no: engine.find_similar(setup.dev[no], 10) for no in queries}
        scored = ev.score_method(top, grades)
        summary = ev.summarise(scored)
        counts = Counter(
            grades.get(no, {}).get(uid, "unrated") for no, ids in top.items() for uid in ids
        )
        out["modes"][name] = {
            "results": 10 * len(queries),
            "rated_2": counts.get(2, 0),
            "rated_1": counts.get(1, 0),
            "rated_0": counts.get(0, 0),
            "unrated": counts.get("unrated", 0),
            "ndcg_at_10": round(summary["ndcg_at_10"], 4),
            "precision_at_10": round(summary["precision_at_10"], 4),
        }
        for no in queries:
            per_product.setdefault(no, {})[name] = {
                "ndcg": scored[no]["ndcg"],
                "precision": scored[no]["precision"],
            }
    same = lower = higher = 0
    differing = {}
    for no, v in per_product.items():
        e, f = v["exact"]["ndcg"], v["fast"]["ndcg"]
        if e is None or f is None:
            continue
        if abs(e - f) < 1e-9:
            same += 1
        elif f < e:
            lower += 1
            differing[no] = {"exact": round(e, 4), "fast": round(f, 4)}
        else:
            higher += 1
            differing[no] = {"exact": round(e, 4), "fast": round(f, 4)}
    out["per_product_ndcg"] = {
        "products_with_something_to_find": same + lower + higher,
        "fast_equal_to_exact": same,
        "fast_lower": lower,
        "fast_higher": higher,
        "differing": differing,
    }
    (RESULTS / "final_search_ratings.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
