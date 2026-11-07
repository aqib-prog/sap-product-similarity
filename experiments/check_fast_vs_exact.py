"""Compare the fast mode with the exact mode of the real search on the 30 development products.

Run from the project folder:  python experiments/check_fast_vs_exact.py
Both modes are the same hybrid (all ten parts, the same weights and rescaling). The only
difference is how the two vector parts are searched: the exact mode compares with every
product, the fast mode uses an HNSW index for the sentence vectors and one for the picture
vectors, and scores only their nearest products. The exact mode is the reference for counting
what the fast mode loses. For 50, 100, 250 and 500 products from each index it counts how many
of the exact top 10 the fast mode loses, scores both modes with my ratings (nDCG@10 and
precision@10), and times the whole search of each mode: the shortlist, the scoring and the
choice of the final 10. Every product is searched 5 times, the fastest run is kept and the
median is reported; one thread. The 20 locked products are hidden from every search.
Results: results/fast_vs_exact_check.json.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from hybrid_setup import build_evaluator

from product_similarity import evaluation as ev
from product_similarity.hybrid import HybridEngine, load_final_vectors

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
POOLS = (50, 100, 250, 500)  # nearest products taken from each of the two indexes
HEADLINE = 100
REPEATS = 5
PART_NAMES = ("category", "brand", "audience", "price", "colour", "weight", "rating")


def median_ms(fn, products: list[str]) -> float:
    times = []
    for pid in products:
        best = np.inf
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fn(pid)
            best = min(best, time.perf_counter() - t0)
        times.append(best * 1000)
    return float(np.median(times))


def compare(label, setup, vectors, hidden) -> dict:
    base, grades, queries = setup.engine, setup.grades, setup.queries
    exact = HybridEngine(base.products, base.id_map, vectors, "exact", hidden_ids=hidden)
    t0 = time.perf_counter()
    fast = HybridEngine(base.products, base.id_map, vectors, "fast", hidden_ids=hidden)
    build_seconds = time.perf_counter() - t0
    index_mb = fast.index_mb()
    ids = [setup.dev[no] for no in queries]
    row = {no: exact.resolve(setup.dev[no]) for no in queries}
    id_row = {uid: i for i, uid in enumerate(exact.ids)}

    exact_top = {no: exact.find_similar(setup.dev[no], 10) for no in queries}
    exact_scores = ev.summarise(ev.score_method(exact_top, grades))
    exact_ms = median_ms(lambda pid: exact.find_similar(pid, 10), ids)

    rows = []
    for pool in POOLS:
        fast.candidates = pool
        fast_top = {no: fast.find_similar(setup.dev[no], 10) for no in queries}
        lost, not_candidates = {}, {}
        for no in queries:
            wanted = {id_row[u] for u in exact_top[no]}
            cands = set(fast.candidate_rows(row[no], pool).tolist())
            not_candidates[no] = len(wanted - cands)
            lost[no] = 10 - len(set(exact_top[no]) & set(fast_top[no]))
        spread = Counter(lost.values())
        scores = ev.summarise(ev.score_method(fast_top, grades))
        out = {
            "candidates": pool,
            "slots": 10 * len(queries),
            "lost_from_final_top10_total": sum(lost.values()),
            "exact_top10_not_in_candidates_total": sum(not_candidates.values()),
            "products_losing_none": spread.get(0, 0),
            "products_losing_1": spread.get(1, 0),
            "products_losing_2": spread.get(2, 0),
            "products_losing_3_or_more": sum(v for k, v in spread.items() if k >= 3),
            "most_lost_by_one_product": max(lost.values()),
            "lost_by_product": {no: n for no, n in lost.items() if n},
            "recall_at_10_vs_exact": round(1 - sum(lost.values()) / (10 * len(queries)), 4),
            "ndcg_at_10": round(scores["ndcg_at_10"], 4),
            "precision_at_10": round(scores["precision_at_10"], 4),
            "unrated_in_top_10": int(scores["unrated_in_top_10"]),
            "ms_per_query": round(median_ms(lambda pid: fast.find_similar(pid, 10), ids), 3),
        }
        rows.append(out)
        print(label, {k: v for k, v in out.items() if k != "lost_by_product"}, flush=True)

    # Why products are lost: the part scores of lost products against the ones that were kept.
    fast.candidates = HEADLINE
    lost_parts, kept_parts = [], []
    for no in queries:
        fast_set = set(fast.find_similar(setup.dev[no], 10))
        sims = exact.attribute_similarities(row[no])
        for uid in exact_top[no]:
            j = id_row[uid]
            attributes = [sims[p][j] for p in PART_NAMES]
            record = {
                "title_tfidf": float(sims["title"][j]),
                "sentence": float(sims["text_vectors"][j]),
                "picture": float(sims["image"][j]),
                "attributes": None if np.isnan(attributes).all() else float(np.nanmean(attributes)),
            }
            (kept_parts if uid in fast_set else lost_parts).append(record)

    def mean_of(records, key):
        values = [r[key] for r in records if r[key] is not None and not np.isnan(r[key])]
        return round(float(np.mean(values)), 3) if values else None

    keys = ("title_tfidf", "sentence", "picture", "attributes")
    return {
        "label": label,
        "products_that_can_be_results": int((~exact._hidden).sum()),
        "exact": {
            "ndcg_at_10": round(exact_scores["ndcg_at_10"], 4),
            "precision_at_10": round(exact_scores["precision_at_10"], 4),
            "unrated_in_top_10": int(exact_scores["unrated_in_top_10"]),
            "ms_per_query": round(exact_ms, 3),
            "memory_mb": exact.memory_mb(),  # the vectors the search holds
        },
        "fast": {
            "index": "HNSW for the sentence vectors and for the picture vectors, M=16, efConstruction=200, efSearch=64",
            "build_seconds": round(build_seconds, 2),
            "index_mb": index_mb,
            "memory_mb": fast.memory_mb(),  # the vectors plus the index
        },
        "by_candidates": rows,
        "average_part_scores_of_the_lost": {k: mean_of(lost_parts, k) for k in keys},
        "average_part_scores_of_the_found": {k: mean_of(kept_parts, k) for k in keys},
        "lost_places": len(lost_parts),
    }


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    setup = build_evaluator()
    vectors = load_final_vectors(
        setup.engine.products,
        ROOT / "data" / "embeddings",
        ROOT / "data" / "images" / "shared" / "excluded.json",
    )
    report = {
        "products": len(setup.queries),
        "reference": "exact mode of the real search",
        "timing_rule": "30 development products, 5 runs each, fastest kept, median; one thread",
        "comparison": compare("final_search", setup, vectors, setup.locked),
    }
    (RESULTS / "fast_vs_exact_check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("saved results/fast_vs_exact_check.json")


if __name__ == "__main__":
    main()
