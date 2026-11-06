"""Use a fast index to pick candidates for the hybrid, and check what is lost.

Run from the project folder:  python experiments/ann_hybrid_check.py
The index holds one combined vector per product: the sentence vector and the picture vector,
scaled by the square roots of their weights in the final 4-weight hybrid, so that a dot
product is the weighted sum of the two cosines. For each of the 30 development products the
index returns N candidates, and the full hybrid score (all ten parts) picks the final 10. The
final 10 are compared with the final 10 of the normal hybrid, which scores all products, and
with my ratings (nDCG@10). The 20 locked products are never in the index.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from hybrid_setup import build_evaluator, expand_four

from product_similarity import evaluation as ev

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EMBEDDINGS = ROOT / "data" / "embeddings"
POOLS = (50, 100, 200, 500)
EF_SEARCH = 64
REPEATS = 5


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    report = json.loads((RESULTS / "hybrid_tuning.json").read_text(encoding="utf-8"))
    weights = expand_four(report["four_weights"]["weights"])
    setup = build_evaluator()
    evaluator, engine, grades, queries = setup.evaluator, setup.engine, setup.grades, setup.queries

    sentence = np.load(EMBEDDINGS / "text_minilm.npz")["vectors"].astype(np.float32)
    picture = np.nan_to_num(engine._image, nan=0.0).astype(np.float32)  # no picture = zeros
    a, b = weights["text_vectors"], weights["image"]
    combined = np.hstack([np.sqrt(a / (a + b)) * sentence, np.sqrt(b / (a + b)) * picture]).astype(
        np.float32
    )
    allowed = np.flatnonzero(~np.isin(engine.ids, list(setup.locked)))
    matrix = np.ascontiguousarray(combined[allowed])
    row_of = {int(r): i for i, r in enumerate(allowed)}

    index = faiss.IndexHNSWFlat(matrix.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = EF_SEARCH
    t0 = time.perf_counter()
    index.add(matrix)
    build_s = time.perf_counter() - t0
    flat = faiss.IndexFlatIP(matrix.shape[1])
    flat.add(matrix)

    exact_top = {no: evaluator.top_ids(weights, no) for no in queries}
    exact_ndcg = ev.summarise(ev.score_method(exact_top, grades))["ndcg_at_10"]

    def best_of(fn):
        best = np.inf
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best * 1000

    exact_ms = float(
        np.median(
            [best_of(lambda i=evaluator.query_row[no]: engine.score_all(i)) for no in queries]
        )
    )
    rows = []
    for source, searcher in (("hnsw", index), ("exact_vectors", flat)):
        for pool in POOLS:
            tops, times = {}, []
            for no in queries:
                q = row_of[evaluator.query_row[no]]
                times.append(
                    best_of(lambda q=q, s=searcher, n=pool: s.search(matrix[q : q + 1], n + 1))
                )
                found = searcher.search(matrix[q : q + 1], pool + 1)[1][0]
                cands = [int(allowed[j]) for j in found if j >= 0 and j != q]
                tops[no] = evaluator.top_ids(weights, no, candidates=cands)
            overlap = float(np.mean([len(set(tops[n]) & set(exact_top[n])) / 10 for n in queries]))
            per = ev.score_method(tops, grades)
            row = {
                "candidates_from": source,
                "candidates": pool,
                "overlap_with_exact_hybrid_at_10": round(overlap, 4),
                "ndcg_at_10": round(ev.summarise(per)["ndcg_at_10"], 4),
                "candidate_search_ms": round(float(np.median(times)), 3),
            }
            rows.append(row)
            print(row, flush=True)
    out = {
        "weights": weights,
        "queries": len(queries),
        "exact_hybrid_ndcg_at_10": round(exact_ndcg, 4),
        "exact_hybrid_ms_per_query": round(exact_ms, 2),
        "index": {
            "type": "HNSW",
            "M": 16,
            "efConstruction": 200,
            "efSearch": EF_SEARCH,
            "build_seconds": round(build_s, 2),
            "vectors": int(matrix.shape[0]),
            "dimensions": int(matrix.shape[1]),
        },
        "results": rows,
    }
    (RESULTS / "ann_hybrid_check.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "results"}))
    assert ev.K == 10 and not setup.locked & {c for t in exact_top.values() for c in t}


if __name__ == "__main__":
    main()
