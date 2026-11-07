"""Check that the half-size vector files give the same results as the full-size vectors.

Run from the project folder:  python experiments/check_vector_files.py
For the exact and the fast mode it compares the real search on the full-size vectors with the
real search on the files in artifacts/vectors/: the top 10 of the 30 development products
(scored with my ratings) and of 1,000 random products (seed 2020, never a locked product).
The 20 locked products are hidden. Results: results/vector_files_check.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from hybrid_setup import build_evaluator

from product_similarity import evaluation as ev
from product_similarity.hybrid import HybridEngine, load_final_vectors
from product_similarity.vector_files import load_vector_artifact

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    setup = build_evaluator()
    base = setup.engine
    full = load_final_vectors(
        base.products,
        ROOT / "data" / "embeddings",
        ROOT / "data" / "images" / "shared" / "excluded.json",
    )
    half = load_vector_artifact(base.products, ROOT / "artifacts" / "vectors")
    assert half is not None, "run experiments/export_vectors.py first"
    rng = np.random.default_rng(2020)
    allowed = np.flatnonzero(~np.isin(base.ids, list(setup.locked)))
    random_rows = rng.choice(allowed, 1000, replace=False)
    report = {
        "number_format_in_files": "float16",
        "max_rounding_error": {
            "picture": float(np.nanmax(np.abs(half.picture - full.picture))),
            "sentence": float(np.nanmax(np.abs(half.sentence - full.sentence))),
        },
        "modes": {},
    }
    for mode in ("exact", "fast"):
        a = HybridEngine(base.products, base.id_map, full, mode, hidden_ids=setup.locked)
        b = HybridEngine(base.products, base.id_map, half, mode, hidden_ids=setup.locked)
        top_a = {no: a.find_similar(setup.dev[no], 10) for no in setup.queries}
        top_b = {no: b.find_similar(setup.dev[no], 10) for no in setup.queries}
        score_a = ev.summarise(ev.score_method(top_a, setup.grades))
        score_b = ev.summarise(ev.score_method(top_b, setup.grades))
        identical, overlap = 0, []
        for i in random_rows:
            uid = str(a.ids[i])
            x, y = a.find_similar(uid, 10), b.find_similar(uid, 10)
            identical += x == y
            overlap.append(len(set(x) & set(y)) / 10)
        report["modes"][mode] = {
            "development_products_with_identical_top_10": sum(
                top_a[no] == top_b[no] for no in setup.queries
            ),
            "development_products": len(setup.queries),
            "ndcg_at_10_full_size": round(score_a["ndcg_at_10"], 4),
            "ndcg_at_10_half_size": round(score_b["ndcg_at_10"], 4),
            "precision_at_10_full_size": round(score_a["precision_at_10"], 4),
            "precision_at_10_half_size": round(score_b["precision_at_10"], 4),
            "unrated_in_half_size_top_10": int(score_b["unrated_in_top_10"]),
            "random_products": len(random_rows),
            "random_products_with_identical_top_10": int(identical),
            "random_average_overlap_at_10": round(float(np.mean(overlap)), 4),
        }
        print(mode, report["modes"][mode], flush=True)
    (RESULTS / "vector_files_check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
