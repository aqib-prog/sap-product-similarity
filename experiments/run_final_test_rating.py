"""Make the blind rating page for the 20 locked final products. Run this once.

Run from the project folder:  python experiments/run_final_test_rating.py
The final search is run in both modes (exact, and fast with the 100 nearest products from each
index) for each of the 20 locked products, exactly as the service runs it, with no product
hidden. The top 10 of both modes are pooled into one blind page: shuffled for each product, with
no method names. Nothing is scored here, and the search, the weights and the code are final:
nothing may be tuned after seeing the result. After rating, save the CSV as
evaluation/judgments/final_grades.csv and run score_final_test.py.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from product_similarity.hybrid import HybridEngine
from product_similarity.judging import write_json, write_sheet
from product_similarity.review_page import write_review_page
from product_similarity.service import baseline_engine
from product_similarity.vector_files import load_vector_artifact

ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = ROOT / "evaluation" / "judgments"
SEED = 2020
NEAREST = 100  # from each of the two indexes, in fast mode


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    base = baseline_engine()
    vectors = load_vector_artifact(base.products, ROOT / "artifacts" / "vectors")
    assert vectors is not None, "the vector files are missing"
    engines = {
        "exact": HybridEngine(base.products, base.id_map, vectors, "exact"),
        "fast": HybridEngine(base.products, base.id_map, vectors, "fast", NEAREST),
    }
    queries = json.loads((ROOT / "evaluation" / "queries.json").read_text(encoding="utf-8"))
    locked = [q for q in queries if q["split"] == "final"]
    assert len(locked) == 20

    rows, sources, counts = [], {}, {}
    for q in locked:
        found: dict[str, list[str]] = {}
        for name, engine in engines.items():
            for uid in engine.find_similar(q["id"], 10):
                found.setdefault(uid, []).append(name)
        order = sorted(found)
        random.Random(f"{SEED}-final-{q['query_no']}").shuffle(order)
        query = base.products[base.resolve(q["id"])]
        sources[q["query_no"]] = {}
        counts[q["query_no"]] = len(order)
        for number, cid in enumerate(order, start=1):
            c = base.products[base.resolve(cid)]
            sources[q["query_no"]][f"{number:02d}"] = {"id": cid, "methods": found[cid]}
            rows.append(
                {
                    "query_no": q["query_no"],
                    "query_title": query.title,
                    "query_category": query.category,
                    "query_price": query.price,
                    "query_image_url": query.image_url,
                    "candidate_no": f"{number:02d}",
                    "candidate_title": c.title,
                    "candidate_category": c.category,
                    "candidate_brand": c.brand,
                    "candidate_price": c.price,
                    "candidate_image_url": c.image_url,
                    "candidate_id": c.uniq_id,
                    "grade": "",
                }
            )
    write_sheet(rows, JUDGMENTS / "final_sheet.csv")
    write_json(sources, JUDGMENTS / "pool_sources_final.json")
    write_review_page(
        rows,
        JUDGMENTS / "final_review.html",
        key="ratings-final-v1",
        filename="final_grades.csv",
    )
    write_json(
        {"products": len(locked), "results_to_rate": len(rows), "per_product": counts},
        ROOT / "results" / "final_test_coverage.json",
    )
    print(f"{len(locked)} locked products, {len(rows)} results to rate")


if __name__ == "__main__":
    main()
