"""Find the results of the final search that nobody has rated yet, and write the blind page.

Run from the project folder:  python experiments/run_final_search_rating.py
The final search (`HybridEngine`) is run in both modes (in fast mode, the 100 nearest products
by sentence vector and by picture vector are scored). Results that already have a rating keep it. The rest go on
a blind page, shuffled for each product, with no method names. No score is computed here.
The 20 locked products are hidden from every search.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from hybrid_setup import SEED, build_evaluator

from product_similarity.hybrid import HybridEngine, load_final_vectors
from product_similarity.judging import write_json, write_sheet
from product_similarity.review_page import write_review_page

ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = ROOT / "evaluation" / "judgments"
RESULTS = ROOT / "results"
ROUND = "final_search_fast"  # rating file: dev_grades_final_search_fast.csv
CANDIDATES = 100  # for each vector part


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    setup = build_evaluator()
    base, dev, grades = setup.engine, setup.dev, setup.grades
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
            base.products, base.id_map, vectors, "fast", CANDIDATES, hidden_ids=setup.locked
        ),
    }
    top10 = {
        name: {no: engine.find_similar(dev[no], 10) for no in setup.queries}
        for name, engine in engines.items()
    }
    found: dict[str, dict[str, list[str]]] = {}
    coverage = {}
    for name, lists in top10.items():
        counts = Counter()
        for no, returned in lists.items():
            for cand in returned:
                if cand in grades.get(no, {}):
                    counts["rated"] += 1
                else:
                    counts["unrated"] += 1
                    found.setdefault(no, {}).setdefault(cand, []).append(name)
        coverage[name] = dict(counts)

    rows, sources = [], {}
    for no in sorted(found):
        order = sorted(found[no])
        random.Random(f"{SEED}-{ROUND}-{no}").shuffle(order)
        qp = base.products[base.resolve(dev[no])]
        sources[no] = {}
        for number, cid in enumerate(order, start=1):
            c = base.products[base.resolve(cid)]
            sources[no][f"{number:02d}"] = {"id": cid, "methods": found[no][cid]}
            rows.append(
                {
                    "query_no": no,
                    "query_title": qp.title,
                    "query_category": qp.category,
                    "query_price": qp.price,
                    "query_image_url": qp.image_url,
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
    write_sheet(rows, JUDGMENTS / f"dev_sheet_{ROUND}.csv")
    write_json(sources, JUDGMENTS / f"pool_sources_dev_{ROUND}.json")
    write_review_page(
        rows,
        JUDGMENTS / f"dev_review_{ROUND}.html",
        key=f"ratings-{ROUND}-v1",
        filename=f"dev_grades_{ROUND}.csv",
    )
    summary = {
        "results_per_search": 10 * len(dev),
        "coverage": coverage,
        "new_results_to_rate": len(rows),
        "products_with_new_results": len(found),
        "locked_returned": 0,
    }
    write_json(summary, RESULTS / f"dev_{ROUND}_coverage.json")
    write_json(top10, ROOT / "data" / "embeddings" / f"dev_{ROUND}_top10.json")  # local only
    print(json.dumps(summary, indent=1))
    assert not setup.locked & {c for m in top10.values() for lst in m.values() for c in lst}


if __name__ == "__main__":
    main()
