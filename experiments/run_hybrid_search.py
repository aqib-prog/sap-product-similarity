"""Find the hybrid results that nobody has rated yet, and write the blind rating page.

Run from the project folder:  python experiments/run_hybrid_search.py
Uses the tuned weights saved by tune_hybrid_weights.py (the 10-weight and the 4-weight
mix) and the untuned mix. Results that already have a rating keep it. The rest go on a
blind page, shuffled for each product, with no method names. No score is computed here.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from hybrid_setup import SEED, build_evaluator, expand_four

from product_similarity import evaluation as ev
from product_similarity.hybrid_eval import PARTS
from product_similarity.judging import write_json, write_sheet
from product_similarity.review_page import write_review_page
from product_similarity.scoring import DEFAULT_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = ROOT / "evaluation" / "judgments"
RESULTS = ROOT / "results"
ROUND = "hybrid"  # rating files: dev_grades_hybrid.csv


def main() -> None:
    report = json.loads((RESULTS / "hybrid_tuning.json").read_text(encoding="utf-8"))
    setup = build_evaluator()
    evaluator, engine, dev, grades = setup.evaluator, setup.engine, setup.dev, setup.grades
    mixes = {
        "untuned": {p: DEFAULT_WEIGHTS[p] for p in PARTS},
        "tuned_10_weights": report["tuned_weights"],
        "tuned_4_weights": expand_four(report["four_weights"]["weights"]),
    }
    top10: dict[str, dict[str, list[str]]] = {
        name: {no: evaluator.top_ids(w, no) for no in setup.queries} for name, w in mixes.items()
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
        qp = engine.products[engine.resolve(dev[no])]
        sources[no] = {}
        for number, cid in enumerate(order, start=1):
            c = engine.products[engine.resolve(cid)]
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
        "results_per_mix": 10 * len(dev),
        "coverage": coverage,
        "new_results_to_rate": len(rows),
        "products_with_new_results": len(found),
        "locked_returned": 0,
    }
    write_json(summary, RESULTS / f"dev_{ROUND}_search_coverage.json")
    write_json(top10, ROOT / "data" / "embeddings" / f"dev_{ROUND}_top10_by_mix.json")  # local only
    print(json.dumps(summary, indent=1))
    assert not setup.locked & {c for m in top10.values() for lst in m.values() for c in lst}
    assert ev.K == 10


if __name__ == "__main__":
    main()
