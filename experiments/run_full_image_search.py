"""CLIP image search over the whole catalogue, and the blind sheet for new results.

Run from the project folder:  python experiments/run_full_image_search.py
For each of the 30 development products, CLIP picks the 10 most similar items among all
downloaded pictures (one copy per item, see product_similarity/dedupe.py). Results that
already have a human rating are matched to it; the new ones go on a blind rating page.
Unrated results are not treated as bad matches and no score is computed here.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from product_similarity import evaluation as ev
from product_similarity.dedupe import SAME_ITEM, pick_distinct
from product_similarity.judging import SEED, write_json, write_sheet
from product_similarity.review_page import write_review_page
from product_similarity.service import default_engine

ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = ROOT / "evaluation" / "judgments"
RESULTS = ROOT / "results"
SHARED = ROOT / "data" / "images" / "shared" / "excluded.json"


def load_ratings() -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {}
    for name in ("dev_grades.csv", "dev_grades_images.csv"):
        for no, by_id in ev.load_grades(JUDGMENTS / name).items():
            merged.setdefault(no, {}).update(by_id)
    return merged


def main() -> None:
    queries = json.loads((ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    locked = {q["id"] for q in queries if q["split"] == "final"}
    dev = {q["query_no"]: q["id"] for q in queries if q["split"] == "dev"}
    grades = load_ratings()
    engine = default_engine()
    not_photos = {r["url"] for r in json.loads(SHARED.read_text(encoding="utf-8"))}

    data = np.load(ROOT / "data" / "embeddings" / "all_clip.npz")
    vec_of = {str(u): i for i, u in enumerate(data["urls"])}
    vectors = data["vectors"] / np.linalg.norm(data["vectors"], axis=1, keepdims=True)
    products = [p for p in engine.products if p.image_url in vec_of]
    assert not locked & {p.uniq_id for p in products}, "a locked product is in the search"
    matrix = vectors[[vec_of[p.image_url] for p in products]]
    links = [p.image_url for p in products]
    row_of = {p.uniq_id: i for i, p in enumerate(products)}
    allowed = np.array([p.image_url not in not_photos for p in products])

    top10: dict[str, list[str]] = {}
    dropped_total: Counter = Counter()
    for no, pid in dev.items():
        q = row_of[pid]
        scores = np.where(allowed, matrix @ matrix[q], -1.0)
        ranked = np.argsort(-scores, kind="stable")[:200]
        kept, dropped = pick_distinct(ranked.tolist(), matrix, links, q, ev.K)
        dropped_total.update(dropped)
        top10[no] = [products[i].uniq_id for i in kept]

    coverage = Counter()
    new_pairs: dict[str, list[str]] = {}
    for no, returned in top10.items():
        for cand in returned:
            if cand in grades.get(no, {}):
                coverage[f"rated_{grades[no][cand]}"] += 1
            else:
                coverage["unrated"] += 1
                new_pairs.setdefault(no, []).append(cand)
    total = sum(coverage.values())

    rows, sources = [], {}
    for no in sorted(new_pairs):
        order = sorted(new_pairs[no])
        random.Random(f"{SEED}-full-images-{no}").shuffle(order)
        qp = engine.products[engine.resolve(dev[no])]
        sources[no] = {}
        for number, cid in enumerate(order, start=1):
            c = engine.products[engine.resolve(cid)]
            sources[no][f"{number:02d}"] = {"id": cid, "methods": ["clip_all"]}
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
    write_sheet(rows, JUDGMENTS / "dev_sheet_full_images.csv")
    write_json(sources, JUDGMENTS / "pool_sources_dev_full_images.json")
    write_review_page(
        rows,
        JUDGMENTS / "dev_review_full_images.html",
        key="ratings-full-images-v1",
        filename="dev_grades_full_images.csv",
    )
    summary = {
        "pictures_searched": int(allowed.sum()),
        "products_searched": len(products),
        "excluded_not_product_photos": len(not_photos),
        "same_item_threshold": SAME_ITEM,
        "dev_products": len(dev),
        "results_total": total,
        "already_rated": total - coverage["unrated"],
        "unrated": coverage["unrated"],
        "rated_0": coverage["rated_0"],
        "rated_1": coverage["rated_1"],
        "rated_2": coverage["rated_2"],
        "dropped_copy_of_query": dropped_total["copy_of_query"],
        "dropped_copy_of_result": dropped_total["copy_of_result"],
        "new_results_to_rate": len(rows),
        "products_with_new_results": len(new_pairs),
    }
    write_json(summary, RESULTS / "dev_full_image_search_coverage.json")
    write_json(top10, ROOT / "data" / "embeddings" / "dev_full_clip_top10.json")  # local only
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
