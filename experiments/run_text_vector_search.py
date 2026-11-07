"""Sentence-vector search on titles, and the blind rating page for new results.

Run from the project folder:  python experiments/run_text_vector_search.py
For each of the 30 development products, the sentence model picks the 10 most similar
titles (one copy per item; the 20 locked products are never candidates). Results that
already have a human rating are matched to it. The new ones go on a blind rating page.
No score is computed here.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from product_similarity import evaluation as ev
from product_similarity.dedupe import SAME_ITEM, pick_distinct, picture_table
from product_similarity.judging import SEED, write_json, write_sheet
from product_similarity.review_page import write_review_page
from product_similarity.service import baseline_engine

ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = ROOT / "evaluation" / "judgments"
RESULTS = ROOT / "results"
RATING_FILES = ["dev_grades.csv", "dev_grades_images.csv", "dev_grades_full_images.csv"]
DEPTH = 200  # how far down the ranking we look to find 10 different items


def load_ratings() -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {}
    for name in RATING_FILES:
        for no, by_id in ev.load_grades(JUDGMENTS / name).items():
            merged.setdefault(no, {}).update(by_id)
    return merged


def main() -> None:
    queries = json.loads((ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    locked = {q["id"] for q in queries if q["split"] == "final"}
    dev = {q["query_no"]: q["id"] for q in queries if q["split"] == "dev"}
    grades = load_ratings()
    engine = baseline_engine()
    products = engine.products

    text = np.load(ROOT / "data" / "embeddings" / "text_minilm.npz")
    assert [str(i) for i in text["ids"]] == [p.uniq_id for p in products], "vector order differs"
    vectors = text["vectors"]
    pics = np.load(ROOT / "data" / "embeddings" / "all_clip.npz")
    pic_unit = pics["vectors"] / np.linalg.norm(pics["vectors"], axis=1, keepdims=True)
    matrix, links = picture_table(
        products, {str(u): i for i, u in enumerate(pics["urls"])}, pic_unit
    )

    is_locked = np.array([p.uniq_id in locked for p in products])
    top10: dict[str, list[str]] = {}
    dropped = Counter()
    for no, pid in dev.items():
        q = engine.resolve(pid)
        scores = np.where(is_locked, -1.0, vectors @ vectors[q])
        ranked = np.argsort(-scores, kind="stable")[:DEPTH]
        kept, gone = pick_distinct(ranked.tolist(), matrix, links, q, ev.K)
        dropped.update(gone)
        top10[no] = [products[i].uniq_id for i in kept]
    assert not locked & {c for lst in top10.values() for c in lst}, "a locked product was returned"

    coverage = Counter()
    new: dict[str, list[str]] = {}
    for no, returned in top10.items():
        for cand in returned:
            if cand in grades.get(no, {}):
                coverage[f"rated_{grades[no][cand]}"] += 1
            else:
                coverage["unrated"] += 1
                new.setdefault(no, []).append(cand)
    total = sum(coverage.values())

    rows, sources = [], {}
    for no in sorted(new):
        order = sorted(new[no])
        random.Random(f"{SEED}-text-vectors-{no}").shuffle(order)
        qp = products[engine.resolve(dev[no])]
        sources[no] = {}
        for number, cid in enumerate(order, start=1):
            c = products[engine.resolve(cid)]
            sources[no][f"{number:02d}"] = {"id": cid, "methods": ["text_vectors"]}
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
    write_sheet(rows, JUDGMENTS / "dev_sheet_text_vectors.csv")
    write_json(sources, JUDGMENTS / "pool_sources_dev_text_vectors.json")
    write_review_page(
        rows,
        JUDGMENTS / "dev_review_text_vectors.html",
        key="ratings-text-vectors-v1",
        filename="dev_grades_text_vectors.csv",
    )
    summary = {
        "titles_searched": int((~is_locked).sum()),
        "same_item_threshold": SAME_ITEM,
        "dev_products": len(dev),
        "results_total": total,
        "already_rated": total - coverage["unrated"],
        "unrated": coverage["unrated"],
        "rated_0": coverage["rated_0"],
        "rated_1": coverage["rated_1"],
        "rated_2": coverage["rated_2"],
        "dropped_copy_of_query": dropped["copy_of_query"],
        "dropped_copy_of_result": dropped["copy_of_result"],
        "new_results_to_rate": len(rows),
        "products_with_new_results": len(new),
    }
    write_json(summary, RESULTS / "dev_text_vector_search_coverage.json")
    write_json(top10, ROOT / "data" / "embeddings" / "dev_text_top10.json")  # local only
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
