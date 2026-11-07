"""Image similarity search for the development products, and the blind rating sheet.

Run from the project folder:  python experiments/run_image_search.py
For each of the 30 development products, ResNet and CLIP each pick the 10 most
similar pictures among the downloaded ones. Results that already have a human
rating are matched to it. The new ones go on a blind sheet for rating.
No score is computed here: unrated results are not treated as bad matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from product_similarity import evaluation as ev
from product_similarity.image_features import top_k
from product_similarity.judging import SEED, write_json, write_sheet
from product_similarity.review_page import write_review_page
from product_similarity.service import baseline_engine
from product_similarity.tracking import log_run, mlflow_available

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT / "data" / "embeddings"
JUDGMENTS = ROOT / "evaluation" / "judgments"
RESULTS = ROOT / "results"
MODELS = ("resnet50", "clip")


def load_vectors(name: str) -> tuple[list[str], np.ndarray]:
    data = np.load(EMBEDDINGS / f"dev_{name}.npz")
    return [str(i) for i in data["ids"]], data["vectors"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    queries = json.loads((ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    locked = {q["id"] for q in queries if q["split"] == "final"}
    dev = {q["query_no"]: q["id"] for q in queries if q["split"] == "dev"}
    grades = ev.load_grades(JUDGMENTS / "dev_grades.csv")
    engine = baseline_engine()

    top10: dict[str, dict[str, list[str]]] = {}
    missing: dict[str, list[str]] = {}
    for name in MODELS:
        ids, vectors = load_vectors(name)
        assert not locked & set(ids), "a locked product is in the embeddings"
        row_of = {uid: i for i, uid in enumerate(ids)}
        top10[name], missing[name] = {}, []
        for no, pid in dev.items():
            if pid not in row_of:
                missing[name].append(no)
                continue
            top10[name][no] = [ids[i] for i in top_k(vectors, row_of[pid], ev.K)]

    # Which results already have a rating, and which are new.
    coverage: dict[str, dict] = {}
    new_pairs: dict[tuple[str, str], list[str]] = {}
    for name in MODELS:
        counts = Counter()
        for no, returned in top10[name].items():
            for candidate in returned:
                if candidate in grades.get(no, {}):
                    counts[f"rated_{grades[no][candidate]}"] += 1
                else:
                    counts["unrated"] += 1
                    new_pairs.setdefault((no, candidate), []).append(name)
        total = sum(counts.values())
        coverage[name] = {
            "queries_searched": len(top10[name]),
            "queries_without_picture": missing[name],
            "results_total": total,
            "already_rated": total - counts["unrated"],
            "unrated": counts["unrated"],
            "share_rated": round((total - counts["unrated"]) / total, 3) if total else 0,
            "rated_0": counts["rated_0"],
            "rated_1": counts["rated_1"],
            "rated_2": counts["rated_2"],
        }

    # Blind sheet: only the new results, shuffled per product, no method names.
    by_query: dict[str, list[str]] = {}
    for (no, candidate), _ in new_pairs.items():
        by_query.setdefault(no, []).append(candidate)
    rows, sources = [], {}
    for no in sorted(by_query):
        order = sorted(by_query[no])
        random.Random(f"{SEED}-images-{no}").shuffle(order)
        query_product = engine.products[engine.resolve(dev[no])]
        sources[no] = {}
        for number, candidate_id in enumerate(order, start=1):
            c = engine.products[engine.resolve(candidate_id)]
            sources[no][f"{number:02d}"] = {
                "id": candidate_id,
                "methods": new_pairs[(no, candidate_id)],
            }
            rows.append(
                {
                    "query_no": no,
                    "query_title": query_product.title,
                    "query_category": query_product.category,
                    "query_price": query_product.price,
                    "query_image_url": query_product.image_url,
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
    sheet = JUDGMENTS / "dev_sheet_images.csv"
    write_sheet(rows, sheet)
    write_json(sources, JUDGMENTS / "pool_sources_dev_images.json")
    write_review_page(
        rows,
        JUDGMENTS / "dev_review_images.html",
        key="ratings-images-v1",
        filename="dev_grades_images.csv",
    )

    RESULTS.mkdir(exist_ok=True)
    summary = {
        "models": coverage,
        "new_results_to_rate": len(rows),
        "products_with_new_results": len(by_query),
    }
    write_json(summary, RESULTS / "dev_image_search_coverage.json")
    write_json(top10, ROOT / "data" / "embeddings" / "dev_image_top10.json")  # local only

    for name, c in coverage.items():
        print(name, c)
    print("new results to rate:", len(rows), "for", len(by_query), "products")

    if args.no_mlflow or not mlflow_available():
        return
    import mlflow

    speed = json.loads((RESULTS / "dev_image_embedding_speed.json").read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    for name in MODELS:
        c = coverage[name]
        log_run(
            "dev-images",
            name,
            {
                "model": name,
                "search": "cosine similarity on picture vectors",
                "search_space": f"{speed['pictures']} downloaded pictures only",
                "k": ev.K,
                "dev_products": len(dev),
                "dimensions": speed[name]["dimensions"],
                "ratings_sha256": hashlib.sha256(
                    (JUDGMENTS / "dev_grades.csv").read_bytes()
                ).hexdigest()[:16],
                "git_commit": commit,
                "mlflow": mlflow.__version__,
            },
            {
                "ms_per_picture": speed[name]["ms_per_image"],
                "share_of_top10_already_rated": c["share_rated"],
                "unrated_in_top10": c["unrated"],
            },
            [
                RESULTS / "dev_image_search_coverage.json",
                RESULTS / "dev_image_embedding_speed.json",
            ],
            tags={
                "status": "waiting for ratings: no nDCG yet",
                "evaluation_set": "development (30 products)",
                "final_set": "locked, not used",
                "limitations": "Image search only looks at the downloaded pictures, not all 30,000 products. "
                "Unrated results are not treated as bad matches. One person rates.",
            },
        )
    print("MLflow: 2 runs recorded in experiment 'dev-images'")


if __name__ == "__main__":
    main()
