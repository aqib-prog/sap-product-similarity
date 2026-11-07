"""Choose test products and build the blind rating sheet.

The sheet mixes the top results of several methods into one shuffled list per
test product. It never says which method found which result, so the person
rating cannot favour a method. The hidden mapping is saved in a separate file.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from product_similarity.preprocessing import Product
from product_similarity.review_page import write_review_page
from product_similarity.scoring import DEFAULT_WEIGHTS, SimilarityEngine

SEED = 2020

# (group name, how many test products to take from it)
STRATA = (
    ("common category", 12),
    ("rare category", 6),
    ("sparse data", 8),
    ("complete data", 8),
    ("repeated title", 8),
    ("low price", 4),
    ("high price", 4),
)

SHEET_COLUMNS = [
    "query_no",
    "query_title",
    "query_category",
    "query_price",
    "query_image_url",
    "candidate_no",
    "candidate_title",
    "candidate_category",
    "candidate_brand",
    "candidate_price",
    "candidate_image_url",
    "candidate_id",
    "grade",
]


def attributes_known(product: Product) -> int:
    """How many of the 8 compared attributes this product has (the title always counts)."""
    return sum(
        [
            True,
            product.category is not None,
            product.brand is not None,
            product.audience is not None,
            product.price is not None,
            bool(product.title_colours or product.listed_colours),
            product.weight_g is not None,
            product.rating is not None,
        ]
    )


def select_queries(products: Sequence[Product], seed: int = SEED) -> list[dict]:
    """Pick test products from varied groups. The same seed always gives the same list.

    Two of every five products go to the locked `final` set (20 in all) and the rest
    to `dev` (30), so both sets cover every group.
    """
    rng = random.Random(seed)
    category_size = Counter(p.category for p in products if p.category)
    title_count = Counter(p.title for p in products)
    groups = {
        "common category": lambda p: p.category and category_size[p.category] >= 500,
        "rare category": lambda p: p.category and category_size[p.category] < 50,
        "sparse data": lambda p: attributes_known(p) <= 4,
        "complete data": lambda p: attributes_known(p) >= 7,
        "repeated title": lambda p: title_count[p.title] >= 3,
        "low price": lambda p: p.price is not None and p.price < 300,
        "high price": lambda p: p.price is not None and p.price > 2000,
    }
    chosen: list[tuple[str, Product]] = []
    taken: set[str] = set()
    for name, count in STRATA:
        pool = sorted(
            (p for p in products if groups[name](p) and p.uniq_id not in taken),
            key=lambda p: p.uniq_id,
        )
        for product in rng.sample(pool, min(count, len(pool))):
            taken.add(product.uniq_id)
            chosen.append((name, product))
    queries = []
    for position, (name, product) in enumerate(chosen):
        queries.append(
            {
                "query_no": f"Q{position + 1:02d}",
                "id": product.uniq_id,
                "title": product.title,
                "group": name,
                "split": "final" if position % 5 in (2, 4) else "dev",
            }
        )
    return queries


def comparison_methods(
    products: Sequence[Product], id_map: dict[str, str]
) -> dict[str, SimilarityEngine]:
    """The methods whose results are pooled for rating."""
    only_title = {**dict.fromkeys(DEFAULT_WEIGHTS, 0.0), "title": 1.0}
    return {
        "baseline": SimilarityEngine(products, id_map),
        "title_only": SimilarityEngine(products, id_map, only_title),
        "attributes_only": SimilarityEngine(products, id_map, {"title": 0.0}),
    }


def build_pool(
    methods: dict[str, SimilarityEngine], query_id: str, per_method: int = 10
) -> dict[str, list[str]]:
    """Map each candidate product to the methods that returned it."""
    pool: dict[str, list[str]] = {}
    for name, engine in methods.items():
        for candidate in engine.find_similar(query_id, per_method):
            pool.setdefault(candidate, []).append(name)
    return pool


def build_sheet(
    methods: dict[str, SimilarityEngine],
    queries: Sequence[dict],
    seed: int = SEED,
) -> tuple[list[dict], dict[str, dict[str, list[str]]]]:
    """Return the blind rating rows and the hidden candidate-to-method mapping."""
    any_engine = next(iter(methods.values()))
    rows: list[dict] = []
    sources: dict[str, dict[str, list[str]]] = {}
    for query in queries:
        product = any_engine.products[any_engine.resolve(query["id"])]
        pool = build_pool(methods, query["id"])
        order = sorted(pool)
        random.Random(f"{seed}-{query['query_no']}").shuffle(order)
        sources[query["query_no"]] = {}
        for number, candidate_id in enumerate(order, start=1):
            candidate = any_engine.products[any_engine.resolve(candidate_id)]
            sources[query["query_no"]][f"{number:02d}"] = pool[candidate_id]
            rows.append(
                {
                    "query_no": query["query_no"],
                    "query_title": product.title,
                    "query_category": product.category,
                    "query_price": product.price,
                    "query_image_url": product.image_url,
                    "candidate_no": f"{number:02d}",
                    "candidate_title": candidate.title,
                    "candidate_category": candidate.category,
                    "candidate_brand": candidate.brand,
                    "candidate_price": candidate.price,
                    "candidate_image_url": candidate.image_url,
                    "candidate_id": candidate.uniq_id,
                    "grade": "",
                }
            )
    return rows, sources


def write_sheet(rows: Sequence[dict], path: Path) -> None:
    """Write the sheet as a CSV that opens correctly in Excel, Numbers and Google Sheets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row[key] is None else row[key]) for key in SHEET_COLUMNS})


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    """Choose the test products and write the files under evaluation/."""
    from product_similarity.service import baseline_engine

    base = baseline_engine()
    products, id_map = base.products, base.id_map
    queries = select_queries(products)
    write_json(queries, Path("evaluation/queries.json"))
    dev = [q for q in queries if q["split"] == "dev"]
    methods = comparison_methods(products, id_map)
    rows, sources = build_sheet(methods, dev)
    write_sheet(rows, Path("evaluation/judgments/dev_sheet.csv"))
    write_json(sources, Path("evaluation/judgments/pool_sources_dev.json"))
    write_review_page(rows, Path("evaluation/judgments/dev_review.html"))
    print(f"{len(queries)} test products ({len(dev)} dev, {len(queries) - len(dev)} final)")
    print(f"{len(rows)} results to rate: open evaluation/judgments/dev_review.html in a browser")


if __name__ == "__main__":
    main()
