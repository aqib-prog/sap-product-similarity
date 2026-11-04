"""Test-product selection, pooling and the blind sheet."""

import csv
import json

import pytest

from product_similarity.data import canonicalize
from product_similarity.judging import (
    SHEET_COLUMNS,
    attributes_known,
    build_pool,
    build_sheet,
    comparison_methods,
    select_queries,
    write_sheet,
)
from product_similarity.preprocessing import build_products


@pytest.fixture(scope="module")
def catalogue():
    import random

    big = [
        ("MensT_Shirts", "Blue Cotton Mens T-Shirt", 250.0),
        ("WomensSarees", "Silk Womens Saree with Blouse", 2500.0),
        ("BoysCaps", "Boys Baseball Cap", 150.0),
    ]
    rows = []
    for i in range(1800):
        category, title, price = big[i % 3]
        sparse = i % 7 == 0
        rows.append(
            {
                "uniq_id": f"{i:032x}",
                "asin": f"A{i}",
                "product_name": f"{title} {i % 40}",
                "crawl_timestamp": "2020-02-07 05:00:00 +0000",
                "weight": "999999999" if sparse or i % 3 else "250 g",
                "sales_price": None if sparse else price + (i % 17),
                "rating": None if sparse else 3.0 + (i % 20) / 10,
                "brand": None if sparse or i % 4 == 1 else f"Brand{i % 9}",
                "colour": "Blue|Red" if i % 2 == 0 and not sparse else None,
                "sales_rank_in_child_category": {} if sparse else {category: "#1"},
                "browsenode": f"N{i % 3}",
            }
        )
    for j in range(60):
        rows.append(
            {
                "uniq_id": f"{10_000 + j:032x}",
                "asin": f"R{j}",
                "product_name": f"Rare Item {j}",
                "crawl_timestamp": "2020-02-07 05:00:00 +0000",
                "weight": "999999999",
                "sales_price": 400.0 + j,
                "sales_rank_in_child_category": {f"RareCategory{j % 12}": "#1"},
                "browsenode": f"R{j % 12}",
            }
        )
    random.Random(0).shuffle(rows)
    kept, id_map = canonicalize(rows)
    products, _ = build_products(kept)
    return products, id_map


def test_selection_is_repeatable_and_has_no_duplicates(catalogue):
    products, _ = catalogue
    first, second = select_queries(products), select_queries(products)
    assert first == second
    ids = [q["id"] for q in first]
    assert len(ids) == len(set(ids)) == 50


def test_split_is_thirty_dev_and_twenty_final(catalogue):
    queries = select_queries(catalogue[0])
    splits = [q["split"] for q in queries]
    assert splits.count("dev") == 30 and splits.count("final") == 20


def test_every_group_is_present_in_both_sets(catalogue):
    queries = select_queries(catalogue[0])
    for split in ("dev", "final"):
        assert {q["group"] for q in queries if q["split"] == split} == {
            "common category",
            "rare category",
            "sparse data",
            "complete data",
            "repeated title",
            "low price",
            "high price",
        }


def test_attributes_known_counts_the_title_at_least(catalogue):
    assert all(1 <= attributes_known(p) <= 8 for p in catalogue[0])


def test_pool_has_no_query_and_records_which_method_found_what(catalogue):
    products, id_map = catalogue
    methods = comparison_methods(products, id_map)
    query = products[0].uniq_id
    pool = build_pool(methods, query)
    assert query not in pool
    assert all(set(found) <= set(methods) for found in pool.values())
    assert 10 <= len(pool) <= 30


def test_sheet_is_blind_and_complete(catalogue, tmp_path):
    products, id_map = catalogue
    methods = comparison_methods(products, id_map)
    queries = [q for q in select_queries(products) if q["split"] == "dev"][:3]
    rows, sources = build_sheet(methods, queries)
    write_sheet(rows, tmp_path / "sheet.csv")
    text = (tmp_path / "sheet.csv").read_text(encoding="utf-8-sig")
    assert "baseline" not in text and "title_only" not in text and "attributes_only" not in text
    with open(tmp_path / "sheet.csv", newline="", encoding="utf-8-sig") as handle:
        read = list(csv.DictReader(handle))
    assert list(read[0]) == SHEET_COLUMNS
    assert all(r["grade"] == "" for r in read)
    assert {r["query_no"] for r in read} == {q["query_no"] for q in queries}
    assert set(sources) == {q["query_no"] for q in queries}
    json.dumps(sources)


def test_shuffling_is_repeatable_and_not_sorted_by_method(catalogue):
    products, id_map = catalogue
    methods = comparison_methods(products, id_map)
    queries = select_queries(products)[:1]
    rows_a, _ = build_sheet(methods, queries)
    rows_b, _ = build_sheet(methods, queries)
    assert [r["candidate_id"] for r in rows_a] == [r["candidate_id"] for r in rows_b]
    first_ten = [r["candidate_id"] for r in rows_a[:10]]
    baseline_top = methods["baseline"].find_similar(queries[0]["id"], 10)
    assert first_ten != baseline_top
