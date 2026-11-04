"""Shared helpers for building small test catalogues."""

import pytest

from product_similarity.data import canonicalize
from product_similarity.preprocessing import build_products
from product_similarity.scoring import SimilarityEngine


@pytest.fixture
def make_row():
    def _make(uid, title, asin=None, **extra):
        row = {
            "uniq_id": uid,
            "asin": asin or f"ASIN{uid}",
            "product_name": title,
            "crawl_timestamp": "2020-02-07 05:00:00 +0000",
            "weight": "999999999",
        }
        row.update(extra)
        return row

    return _make


@pytest.fixture
def make_engine():
    def _make(rows, **kwargs):
        kept, id_map = canonicalize(rows)
        products, _ = build_products(kept)
        return SimilarityEngine(products, id_map, **kwargs)

    return _make


@pytest.fixture
def catalogue(make_row):
    """Two similar T-shirts, an unrelated saree and a product with only a title."""
    shirt = {"sales_rank_in_child_category": {"MensT_Shirts": "#1"}, "browsenode": "N1"}
    saree = {"sales_rank_in_child_category": {"WomensSarees": "#1"}, "browsenode": "N2"}
    return [
        make_row(
            "a", "Blue Cotton Mens T-Shirt", brand="Acme", sales_price=500.0, rating=4.0, **shirt
        ),
        make_row(
            "b",
            "Blue Cotton Mens T-Shirt Round Neck",
            brand="Acme",
            sales_price=520.0,
            rating=4.2,
            **shirt,
        ),
        make_row(
            "c", "Red Silk Womens Saree", brand="Zed", sales_price=3000.0, rating=3.0, **saree
        ),
        make_row("d", "Mystery item"),
    ]
