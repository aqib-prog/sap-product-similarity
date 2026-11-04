"""Tests for cleaning, derived fields and the product pipeline."""

import math

import pytest

from product_similarity.data import canonicalize
from product_similarity.preprocessing import (
    build_brand_lookup,
    build_brand_map,
    build_node_category_map,
    build_products,
    clean_brand,
    clean_title,
    derive_audience,
    extract_colours,
    first_image_url,
    parse_page_reviews,
    parse_weight,
    price_fields,
    primary_category,
    recover_brand_from_title,
)


def _row(uniq_id, **extra):
    base = {
        "uniq_id": uniq_id,
        "asin": "A" + uniq_id,
        "product_name": "Test Shirt",
        "crawl_timestamp": "2020-02-07 05:00:00 +0000",
    }
    base.update(extra)
    return base


# ---- weight ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, grams",
    [
        ("200 g", 200.0),
        ("99.8 g", 99.8),
        ("1.4 Kg", 1400.0),
        ("2 Kilograms", 2000.0),
        ("5 Grams", 5.0),
    ],
)
def test_parse_weight_converts_units_to_grams(raw, grams):
    assert parse_weight(raw) == (pytest.approx(grams), None)


@pytest.mark.parametrize(
    "raw, problem",
    [
        ("999999999", "placeholder"),
        (None, "placeholder"),
        ("136 Kilograms", "out_of_range"),
        ("0.5 g", "out_of_range"),
        ("12 ounces", "unknown_unit"),
        ("The Slim shirt even smoothest out your back", "not_a_weight"),
    ],
)
def test_parse_weight_rejects_with_reason(raw, problem):
    assert parse_weight(raw) == (None, problem)


def test_parse_weight_handles_non_breaking_space():
    assert parse_weight("249\xa0g")[0] == 249.0


# ---- text and brand -------------------------------------------------------


def test_clean_title_fixes_spacing_but_keeps_sizes_and_packs():
    assert clean_title("  Shirt\xa0(Pack of 2)   XL ") == "Shirt (Pack of 2) XL"


def test_clean_title_repairs_broken_characters():
    assert clean_title("Speedo Womenâ€™s Swimsuit") == "Speedo Women\u2019s Swimsuit"
    assert clean_title("CTMÂ® Long Bill Cap") == "CTM\u00ae Long Bill Cap"
    assert clean_title("Hat 360Ã‚Â° Protection") == "Hat 360\u00b0 Protection"


def test_clean_title_leaves_normal_accents_alone():
    assert clean_title("Art Décor Silk Saree") == "Art Décor Silk Saree"


def test_clean_brand_nulls_placeholders_and_trims():
    assert clean_brand("  Nike   Air ") == "Nike Air"
    assert clean_brand("Unknown") is None
    assert clean_brand("N/A") is None
    assert clean_brand(None) is None
    assert clean_brand("Generic") == "Generic"


def test_brand_map_merges_spelling_variants_to_most_common():
    mapping = build_brand_map(["Print Mafias"] * 3 + ["print mafias", "Nautinati", "nauti nati"])
    assert mapping["print mafias"] == "Print Mafias"
    assert mapping["Nautinati"] == mapping["nauti nati"]
    assert mapping["Print Mafias"] == "Print Mafias"


def test_brand_recovery_matches_longest_known_brand_prefix():
    lookup = build_brand_lookup(["Allen Solly", "Allen", "Puma"])
    assert recover_brand_from_title("Allen Solly Men's Shirt", lookup) == "Allen Solly"
    assert recover_brand_from_title("Puma Unisex Tee", lookup) == "Puma"
    assert recover_brand_from_title("Plain cotton tee", lookup) is None


# ---- category -------------------------------------------------------------


def test_primary_category_uses_leaf_and_ignores_catch_all():
    assert (
        primary_category({"sales_rank_in_child_category": {"MensT_Shirts": "#5"}}) == "MensT_Shirts"
    )
    assert primary_category({"sales_rank_in_child_category": {"Fashion": "#5"}}) is None
    assert primary_category({}) is None


def test_node_map_fills_missing_category():
    rows = [
        _row("1", browsenode="N1", sales_rank_in_child_category={"WomensSarees": "#1"}),
        _row("2", browsenode="N1", sales_rank_in_child_category={"WomensSarees": "#2"}),
        _row("3", browsenode="N1", sales_rank_in_child_category={"Fashion": "#3"}),
        _row("4", browsenode="N1"),
    ]
    assert build_node_category_map(rows) == {"N1": "WomensSarees"}
    products, report = build_products(rows)
    assert [p.category for p in products] == ["WomensSarees"] * 4
    assert [p.category_source for p in products] == ["child", "child", "node", "node"]
    assert report.category_from_node == 2


# ---- reviews, price, image ------------------------------------------------


def test_page_reviews_states():
    counted = {
        "product_details__k_v_pairs": {
            "Customer_Reviews": "4.5 out of 5 stars 1,234 customer reviews"
        }
    }
    assert parse_page_reviews(counted) == (4.5, 1234, "counted")
    none = {"product_details__k_v_pairs": {"Customer_Reviews": "Be the first to review this item"}}
    assert parse_page_reviews(none) == (None, 0, "none")
    assert parse_page_reviews({}) == (None, None, "unknown")
    broken = {"product_details__k_v_pairs": {"Customer_Reviews": "3.0 out of 5 stars"}}
    assert parse_page_reviews(broken) == (None, None, "unknown")


def test_price_fields():
    price, log_price = price_fields({"sales_price": 1000.0})
    assert price == 1000.0 and log_price == pytest.approx(3.0)
    assert price_fields({"sales_price": None}) == (None, None)
    assert price_fields({"sales_price": 0.0}) == (None, None)
    assert price_fields({}) == (None, None)


def test_first_image_url():
    assert first_image_url({"medium": "https://x/a.jpg|https://x/b.jpg"}) == "https://x/a.jpg"
    assert first_image_url({"medium": ""}) is None
    assert first_image_url({}) is None


# ---- colour and audience --------------------------------------------------


def test_extract_colours_uses_fixed_vocabulary():
    assert extract_colours("Men's Navy Blue Cotton Shirt") == ("navy",)
    assert extract_colours("Black & White Stripes, grey sole") == ("black", "grey", "white")
    assert extract_colours("Blackberry phone cover") == ()
    assert extract_colours(None) == ()


def test_derive_audience_prefers_category_then_title():
    assert derive_audience("Plain top", "WomensTops") == "women"
    assert derive_audience("Men's Slim Fit Shirt", None) == "men"
    assert derive_audience("Women's Kurta", None) == "women"
    assert derive_audience("Girls Frock", None) == "girls"
    assert derive_audience("Men & Women Sports Cap", None) == "unisex"
    assert derive_audience("Cotton Towel", None) is None
    assert derive_audience("Baby Girls' Knit Top", None) == "baby"
    assert derive_audience("Kids Boys Half Sleeve T-Shirt", None) == "boys"
    assert derive_audience("Boys & Girls Cotton Cap", None) == "unisex"


# ---- duplicates and the full pipeline -------------------------------------


def test_canonicalize_keeps_latest_then_lowest_id_and_maps_every_id():
    rows = [
        _row("b", asin="X", crawl_timestamp="2020-02-06 01:00:00 +0000"),
        _row("a", asin="X", crawl_timestamp="2020-02-07 01:00:00 +0000"),
        _row("c", asin="X", crawl_timestamp="2020-02-07 01:00:00 +0000"),
        _row("d", asin="Y"),
    ]
    kept, id_map = canonicalize(rows)
    assert [r["uniq_id"] for r in kept] == ["a", "d"]
    assert id_map == {"a": "a", "b": "a", "c": "a", "d": "d"}
    assert len(rows) == 4


def test_pipeline_leaves_missing_values_missing():
    products, report = build_products([_row("1")])
    p = products[0]
    assert p.brand is None and p.price is None and p.log_price is None
    assert p.weight_g is None and p.category is None and p.image_url is None
    assert p.review_state == "unknown"
    assert report.category_missing == 1


def test_pipeline_does_not_change_raw_rows_and_recovery_is_off_by_default():
    rows = [
        _row("1", brand="Puma", weight="999999999", sales_price=500.0, product_name="Puma Tee"),
        _row("2", brand=None, weight="250 g", product_name="Puma Cap"),
    ]
    snapshot = [dict(r) for r in rows]
    products, _ = build_products(rows)
    assert rows == snapshot
    assert products[1].brand is None
    assert products[1].weight_g == 250.0
    recovered, report = build_products(rows, recover_brands=True)
    assert recovered[1].brand == "Puma" and report.brand_recovered == 1


def test_pipeline_reports_rejected_weights():
    rows = [_row("1", weight="136 Kilograms"), _row("2", weight="oddly worded"), _row("3")]
    _, report = build_products(rows)
    assert report.weight_problems == {"out_of_range": 1, "not_a_weight": 1}


def test_log_price_is_base_ten():
    products, _ = build_products([_row("1", sales_price=100.0)])
    assert products[0].log_price == pytest.approx(math.log10(100.0))
