"""Attribute scores, weighting, missing data and the image slot."""

import numpy as np
import pytest

from product_similarity.scoring import DEFAULT_WEIGHTS, SimilarityEngine


def test_similar_product_ranks_above_unrelated(make_engine, catalogue):
    engine = make_engine(catalogue)
    assert engine.find_similar("a", 2) == ["b", "c"]


def test_default_weights_cover_every_attribute_and_sum_to_one():
    assert round(sum(DEFAULT_WEIGHTS.values()), 6) == 1.0
    assert DEFAULT_WEIGHTS["image"] == 0.0


def test_missing_attributes_are_skipped_not_filled(make_engine, catalogue):
    engine = make_engine(catalogue)
    result = engine.explain("a", "d")
    assert set(result["attributes"]) == {"title"}
    assert {"brand", "price", "category"} <= set(result["skipped"])
    assert result["score"] == pytest.approx(result["attributes"]["title"]["similarity"], abs=1e-6)


def test_shares_add_up_to_the_score(make_engine, catalogue):
    engine = make_engine(catalogue)
    result = engine.explain("a", "b")
    total = sum(part["share"] for part in result["attributes"].values())
    assert total == pytest.approx(result["score"], abs=1e-5)


def test_price_score_falls_with_the_ratio_between_prices(make_engine, make_row):
    rows = [
        make_row("q", "shirt", sales_price=500.0),
        make_row("same", "shirt", sales_price=500.0),
        make_row("tenfold", "shirt", sales_price=5000.0),
        make_row("none", "shirt"),
    ]
    engine = make_engine(rows)
    price = engine.attribute_similarities(engine.resolve("q"))["price"]
    assert price[engine.resolve("same")] == 1.0
    assert price[engine.resolve("tenfold")] == pytest.approx(0.0)
    assert np.isnan(price[engine.resolve("none")])


def test_colour_score_is_overlap_between_colour_sets(make_engine, make_row):
    rows = [
        make_row("q", "Blue Shirt"),
        make_row("half", "Blue White Shirt"),
        make_row("none", "Red Shirt"),
        make_row("unknown", "Plain Shirt"),
    ]
    engine = make_engine(rows)
    colour = engine.attribute_similarities(engine.resolve("q"))["colour"]
    assert colour[engine.resolve("half")] == pytest.approx(0.5)
    assert colour[engine.resolve("none")] == 0.0
    assert np.isnan(colour[engine.resolve("unknown")])


def test_bad_weights_are_rejected(catalogue, make_engine):
    with pytest.raises(ValueError):
        make_engine(catalogue, weights={"colour_of_the_moon": 1.0})
    with pytest.raises(ValueError):
        make_engine(catalogue, weights={"title": -0.1})


def test_engine_needs_products():
    with pytest.raises(ValueError):
        SimilarityEngine([])


def test_image_slot_is_inactive_at_weight_zero(make_engine, catalogue):
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    without = make_engine(catalogue).find_similar("a", 3)
    with_zero = make_engine(catalogue, image_embeddings=vectors).find_similar("a", 3)
    assert with_zero == without


def test_image_slot_changes_ranking_when_weighted(make_engine, catalogue):
    # The saree "c" looks identical to "a"; the other T-shirt "b" does not.
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    engine = make_engine(catalogue, weights={"image": 5.0}, image_embeddings=vectors)
    assert engine.find_similar("a", 1) == ["c"]
    assert "image" in engine.explain("a", "c")["attributes"]


def test_missing_image_is_skipped(make_engine, catalogue):
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [np.nan, np.nan]])
    engine = make_engine(catalogue, weights={"image": 1.0}, image_embeddings=vectors)
    result = engine.explain("a", "d")
    assert "image" in result["skipped"]
    assert np.isfinite(result["score"])


def test_image_embeddings_need_one_row_per_product(make_engine, catalogue):
    with pytest.raises(ValueError):
        make_engine(catalogue, image_embeddings=np.ones((2, 3)))
