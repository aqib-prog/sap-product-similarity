"""The final search: exact and fast modes, one copy of each item, hidden products."""

import json
from pathlib import Path

import numpy as np
import pytest

from product_similarity.data import canonicalize
from product_similarity.hybrid import (
    ATTRIBUTES,
    FINAL_MIX,
    FinalVectors,
    HybridEngine,
    expand_four,
)
from product_similarity.preprocessing import build_products
from product_similarity.scoring import DEFAULT_WEIGHTS

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def small(make_row):
    """Six products with made-up vectors. Products 'a' and 'a2' share a picture."""
    shirt = {"sales_rank_in_child_category": {"MensT_Shirts": "#1"}, "browsenode": "N1"}
    rows = [
        make_row(uid, title, brand="Acme", sales_price=500.0 + k, rating=4.0, **shirt)
        for k, (uid, title) in enumerate(
            [
                ("a", "Blue Cotton Mens T-Shirt"),
                ("a2", "Blue Cotton Mens T-Shirt Size L"),
                ("b", "Blue Cotton Mens T-Shirt Round Neck"),
                ("c", "Green Cotton Mens T-Shirt"),
                ("d", "Cotton Mens Polo Shirt"),
                ("e", "Mens Vest"),
            ]
        )
    ]
    kept, id_map = canonicalize(rows)
    products, _ = build_products(kept)
    rng = np.random.default_rng(1)
    n = len(products)
    picture = rng.normal(size=(n, 6)).astype(np.float32)
    picture[1] = picture[0]  # a2 has the same picture as a
    picture /= np.linalg.norm(picture, axis=1, keepdims=True)
    sentence = rng.normal(size=(n, 5)).astype(np.float32)
    sentence /= np.linalg.norm(sentence, axis=1, keepdims=True)
    links = [f"link-{k}" for k in range(n)]
    links[1] = links[0]
    vectors = FinalVectors(picture, sentence, {"image": 0.0, "text_vectors": 0.0}, picture, links)
    return products, id_map, vectors


def test_final_mix_is_the_tuned_mix():
    saved = json.loads((ROOT / "results" / "hybrid_tuning.json").read_text(encoding="utf-8"))
    assert saved["four_weights"]["weights"] == FINAL_MIX


def test_expand_four_keeps_the_total_weight():
    weights = expand_four(FINAL_MIX)
    assert sum(weights.values()) == pytest.approx(sum(FINAL_MIX.values()))
    assert set(weights) == set(DEFAULT_WEIGHTS)
    ratios = {k: weights[k] / weights["category"] for k in ATTRIBUTES}
    assert ratios["brand"] == pytest.approx(DEFAULT_WEIGHTS["brand"] / DEFAULT_WEIGHTS["category"])


def test_exact_mode_returns_one_copy_of_each_item(small):
    products, id_map, vectors = small
    engine = HybridEngine(products, id_map, vectors, "exact")
    found = engine.find_similar("b", 5)
    assert "b" not in found
    assert not {"a", "a2"} <= set(found)  # the two copies are shown once
    assert len(found) == len(set(found)) == 4  # 6 products, minus itself, minus one copy


def test_a_copy_of_the_query_is_not_returned(small):
    products, id_map, vectors = small
    engine = HybridEngine(products, id_map, vectors, "exact")
    assert "a2" not in engine.find_similar("a", 5)


def test_hidden_products_are_never_returned(small):
    products, id_map, vectors = small
    engine = HybridEngine(products, id_map, vectors, "exact", hidden_ids={"c", "d"})
    assert not {"c", "d"} & set(engine.find_similar("b", 5))


def test_scores_come_best_first(small):
    products, id_map, vectors = small
    scored = HybridEngine(products, id_map, vectors, "exact").find_similar_with_scores("b", 4)
    values = [s for _, s in scored]
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "3"])
def test_num_similar_must_be_a_whole_number(small, bad):
    products, id_map, vectors = small
    with pytest.raises(ValueError):
        HybridEngine(products, id_map, vectors, "exact").find_similar("a", bad)


def test_unknown_mode_is_refused(small):
    products, id_map, vectors = small
    with pytest.raises(ValueError):
        HybridEngine(products, id_map, vectors, "turbo")


def test_fast_mode_with_every_product_as_candidate_equals_exact(small):
    pytest.importorskip("faiss")
    products, id_map, vectors = small
    exact = HybridEngine(products, id_map, vectors, "exact")
    fast = HybridEngine(products, id_map, vectors, "fast", candidates=50)
    for uid in ("a", "b", "c", "d", "e"):
        assert fast.find_similar(uid, 4) == exact.find_similar(uid, 4)


def test_fast_mode_finds_a_product_that_has_no_picture(small):
    """A missing picture is skipped like in the hybrid: it is found by its sentence vector."""
    pytest.importorskip("faiss")
    products, id_map, vectors = small
    picture = vectors.picture.copy()
    picture[3] = np.nan  # product "c" has no picture
    vectors = FinalVectors(
        picture, vectors.sentence, vectors.floors, vectors.copies_matrix, vectors.copies_links
    )
    exact = HybridEngine(products, id_map, vectors, "exact")
    fast = HybridEngine(products, id_map, vectors, "fast", candidates=50)
    assert "c" in exact.find_similar("b", 5)
    assert fast.find_similar("b", 5) == exact.find_similar("b", 5)


def test_a_query_without_a_picture_still_gets_results_in_fast_mode(small):
    pytest.importorskip("faiss")
    products, id_map, vectors = small
    picture = vectors.picture.copy()
    picture[2] = np.nan  # product "b" has no picture
    vectors = FinalVectors(
        picture, vectors.sentence, vectors.floors, vectors.copies_matrix, vectors.copies_links
    )
    fast = HybridEngine(products, id_map, vectors, "fast", candidates=50)
    exact = HybridEngine(products, id_map, vectors, "exact")
    assert fast.find_similar("b", 4) == exact.find_similar("b", 4)


def test_hidden_products_are_not_in_the_indexes(small):
    pytest.importorskip("faiss")
    products, id_map, vectors = small
    fast = HybridEngine(products, id_map, vectors, "fast", candidates=50, hidden_ids={"d"})
    assert "d" not in fast.find_similar("b", 5)


def test_the_mode_can_be_chosen_for_each_search(small):
    pytest.importorskip("faiss")
    products, id_map, vectors = small
    engine = HybridEngine(products, id_map, vectors, "exact", candidates=50)
    exact = engine.find_similar("b", 4)
    assert engine.find_similar("b", 4, mode="fast") == exact  # the indexes are built on first use
    assert engine.find_similar("b", 4, mode="exact") == exact
    with pytest.raises(ValueError):
        engine.find_similar("b", 4, mode="turbo")


def test_weights_for_one_search_change_the_ranking_and_not_the_engine(small):
    products, id_map, vectors = small
    engine = HybridEngine(products, id_map, vectors, "exact")
    default = engine.find_similar("b", 4)
    picture_only = {"image": 1.0}
    other = engine.find_similar("b", 4, weights=picture_only)
    assert engine.find_similar("b", 4) == default  # the engine's own weights are unchanged
    assert len(other) == len(default) == 4


def test_search_says_which_mode_ran(small):
    products, id_map, vectors = small
    engine = HybridEngine(products, id_map, vectors, "exact")
    assert engine.search("b", 3)[1] == "exact"


def test_the_fast_mode_without_a_vector_weight_falls_back_to_the_exact_mode(small):
    pytest.importorskip("faiss")
    products, id_map, vectors = small
    engine = HybridEngine(products, id_map, vectors, "exact", candidates=50)
    ranked, used = engine.search("b", 3, mode="fast", weights={"brand": 1.0})
    assert used == "exact"
    assert ranked == engine.search("b", 3, mode="exact", weights={"brand": 1.0})[0]
    assert engine.search("b", 3, mode="fast", weights={"image": 1.0})[1] == "fast"


def test_weights_with_an_unknown_part_or_a_negative_value_are_refused(small):
    products, id_map, vectors = small
    engine = HybridEngine(products, id_map, vectors, "exact")
    with pytest.raises(ValueError):
        engine.search("b", 3, weights={"speed": 1.0})
    with pytest.raises(ValueError):
        engine.search("b", 3, weights={"image": -1.0})
