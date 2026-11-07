"""The weights a user can set: reading, checking and the cache form."""

import pytest

from product_similarity.hybrid import FINAL_MIX, expand_four
from product_similarity.weights import (
    ADVANCED_NAMES,
    SIMPLE_NAMES,
    WeightsError,
    parse_weights,
    tuned_weights,
    weights_key,
)


def test_the_simple_way_fills_all_ten_parts():
    weights = parse_weights("title:0.2,text_vectors:0.3,image:0.4,attributes:0.1")
    assert set(weights) == set(ADVANCED_NAMES)
    assert weights["image"] == 0.4 and weights["title"] == 0.2
    assert sum(weights[p] for p in ("category", "brand", "price", "audience")) > 0


def test_the_seven_attributes_share_the_attributes_weight():
    weights = parse_weights("attributes:0.7,image:0.3")
    assert sum(
        weights[p] for p in ("category", "brand", "audience", "price", "colour", "weight", "rating")
    ) == pytest.approx(0.7)


def test_the_advanced_way_names_the_parts_and_leaves_the_rest_at_zero():
    weights = parse_weights("brand:1,price:0.5")
    assert weights["brand"] == 1.0 and weights["price"] == 0.5
    assert weights["image"] == 0.0 and weights["title"] == 0.0


def test_one_weight_of_1_and_the_rest_left_out_is_allowed():
    weights = parse_weights("image:1")
    assert weights["image"] == 1.0 and sum(weights.values()) == 1.0


def test_the_weights_do_not_have_to_add_up_to_one():
    assert parse_weights("title:1,image:1")["image"] == 1.0


def test_spaces_are_allowed():
    assert parse_weights(" image : 0.5 , title : 0.5 ")["title"] == 0.5


@pytest.mark.parametrize(
    "text, problem",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("image", "name:value"),
        ("image:", "name:value"),
        (":0.5", "name:value"),
        ("colour:red", "not a number"),
        ("image:abc", "not a number"),
        ("image:-0.1", "from 0 to 1"),
        ("image:1.1", "from 0 to 1"),
        ("image:nan", "from 0 to 1"),
        ("speed:0.5", "unknown weight"),
        ("image:0.5,image:0.2", "twice"),
        ("image:0,title:0", "above 0"),
        ("attributes:0.5,brand:0.5", "do not mix"),
    ],
)
def test_bad_weights_are_refused_with_a_sentence(text, problem):
    with pytest.raises(WeightsError, match=problem):
        parse_weights(text)


def test_the_cache_form_is_the_same_for_the_same_weights():
    a = parse_weights("image:0.5,title:0.5")
    b = parse_weights("title:0.5, image:0.5")
    assert weights_key(a) == weights_key(b)
    assert weights_key(a) != weights_key(parse_weights("image:0.6,title:0.4"))
    assert weights_key(None) is None


def test_the_tuned_weights_come_in_both_ways():
    tuned = tuned_weights()
    assert set(tuned["simple"]) == set(SIMPLE_NAMES)
    assert set(tuned["advanced"]) == set(ADVANCED_NAMES)
    assert sum(tuned["simple"].values()) == pytest.approx(sum(FINAL_MIX.values()), abs=1e-4)
    assert tuned["advanced"]["image"] == pytest.approx(expand_four(FINAL_MIX)["image"], abs=1e-6)
