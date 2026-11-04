"""The public function: lookups, limits and repeatable results."""

import pytest

from product_similarity import service
from product_similarity.scoring import ProductNotFoundError


@pytest.fixture
def api(monkeypatch, make_engine, make_row, catalogue):
    """Point the public function at a small engine that also has a repeated product."""
    repeat = make_row(
        "a_old",
        "Blue Cotton Mens T-Shirt",
        asin="ASINa",
        brand="Acme",
        crawl_timestamp="2020-02-06 05:00:00 +0000",
    )
    twin_1 = make_row("t1", "Green Hat")
    twin_2 = make_row("t2", "Green Hat")
    engine = make_engine([*catalogue, repeat, twin_1, twin_2])
    service.default_engine.cache_clear()
    monkeypatch.setattr(service, "build_engine", lambda: engine)
    yield engine
    service.default_engine.cache_clear()


def test_returns_a_list_of_ids(api):
    result = service.find_similar_products("a", 2)
    assert isinstance(result, list) and len(result) == 2
    assert all(isinstance(i, str) for i in result)


def test_query_and_its_repeat_copy_are_never_returned(api):
    result = service.find_similar_products("a", 100)
    assert "a" not in result and "a_old" not in result


def test_an_old_repeated_id_gives_the_kept_products_results(api):
    assert service.find_similar_products("a_old", 3) == service.find_similar_products("a", 3)


def test_unknown_id_raises(api):
    with pytest.raises(ProductNotFoundError):
        service.find_similar_products("does-not-exist", 3)


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "3", None])
def test_num_similar_must_be_a_positive_whole_number(api, bad):
    with pytest.raises(ValueError):
        service.find_similar_products("a", bad)


def test_asking_for_more_than_exist_returns_all_others(api):
    result = service.find_similar_products("a", 10_000)
    assert len(result) == len(api.products) - 1
    assert len(set(result)) == len(result)


def test_results_are_repeatable(api):
    assert service.find_similar_products("c", 5) == service.find_similar_products("c", 5)


def test_identical_products_are_ordered_by_id(api):
    ranked = api.find_similar_with_scores("a", 100)
    scores = dict(ranked)
    assert scores["t1"] == scores["t2"]
    ids = [uid for uid, _ in ranked]
    assert ids.index("t1") < ids.index("t2")
