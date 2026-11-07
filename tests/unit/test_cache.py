"""The in-process cache: saved answers, size limit, counts and many threads."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from product_similarity.cache import SearchCache


def test_a_saved_answer_comes_back():
    cache = SearchCache(3)
    cache.put(("a", 5), ["x", "y"])
    assert cache.get(("a", 5)) == ["x", "y"]


def test_an_unknown_question_gives_none():
    assert SearchCache(3).get(("a", 5)) is None


def test_different_questions_have_different_entries():
    cache = SearchCache(3)
    cache.put(("a", 5), "five")
    cache.put(("a", 10), "ten")
    assert cache.get(("a", 5)) == "five" and cache.get(("a", 10)) == "ten"


def test_the_least_recently_used_answer_is_dropped_when_full():
    cache = SearchCache(2)
    cache.put("one", 1)
    cache.put("two", 2)
    cache.get("one")  # "two" is now the least recently used
    cache.put("three", 3)
    assert cache.get("two") is None
    assert cache.get("one") == 1 and cache.get("three") == 3


def test_putting_a_key_again_replaces_its_answer_without_growing():
    cache = SearchCache(2)
    cache.put("a", 1)
    cache.put("a", 2)
    assert cache.get("a") == 2 and cache.stats()["entries"] == 1


def test_a_size_of_zero_turns_the_cache_off():
    cache = SearchCache(0)
    cache.put("a", 1)
    assert cache.get("a") is None
    assert cache.stats()["enabled"] is False and cache.stats()["entries"] == 0


def test_a_negative_size_is_refused():
    with pytest.raises(ValueError):
        SearchCache(-1)


def test_hits_misses_and_the_hit_rate_are_counted():
    cache = SearchCache(5)
    cache.get("a")  # miss
    cache.put("a", 1)
    cache.get("a")  # hit
    cache.get("a")  # hit
    cache.get("b")  # miss
    stats = cache.stats()
    assert (stats["hits"], stats["misses"], stats["hit_rate"]) == (2, 2, 0.5)


def test_clear_forgets_everything():
    cache = SearchCache(5)
    cache.put("a", 1)
    cache.get("a")
    cache.clear()
    assert cache.stats() == {
        "enabled": True,
        "max_entries": 5,
        "entries": 0,
        "hits": 0,
        "misses": 0,
        "hit_rate": 0.0,
    }


def test_many_threads_do_not_break_the_cache():
    cache = SearchCache(50)

    def work(n):
        for i in range(200):
            key = (n + i) % 80
            if cache.get(key) is None:
                cache.put(key, key * 2)
        return True

    with ThreadPoolExecutor(8) as pool:
        assert all(pool.map(work, range(8)))
    stats = cache.stats()
    assert stats["entries"] <= 50
    assert stats["hits"] + stats["misses"] == 8 * 200
