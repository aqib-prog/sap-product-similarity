"""Checks on the real dataset. Slower than the unit tests (about 2 seconds to build)."""

import random
import statistics
import time
from pathlib import Path

import pytest

from product_similarity.service import build_engine

ARCHIVE = Path(__file__).resolve().parents[2] / "data" / "archive.zip"
pytestmark = pytest.mark.skipif(not ARCHIVE.exists(), reason="dataset archive not present")


@pytest.fixture(scope="module")
def engine():
    return build_engine(archive=ARCHIVE)


def test_every_original_id_resolves(engine):
    assert len(engine.id_map) == 30_000
    assert all(engine.resolve(uid) is not None for uid in engine.id_map)


def test_queries_return_valid_unique_results_without_the_query(engine):
    rng = random.Random(1)
    known = set(engine.ids)
    for uid in rng.sample(sorted(engine.id_map), 25):
        result = engine.find_similar(uid, 10)
        kept = engine.id_map[uid]
        assert len(result) == len(set(result)) == 10
        assert kept not in result and uid not in result
        assert set(result) <= known


def test_a_query_is_fast_enough(engine):
    rng = random.Random(2)
    times = []
    for uid in rng.sample(sorted(engine.id_map), 20):
        start = time.perf_counter()
        engine.find_similar(uid, 10)
        times.append(time.perf_counter() - start)
    assert statistics.median(times) < 0.25
