"""Measure what the in-process cache saves.

Run from the project folder:  python experiments/measure_cache.py
Part 1: the time of one request when the answer is not in the cache (a miss) and when it is
(a hit), through the real web service (FastAPI test client, final search in exact mode). 200
random products (seed 2020, never a locked product); for each one, the cache is emptied, then
the request is made twice: the first is a miss and the second a hit. Five rounds each, the
fastest kept, the median of the 200 reported.
Part 2: a simulation of traffic where a few products are asked for often. 2,000 requests are
drawn from all products with a popularity that falls off like 1/rank (a Zipf shape, an
assumption: I have no real traffic), for several cache sizes. It reports the hit rate and the
time per request, and the memory of the cache. Results: results/cache_benchmark.json.
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from product_similarity.api import create_app
from product_similarity.service import build_final_engine

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 2020
PRODUCTS = 200
ROUNDS = 5
REQUESTS = 2000
SIZES = (0, 100, 1000, 5000)
NUM_SIMILAR = 10


def request_ms(client: TestClient, product_id: str) -> tuple[float, str]:
    t0 = time.perf_counter()
    response = client.get(
        "/find_similar_products", params={"product_id": product_id, "num_similar": NUM_SIMILAR}
    )
    elapsed = (time.perf_counter() - t0) * 1000
    assert response.status_code == 200
    return elapsed, response.headers["X-Cache"]


def main() -> None:
    quiet_locked = json.loads((ROOT / "evaluation" / "queries.json").read_text(encoding="utf-8"))
    locked = {q["id"] for q in quiet_locked if q["split"] == "final"}
    engine = build_final_engine("exact")
    allowed = [str(uid) for uid in engine.ids if str(uid) not in locked]
    rng = np.random.default_rng(SEED)

    # ---- part 1: miss against hit ----
    chosen = rng.choice(len(allowed), PRODUCTS, replace=False)
    misses, hits = [], []
    with TestClient(create_app(lambda: engine, cache_size=1024)) as client:
        cache = client.app.state.cache
        for index in chosen:
            product_id = allowed[index]
            best_miss, best_hit = np.inf, np.inf
            for _ in range(ROUNDS):
                cache.clear()
                miss_ms, miss_state = request_ms(client, product_id)
                hit_ms, hit_state = request_ms(client, product_id)
                assert (miss_state, hit_state) == ("miss", "hit")
                best_miss, best_hit = min(best_miss, miss_ms), min(best_hit, hit_ms)
            misses.append(best_miss)
            hits.append(best_hit)
    part1 = {
        "products": PRODUCTS,
        "rounds": ROUNDS,
        "miss_median_ms": round(float(np.median(misses)), 3),
        "miss_95th_percentile_ms": round(float(np.percentile(misses, 95)), 3),
        "hit_median_ms": round(float(np.median(hits)), 3),
        "hit_95th_percentile_ms": round(float(np.percentile(hits, 95)), 3),
    }
    part1["times_faster"] = round(part1["miss_median_ms"] / part1["hit_median_ms"], 1)
    print(part1, flush=True)

    # ---- memory of one cached answer ----
    tracemalloc.start()
    memory_cache_client = TestClient(create_app(lambda: engine, cache_size=5000))
    with memory_cache_client as client:
        before = tracemalloc.get_traced_memory()[0]
        for index in rng.choice(len(allowed), 1000, replace=False):
            request_ms(client, allowed[index])
        after = tracemalloc.get_traced_memory()[0]
        entries = client.app.state.cache.stats()["entries"]
    tracemalloc.stop()
    kb_per_answer = round((after - before) / entries / 1024, 2)

    # ---- part 2: traffic where a few products are asked often ----
    ranks = np.arange(1, len(allowed) + 1)
    popularity = 1.0 / ranks
    popularity /= popularity.sum()
    order = rng.permutation(len(allowed))  # which product has which popularity rank
    draws = order[rng.choice(len(allowed), REQUESTS, p=popularity)]
    traffic = []
    for size in SIZES:
        with TestClient(create_app(lambda: engine, cache_size=size)) as client:
            t0 = time.perf_counter()
            for index in draws:
                request_ms(client, allowed[index])
            total_s = time.perf_counter() - t0
            stats = client.app.state.cache.stats()
        traffic.append(
            {
                "cache_size": size,
                "requests": REQUESTS,
                "hit_rate": stats["hit_rate"],
                "average_ms_per_request": round(total_s * 1000 / REQUESTS, 3),
                "different_products_asked": int(len(set(draws.tolist()))),
            }
        )
        print(traffic[-1], flush=True)

    report = {
        "search": engine.search_name,
        "num_similar": NUM_SIMILAR,
        "timing_rule": "200 random products, cache emptied before each pair, 5 rounds, fastest kept, median",
        "miss_against_hit": part1,
        "memory_kb_per_cached_answer": kb_per_answer,
        "traffic_simulation": {
            "popularity": "1/rank (Zipf shape), an assumption, not real traffic",
            "results": traffic,
        },
        "python": sys.version.split()[0],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "cache_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("saved results/cache_benchmark.json")


if __name__ == "__main__":
    main()
