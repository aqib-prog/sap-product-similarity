"""The /metrics page: the counters follow what the service really does."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from product_similarity.api import create_app
from product_similarity.data import canonicalize
from product_similarity.hybrid import FinalVectors, HybridEngine
from product_similarity.preprocessing import build_products

IDS = ["a" * 32, "b" * 32, "c" * 32, "d" * 32, "e" * 32, "f" * 32]
A = IDS[0]


@pytest.fixture
def client(make_row):
    shirt = {"sales_rank_in_child_category": {"MensT_Shirts": "#1"}, "browsenode": "N1"}
    titles = [
        "Blue Cotton Mens T-Shirt",
        "Blue Cotton Mens T-Shirt Round Neck",
        "Green Cotton Mens T-Shirt",
        "Cotton Mens Polo Shirt",
        "Mens Vest",
        "Red Silk Womens Saree",
    ]
    rows = [
        make_row(uid, title, brand="Acme", sales_price=500.0 + k, rating=4.0, **shirt)
        for k, (uid, title) in enumerate(zip(IDS, titles))
    ]
    kept, id_map = canonicalize(rows)
    products, _ = build_products(kept)
    rng = np.random.default_rng(5)
    n = len(products)
    picture = rng.normal(size=(n, 6)).astype(np.float32)
    picture /= np.linalg.norm(picture, axis=1, keepdims=True)
    sentence = rng.normal(size=(n, 5)).astype(np.float32)
    sentence /= np.linalg.norm(sentence, axis=1, keepdims=True)
    links = [f"link-{k}" for k in range(n)]
    vectors = FinalVectors(picture, sentence, {"image": 0.0, "text_vectors": 0.0}, picture, links)
    engine = HybridEngine(products, id_map, vectors, "exact", candidates=50)
    with TestClient(create_app(lambda: engine)) as c:
        yield c


def value(client, name, **labels):
    """One number from the /metrics page, or 0 when that series does not exist yet."""
    for line in client.get("/metrics").text.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        series, _, number = line.rpartition(" ")
        series_name, _, rest = series.partition("{")
        if series_name != name:
            continue
        found = dict(part.split("=") for part in rest.rstrip("}").split(",") if part)
        if all(found.get(k) == f'"{v}"' for k, v in labels.items()):
            return float(number)
    return 0.0


def ask(client, num_similar=3, **extra):
    return client.get(
        "/find_similar_products",
        params={"product_id": A, "num_similar": num_similar, **extra},
    )


def test_metrics_page_is_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE ps_http_requests_total counter" in response.text


def test_a_miss_then_a_hit_are_counted(client):
    ask(client)
    ask(client)
    assert value(client, "ps_searches_total", mode="exact", cache="miss") == 1
    assert value(client, "ps_searches_total", mode="exact", cache="hit") == 1
    assert (
        value(client, "ps_http_requests_total", endpoint="/find_similar_products", status="200")
        == 2
    )
    assert value(client, "ps_search_seconds_count", mode="exact") == 1  # only the real search
    assert value(client, "ps_cache_entries") == 1


def test_the_mode_that_ran_is_counted(client):
    ask(client, mode="fast")
    assert value(client, "ps_searches_total", mode="fast", cache="miss") == 1
    ask(client, mode="fast")
    assert value(client, "ps_searches_total", mode="fast", cache="hit") == 1


def test_errors_are_counted_by_code_and_status(client):
    ask(client, num_similar=0)
    ask(client, mode="turbo")
    client.get("/find_similar_products", params={"product_id": "9" * 32, "num_similar": 3})
    assert value(client, "ps_errors_total", error="invalid_input") == 1
    assert value(client, "ps_errors_total", error="invalid_mode") == 1
    assert value(client, "ps_errors_total", error="product_not_found") == 1
    assert (
        value(client, "ps_http_requests_total", endpoint="/find_similar_products", status="422")
        == 2
    )
    assert (
        value(client, "ps_http_requests_total", endpoint="/find_similar_products", status="404")
        == 1
    )


def test_reading_metrics_is_not_counted_and_ready_is_one(client):
    client.get("/metrics")
    client.get("/metrics")
    assert value(client, "ps_http_requests_total", endpoint="/metrics", status="200") == 0
    assert value(client, "ps_ready") == 1
