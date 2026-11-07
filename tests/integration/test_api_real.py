"""A few real requests against the real dataset (about 3 seconds to start)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from product_similarity.api import create_app
from product_similarity.service import build_engine, build_final_engine

ARCHIVE = Path(__file__).resolve().parents[2] / "data" / "archive.zip"
pytestmark = pytest.mark.skipif(not ARCHIVE.exists(), reason="dataset archive not present")

FIRST_ID = "26d41bdc1495de290bc8e6062d927729"


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(lambda: build_engine(archive=ARCHIVE))) as c:
        yield c


def test_real_request_returns_five_ids(client):
    response = client.get(
        "/find_similar_products", params={"product_id": FIRST_ID, "num_similar": 5}
    )
    assert response.status_code == 200
    ids = response.json()
    assert len(ids) == len(set(ids)) == 5
    assert FIRST_ID not in ids


def test_service_is_ready_with_all_products(client):
    assert client.get("/ready").json() == {
        "status": "ready",
        "products": 29529,
        "search": "baseline",
        "modes": [],
        "fast_ready": False,
    }


def test_unknown_real_id_gives_404(client):
    response = client.get(
        "/find_similar_products", params={"product_id": "0" * 32, "num_similar": 5}
    )
    assert response.status_code == 404


def test_real_details_view_has_titles(client):
    response = client.get(
        "/find_similar_products/details", params={"product_id": FIRST_ID, "num_similar": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"]["id"] == FIRST_ID
    assert len(body["results"]) == 3 and all(r["title"] for r in body["results"])


VECTORS = ARCHIVE.parent.parent / "artifacts" / "vectors"


@pytest.mark.skipif(not VECTORS.exists(), reason="the shipped vector files are not present")
def test_the_final_search_runs_with_the_shipped_vectors(monkeypatch):
    monkeypatch.setenv("PS_REQUIRE_VECTORS", "1")
    monkeypatch.setenv("PS_VECTORS_DIR", str(VECTORS))
    with TestClient(create_app(lambda: build_final_engine("exact"))) as final:
        assert final.get("/ready").json()["search"] == "hybrid (exact)"
        response = final.get(
            "/find_similar_products", params={"product_id": FIRST_ID, "num_similar": 5}
        )
        ids = response.json()
        assert response.status_code == 200
        assert len(ids) == len(set(ids)) == 5 and FIRST_ID not in ids


@pytest.mark.skipif(not VECTORS.exists(), reason="the shipped vector files are not present")
def test_the_mode_and_the_weights_work_on_the_real_data(monkeypatch):
    pytest.importorskip("faiss")
    monkeypatch.setenv("PS_REQUIRE_VECTORS", "1")
    monkeypatch.setenv("PS_VECTORS_DIR", str(VECTORS))
    with TestClient(create_app(lambda: build_final_engine("exact"))) as final:
        params = {"product_id": FIRST_ID, "num_similar": 5}
        fast = final.get("/find_similar_products", params={**params, "mode": "fast"})
        assert fast.status_code == 200 and fast.headers["X-Search-Mode"] == "fast"
        assert len(set(fast.json())) == 5
        default = final.get("/find_similar_products", params=params).json()
        picture = final.get("/find_similar_products", params={**params, "weights": "image:1"})
        title = final.get("/find_similar_products", params={**params, "weights": "title:1"})
        assert picture.status_code == title.status_code == 200
        assert picture.json() != title.json() and len(default) == 5
