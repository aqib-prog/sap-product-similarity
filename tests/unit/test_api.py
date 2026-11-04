"""Every response the web service can give, using a small catalogue."""

import pytest
from fastapi.testclient import TestClient

from product_similarity.api import create_app

A, B, C, D = ("a" * 32, "b" * 32, "c" * 32, "d" * 32)
OLD = "e" * 32
UNKNOWN = "f" * 32


@pytest.fixture
def engine(make_row, make_engine):
    shirt = {"sales_rank_in_child_category": {"MensT_Shirts": "#1"}, "browsenode": "N1"}
    saree = {"sales_rank_in_child_category": {"WomensSarees": "#1"}, "browsenode": "N2"}
    rows = [
        make_row(
            A, "Blue Cotton Mens T-Shirt", asin="X1", brand="Acme", sales_price=500.0, **shirt
        ),
        make_row(
            B, "Blue Cotton Mens T-Shirt Round Neck", brand="Acme", sales_price=520.0, **shirt
        ),
        make_row(C, "Red Silk Womens Saree", brand="Zed", sales_price=3000.0, **saree),
        make_row(D, "Mystery item"),
        make_row(
            OLD, "Blue Cotton Mens T-Shirt", asin="X1", crawl_timestamp="2020-02-06 05:00:00 +0000"
        ),
    ]
    return make_engine(rows)


@pytest.fixture
def client(engine):
    with TestClient(create_app(lambda: engine)) as c:
        yield c


def get(client, product_id=A, num_similar=2):
    return client.get(
        "/find_similar_products", params={"product_id": product_id, "num_similar": num_similar}
    )


def test_returns_a_plain_json_list_of_ids(client):
    response = get(client)
    assert response.status_code == 200
    assert response.json() == [B, C]


def test_never_returns_the_query_or_its_repeat(client):
    ids = get(client, num_similar=100).json()
    assert A not in ids and OLD not in ids


def test_an_old_repeated_id_works(client):
    assert get(client, OLD).json() == get(client, A).json()


def test_asking_for_more_than_exist_returns_all_others(client):
    assert len(get(client, num_similar=10_000).json()) == 3


def test_unknown_product_gives_404(client):
    response = get(client, UNKNOWN)
    assert response.status_code == 404
    assert response.json() == {"detail": "Product not found"}


@pytest.mark.parametrize("bad_id", ["abc", "A" * 32, "g" * 32, "a" * 31, "a" * 33, ""])
def test_malformed_id_gives_422(client, bad_id):
    assert get(client, bad_id).status_code == 422


@pytest.mark.parametrize("bad_number", [0, -1, "abc", 2.5, ""])
def test_bad_num_similar_gives_422(client, bad_number):
    assert get(client, num_similar=bad_number).status_code == 422


def test_missing_parameters_give_422(client):
    assert client.get("/find_similar_products").status_code == 422
    assert client.get("/find_similar_products", params={"product_id": A}).status_code == 422
    assert client.get("/find_similar_products", params={"num_similar": 3}).status_code == 422


def test_live_and_ready(client):
    assert client.get("/live").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["products"] == 4


def test_service_reports_not_ready_if_data_cannot_load():
    def broken():
        raise RuntimeError("no data")

    with TestClient(create_app(broken)) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/ready").status_code == 503
        assert get(client).status_code == 503


def test_unexpected_errors_are_hidden_from_the_caller(engine):
    def explode(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    engine.find_similar = explode
    with TestClient(create_app(lambda: engine), raise_server_exceptions=False) as client:
        response = get(client)
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "secret" not in response.text


def test_details_view_matches_the_plain_list_and_adds_readable_fields(client):
    plain = get(client).json()
    response = client.get(
        "/find_similar_products/details", params={"product_id": A, "num_similar": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == {"product_id": A, "num_similar": 2}
    assert body["query"]["id"] == A
    assert isinstance(body["response_time_ms"], float) and body["response_time_ms"] >= 0
    assert [r["id"] for r in body["results"]] == plain
    first = body["results"][0]
    assert {"title", "brand", "category", "price", "image_url", "score"} <= set(first)
    assert first["title"] == "Blue Cotton Mens T-Shirt Round Neck"


def test_details_view_has_the_same_errors(client):
    params = {"num_similar": 2}
    assert (
        client.get(
            "/find_similar_products/details", params={"product_id": UNKNOWN, **params}
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/find_similar_products/details", params={"product_id": "abc", **params}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/find_similar_products/details", params={"product_id": A, "num_similar": 0}
        ).status_code
        == 422
    )
