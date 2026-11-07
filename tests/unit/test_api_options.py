"""The mode and weights options, the input checks and the error shape, with a small catalogue."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from product_similarity.api import create_app
from product_similarity.data import canonicalize
from product_similarity.hybrid import FinalVectors, HybridEngine
from product_similarity.preprocessing import build_products

IDS = ["a" * 32, "b" * 32, "c" * 32, "d" * 32, "e" * 32, "f" * 32]
A, B, C = IDS[0], IDS[1], IDS[2]
UNKNOWN = "9" * 32


@pytest.fixture
def hybrid(make_row):
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
    return HybridEngine(products, id_map, vectors, "exact", candidates=50)


@pytest.fixture
def client(hybrid):
    with TestClient(create_app(lambda: hybrid)) as c:
        yield c


def ask(client, product_id=A, num_similar=3, **extra):
    return client.get(
        "/find_similar_products",
        params={"product_id": product_id, "num_similar": num_similar, **extra},
    )


# ---- input checks and the shape of every error ----


@pytest.mark.parametrize(
    "params, text",
    [
        ({"num_similar": 3}, "product_id is required"),
        ({"product_id": A}, "num_similar is required"),
        ({"product_id": "abc", "num_similar": 3}, "32 lowercase hexadecimal"),
        ({"product_id": A, "num_similar": 0}, "from 1 to"),
        ({"product_id": A, "num_similar": -2}, "from 1 to"),
        ({"product_id": A, "num_similar": "many"}, "from 1 to"),
        ({"product_id": A, "num_similar": 1.5}, "from 1 to"),
        ({"product_id": A, "num_similar": 10_000}, "from 1 to 1000"),
    ],
)
def test_bad_input_gives_422_with_a_sentence(client, params, text):
    response = client.get("/find_similar_products", params=params)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_input" and text in body["message"]
    assert body["problems"]


def test_the_largest_allowed_number_works(client):
    assert ask(client, num_similar=1000).status_code == 200


def test_an_unknown_product_gives_404_with_a_code(client):
    response = ask(client, UNKNOWN)
    assert response.status_code == 404
    assert response.json()["error"] == "product_not_found"


def test_an_unknown_address_gives_a_json_error(client):
    response = client.get("/nothing_here")
    assert response.status_code == 404 and response.json()["error"] == "not_found"


def test_a_wrong_method_gives_a_json_error(client):
    response = client.post("/find_similar_products")
    assert response.status_code == 405 and response.json()["error"] == "method_not_allowed"


def test_not_ready_gives_503_with_a_code():
    def broken():
        raise RuntimeError("no data")

    with TestClient(create_app(broken)) as client:
        response = ask(client)
    assert response.status_code == 503 and response.json()["error"] == "not_ready"


# ---- mode ----


def test_the_mode_that_ran_is_reported(client):
    response = ask(client)
    assert response.headers["X-Search-Mode"] == "exact"
    assert ask(client, mode="exact").headers["X-Search-Mode"] == "exact"


def test_an_unknown_mode_gives_422(client):
    response = ask(client, mode="turbo")
    assert response.status_code == 422 and response.json()["error"] == "invalid_mode"
    assert "exact" in response.json()["message"]


def test_the_fast_mode_gives_the_same_answer_when_every_product_is_a_candidate(client):
    pytest.importorskip("faiss")
    assert ask(client, mode="fast").json() == ask(client, mode="exact").json()
    assert ask(client, mode="fast").headers["X-Search-Mode"] == "fast"


def test_the_fast_mode_without_a_vector_weight_runs_the_exact_mode_and_says_so(client):
    pytest.importorskip("faiss")
    response = client.get(
        "/find_similar_products/details",
        params={"product_id": A, "num_similar": 3, "mode": "fast", "weights": "brand:1"},
    )
    body = response.json()
    assert body["mode"] == "exact" and "mode_note" in body


def test_the_details_say_which_mode_and_weights_ran(client):
    body = client.get(
        "/find_similar_products/details", params={"product_id": A, "num_similar": 3}
    ).json()
    assert body["mode"] == "exact" and body["weights"] == "tuned"
    assert "mode_note" not in body


# ---- weights ----


def test_bad_weights_give_422_with_the_reason(client):
    response = ask(client, weights="image:2")
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_weights"
    assert "from 0 to 1" in response.json()["message"]


def test_the_weights_change_the_answer(client):
    picture_only = ask(client, weights="image:1").json()
    title_only = ask(client, weights="title:1").json()
    assert picture_only != title_only
    assert len(picture_only) == len(title_only) == 3


def test_the_tuned_weights_given_explicitly_give_the_default_answer(client):
    tuned = client.get("/weights").json()["tuned_weights"]["simple"]
    text = ",".join(f"{name}:{value}" for name, value in tuned.items())
    assert ask(client, weights=text).json() == ask(client).json()


def test_the_details_show_the_weights_that_were_used(client):
    body = client.get(
        "/find_similar_products/details",
        params={"product_id": A, "num_similar": 3, "weights": "image:1"},
    ).json()
    assert body["weights"]["image"] == 1.0 and body["weights"]["title"] == 0.0


# ---- the cache question includes the mode and the weights ----


def test_other_weights_are_another_question_for_the_cache(client):
    assert ask(client, weights="image:1").headers["X-Cache"] == "miss"
    assert ask(client, weights="image:1").headers["X-Cache"] == "hit"
    assert ask(client, weights="title:1").headers["X-Cache"] == "miss"
    assert ask(client).headers["X-Cache"] == "miss"  # the tuned weights are their own question


def test_the_same_weights_written_differently_share_an_entry(client):
    ask(client, weights="image:0.5,title:0.5")
    assert ask(client, weights="title:0.5, image:0.5").headers["X-Cache"] == "hit"


def test_another_mode_is_another_question_for_the_cache(client):
    pytest.importorskip("faiss")
    assert ask(client, mode="exact").headers["X-Cache"] == "miss"
    assert ask(client, mode="fast").headers["X-Cache"] == "miss"
    assert ask(client, mode="fast").headers["X-Cache"] == "hit"


# ---- the endpoint for the page ----


def test_the_weights_endpoint_lists_modes_limits_and_tuned_weights(client):
    body = client.get("/weights").json()
    assert body["modes"] == ["exact", "fast"] and body["default_mode"] == "exact"
    assert body["max_results"] == 1000 and body["weight_range"] == {"min": 0, "max": 1}
    assert set(body["tuned_weights"]) == {"simple", "advanced"}
    assert len(body["tuned_weights"]["simple"]) == 4
    assert len(body["tuned_weights"]["advanced"]) == 10


def test_ready_lists_the_modes_and_whether_the_fast_mode_is_ready(client):
    body = client.get("/ready").json()
    assert body["modes"] == ["exact", "fast"] and body["fast_ready"] is False


def test_the_fast_mode_is_ready_after_it_was_prepared(hybrid):
    pytest.importorskip("faiss")
    hybrid.prepare_fast()
    with TestClient(create_app(lambda: hybrid)) as client:
        assert client.get("/ready").json()["fast_ready"] is True
