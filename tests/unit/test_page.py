"""The page at /: it is served without loading the data, and it has the parts people use."""

import re

import pytest
from fastapi.testclient import TestClient

from product_similarity.api import create_app, page_html


@pytest.fixture
def client():
    def not_ready():
        raise RuntimeError("no data in this test")

    with TestClient(create_app(not_ready)) as c:
        yield c


def test_the_page_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Product similarity" in response.text


def test_the_page_has_the_parts_people_use(client):
    html = client.get("/").text
    for part in (
        'id="product-id"',  # the product box
        'id="random"',  # random product
        'id="mode"',  # exact / fast
        'id="preset"',  # best / picture / title / custom
        'id="sliders"',  # the weight sliders
        'id="count"',  # number of results
        'id="time"',  # the response time
        'id="answer-pill"',  # new search or from the cache
        'id="results"',  # the result cards
    ):
        assert part in html, part


def test_the_header_links_to_the_service_pages_and_the_dashboard(client):
    html = client.get("/").text
    for link in (
        'href="/docs"',
        'href="/metrics"',
        'href="/ready"',
        'href="http://localhost:3000"',
    ):
        assert link in html, link


def test_the_page_calls_only_this_service(client):
    html = client.get("/").text
    calls = set(re.findall(r'fetch\("(/[a-z_/]*)', html))
    calls |= set(re.findall(r'"(/find_similar_products/details)\?', html))
    assert calls == {"/find_similar_products/details", "/cache", "/ready", "/weights"}
    # the only outside file is the font, never a script
    assert not re.search(r"<script[^>]+src=", html)


def test_the_page_never_puts_data_into_html(client):
    # results are written with textContent, so a product title cannot add markup
    html = client.get("/").text
    assert ".innerHTML" not in html and "insertAdjacentHTML" not in html


def test_the_ids_for_random_product_are_valid(client):
    html = client.get("/").text
    block = html.split("var RANDOM_IDS = [")[1].split("];")[0]
    ids = re.findall(r'"([^"]+)"', block)
    assert len(ids) == 30 and len(set(ids)) == 30
    assert all(re.fullmatch(r"[0-9a-f]{32}", i) for i in ids)


def test_favicon_is_empty_not_an_error(client):
    assert client.get("/favicon.ico").status_code == 204


def test_the_page_file_is_read_once():
    assert page_html() is page_html()
