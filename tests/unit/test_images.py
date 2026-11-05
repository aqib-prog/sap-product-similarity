"""Safe downloads, quality checks and sample selection (no real internet is used)."""

import io
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from product_similarity.images import (  # noqa: E402
    ImageFetcher,
    inspect_image,
    mark_duplicates,
    select_image_sample,
    shared_image_urls,
)
from product_similarity.preprocessing import Product  # noqa: E402


def _jpeg(width=400, height=400, colour=(200, 40, 40), noisy=True):
    import random

    image = Image.new("RGB", (width, height), colour)
    if noisy:
        rng = random.Random(1)
        pixels = image.load()
        for _ in range(width * height // 3):
            pixels[rng.randrange(width), rng.randrange(height)] = (
                rng.randrange(256),
                rng.randrange(256),
                rng.randrange(256),
            )
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


class _Handler(BaseHTTPRequestHandler):
    routes = {}

    def do_GET(self):
        code, ctype, body, headers = self.routes.get(self.path, (404, "text/plain", b"no", {}))
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.routes = {
        "/ok.jpg": (200, "image/jpeg", _jpeg(), {}),
        "/html": (200, "text/html", b"<html></html>", {}),
        "/big.jpg": (200, "image/jpeg", b"x" * 5000, {}),
        "/moved": (302, "text/plain", b"", {"Location": "http://elsewhere.test/x.jpg"}),
    }
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def fetcher(tmp_path, **kw):
    return ImageFetcher(
        tmp_path, allowed_hosts={"127.0.0.1"}, allow_insecure=True, min_interval=0, **kw
    )


def test_good_image_is_saved(tmp_path, server):
    result = fetcher(tmp_path).fetch(server + "/ok.jpg")
    assert result.ok and result.n_bytes > 1000 and result.http_status == 200
    assert (tmp_path / result.path.split("/")[-1]).exists()


def test_other_hosts_and_plain_http_are_refused_without_a_request(tmp_path):
    strict = ImageFetcher(tmp_path, min_interval=0)
    assert strict.fetch("https://example.com/a.jpg").error == "host_not_allowed"
    assert strict.fetch("http://images-na.ssl-images-amazon.com/a.jpg").error == "not_https"


@pytest.mark.parametrize(
    "path, error",
    [("/missing.jpg", "http_404"), ("/html", "not_an_image"), ("/moved", "http_302")],
)
def test_bad_responses_are_recorded_not_raised(tmp_path, server, path, error):
    result = fetcher(tmp_path).fetch(server + path, retries=0)
    assert not result.ok and result.error == error


def test_oversized_downloads_are_stopped(tmp_path, server):
    result = fetcher(tmp_path, max_bytes=1000).fetch(server + "/big.jpg", retries=0)
    assert result.error == "too_large" and not result.ok


def test_inspect_good_image(tmp_path):
    path = tmp_path / "a.jpg"
    path.write_bytes(_jpeg())
    info = inspect_image(path)
    assert info["decodes"] and info["width"] == 400 and info["flags"] == []


def test_inspect_flags_problems(tmp_path):
    tiny = tmp_path / "tiny.jpg"
    tiny.write_bytes(_jpeg(50, 50))
    assert "tiny" in inspect_image(tiny)["flags"]
    blank = tmp_path / "blank.jpg"
    blank.write_bytes(_jpeg(400, 400, (255, 255, 255), noisy=False))
    assert "mostly_blank" in inspect_image(blank)["flags"]
    wide = tmp_path / "wide.jpg"
    wide.write_bytes(_jpeg(1200, 300))
    assert "extreme_shape" in inspect_image(wide)["flags"]
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    assert inspect_image(broken) == {"decodes": False, "flags": ["cannot_decode"]}


def test_same_picture_on_different_products_is_flagged():
    rows = [
        {"uniq_id": "a", "hash": "1", "flags": []},
        {"uniq_id": "b", "hash": "1", "flags": []},
        {"uniq_id": "c", "hash": "2", "flags": []},
    ]
    mark_duplicates(rows)
    assert rows[0]["flags"] == rows[1]["flags"] == ["same_picture_as_another_product"]
    assert rows[2]["flags"] == []


def _product(i, **kw):
    base = dict(
        uniq_id=f"{i:032x}",
        asin=f"A{i}",
        title=f"Title {i % 12}",
        brand=None,
        category=f"C{i % 40}",
        category_source="child",
        audience="men",
        price=None,
        log_price=None,
        weight_g=None,
        rating=None,
        page_rating=None,
        review_count=None,
        review_state="unknown",
        title_colours=(),
        listed_colours=(),
        image_url=f"https://images-na.ssl-images-amazon.com/images/I/p{i}.jpg",
    )
    base.update(kw)
    return Product(**base)


def test_shared_urls_need_enough_products():
    urls = {f"p{i}": ["https://x/shared.jpg"] for i in range(6)}
    urls["q"] = ["https://x/alone.jpg"]
    assert set(shared_image_urls(urls, minimum=5)) == {"https://x/shared.jpg"}


def test_sample_excludes_locked_products_and_is_repeatable():
    products = [_product(i) for i in range(600)]
    urls = {p.uniq_id: [p.image_url] for p in products}
    for p in products[:120]:
        urls[p.uniq_id] = [f"https://x/shared{p.asin[1:]}".rsplit("0", 1)[0] + "z.jpg"]
    locked = {p.uniq_id for p in products[::3]}
    first = select_image_sample(products, urls, locked)
    assert first == select_image_sample(products, urls, locked)
    assert not {s["uniq_id"] for s in first} & locked
    assert len({s["uniq_id"] for s in first}) == len(first) <= 100
    assert {s["group"] for s in first} >= {"random", "rare category", "repeated title"}
