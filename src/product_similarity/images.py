"""Safe image download, quality checks and sample selection.

Offline tooling only: the web service never imports this module. Downloads are
limited to an allowed host, use https, have timeouts, a size limit and a slow
request rate, and never follow redirects to another place.
"""

from __future__ import annotations

import hashlib
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from product_similarity.judging import attributes_known
from product_similarity.preprocessing import Product

ALLOWED_HOSTS = frozenset({"images-na.ssl-images-amazon.com"})
USER_AGENT = "product-similarity-research/0.1 (small sample check for a technical exercise)"
MAX_BYTES = 2_000_000
TIMEOUT_SECONDS = 15
MIN_SECONDS_BETWEEN_REQUESTS = 0.25  # at most 4 requests per second, one at a time
SEED = 2020

# Quality thresholds (shown in the report so they can be questioned).
TINY_SIDE = 100
LOW_RESOLUTION_SIDE = 300
BLANK_STD = 10.0
BLANK_WHITE_SHARE = 0.985
BLURRY_LAPLACIAN = 15.0
EXTREME_ASPECT = 2.5
NEAR_WHITE = 245

# (group, how many images) for the sample.
SAMPLE_PLAN = (
    ("random", 40),
    ("rare category", 20),
    ("shared image", 15),
    ("sparse data", 15),
    ("repeated title", 10),
)


# ---- download -------------------------------------------------------------


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401
        return None


@dataclass
class DownloadResult:
    url: str
    ok: bool
    error: str = ""
    http_status: int | None = None
    content_type: str = ""
    n_bytes: int = 0
    seconds: float = 0.0
    attempts: int = 0
    path: str = ""


class ImageFetcher:
    """Downloads images one at a time, slowly, from allowed hosts only."""

    def __init__(
        self,
        folder: Path,
        allowed_hosts: Iterable[str] = ALLOWED_HOSTS,
        allow_insecure: bool = False,
        max_bytes: int = MAX_BYTES,
        timeout: float = TIMEOUT_SECONDS,
        min_interval: float = MIN_SECONDS_BETWEEN_REQUESTS,
    ) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.allowed_hosts = frozenset(allowed_hosts)
        self.allow_insecure = allow_insecure
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_request = 0.0
        self._opener = urllib.request.build_opener(_RefuseRedirects)

    def path_for(self, url: str) -> Path:
        return self.folder / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".jpg")

    def _check_url(self, url: str) -> str:
        parts = urllib.parse.urlparse(url)
        if parts.scheme != "https" and not (self.allow_insecure and parts.scheme == "http"):
            return "not_https"
        if parts.hostname not in self.allowed_hosts:
            return "host_not_allowed"
        return ""

    def _wait(self) -> None:
        gap = time.monotonic() - self._last_request
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last_request = time.monotonic()

    def fetch(self, url: str, retries: int = 1) -> DownloadResult:
        """Download one image. Returns a result, and never raises for a bad download."""
        problem = self._check_url(url)
        if problem:
            return DownloadResult(url, False, problem)
        result = DownloadResult(url, False)
        for attempt in range(1, retries + 2):
            result.attempts = attempt
            self._wait()
            started = time.monotonic()
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with self._opener.open(request, timeout=self.timeout) as response:
                    result.http_status = response.status
                    result.content_type = response.headers.get("Content-Type", "")
                    if not result.content_type.startswith("image/"):
                        result.error = "not_an_image"
                        break
                    data = response.read(self.max_bytes + 1)
                if len(data) > self.max_bytes:
                    result.error = "too_large"
                    break
                path = self.path_for(url)
                path.write_bytes(data)
                result.ok, result.error = True, ""
                result.n_bytes, result.path = len(data), str(path)
                result.seconds = time.monotonic() - started
                return result
            except urllib.error.HTTPError as exc:
                result.http_status = exc.code
                result.error = f"http_{exc.code}"
                if exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                result.error = "timeout" if "timed out" in str(exc).lower() else "network_error"
            result.seconds = time.monotonic() - started
            time.sleep(1.0)
        return result


# ---- quality checks -------------------------------------------------------


def dhash(gray_image) -> str:
    """A 64-bit fingerprint of what a picture looks like (same picture, same hash)."""
    import numpy as np

    small = np.asarray(gray_image.resize((9, 8)), dtype=float)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def inspect_image(path: Path) -> dict:
    """Open an image and measure its quality. Never raises: a bad file gives flags."""
    import numpy as np
    from PIL import Image

    info: dict = {"decodes": False, "flags": []}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            info.update(
                format=image.format, mode=image.mode, width=image.width, height=image.height
            )
            gray = image.convert("L")
            colour = image.convert("RGB")
    except Exception:  # noqa: BLE001 - any decoding problem is a result, not a crash
        info["flags"].append("cannot_decode")
        return info
    info["decodes"] = True
    pixels = np.asarray(gray, dtype=float)
    laplacian = (
        -4 * pixels[1:-1, 1:-1]
        + pixels[:-2, 1:-1]
        + pixels[2:, 1:-1]
        + pixels[1:-1, :-2]
        + pixels[1:-1, 2:]
    )
    info["pixel_std"] = round(float(pixels.std()), 2)
    info["white_share"] = round(float((pixels >= NEAR_WHITE).mean()), 4)
    info["sharpness"] = round(float(laplacian.var()), 2) if laplacian.size else 0.0
    info["hash"] = dhash(gray)
    width, height = info["width"], info["height"]
    flags = info["flags"]
    if min(width, height) < TINY_SIDE:
        flags.append("tiny")
    elif min(width, height) < LOW_RESOLUTION_SIDE:
        flags.append("low_resolution")
    if info["pixel_std"] < BLANK_STD or info["white_share"] > BLANK_WHITE_SHARE:
        flags.append("mostly_blank")
    if info["sharpness"] < BLURRY_LAPLACIAN:
        flags.append("blurry")
    if max(width / height, height / width) > EXTREME_ASPECT:
        flags.append("extreme_shape")
    t = time.perf_counter()
    colour.resize((224, 224))
    info["prep_ms"] = round((time.perf_counter() - t) * 1000, 2)
    return info


def mark_duplicates(rows: Sequence[dict]) -> None:
    """Add a `duplicate_of_another` flag when two different products show the same picture."""
    owners: dict[str, set[str]] = {}
    for row in rows:
        if row.get("hash"):
            owners.setdefault(row["hash"], set()).add(row["uniq_id"])
    for row in rows:
        if row.get("hash") and len(owners[row["hash"]]) > 1:
            row["flags"].append("same_picture_as_another_product")


# ---- sample selection -----------------------------------------------------


def shared_image_urls(
    urls_by_product: Mapping[str, Sequence[str]], minimum: int = 5
) -> dict[str, list[str]]:
    """Image URLs used by at least `minimum` different products, with those products."""
    users: dict[str, list[str]] = {}
    for uid, urls in urls_by_product.items():
        for url in set(urls):
            users.setdefault(url, []).append(uid)
    return {u: sorted(p) for u, p in users.items() if len(p) >= minimum}


def select_image_sample(
    products: Sequence[Product],
    urls_by_product: Mapping[str, Sequence[str]],
    locked_ids: Iterable[str],
    seed: int = SEED,
) -> list[dict]:
    """Choose the images to check. The locked final products are never included."""
    rng = random.Random(seed)
    locked = set(locked_ids)
    usable = sorted(
        (p for p in products if p.image_url and p.uniq_id not in locked), key=lambda p: p.uniq_id
    )
    by_id = {p.uniq_id: p for p in usable}
    category_size = Counter(p.category for p in products if p.category)
    title_count = Counter(p.title for p in products)
    shared = shared_image_urls({k: v for k, v in urls_by_product.items() if k in by_id})
    taken: set[str] = set()
    sample: list[dict] = []

    def add(group: str, product: Product, url: str) -> None:
        taken.add(product.uniq_id)
        sample.append(
            {
                "uniq_id": product.uniq_id,
                "group": group,
                "url": url,
                "title": product.title,
                "category": product.category or "",
            }
        )

    for group, count in SAMPLE_PLAN:
        if group == "shared image":
            urls = sorted(shared)
            rng.shuffle(urls)
            chosen = 0
            for url in urls:
                owners = [uid for uid in shared[url] if uid not in taken]
                if owners and chosen < count:
                    add(group, by_id[owners[0]], url)
                    chosen += 1
            continue
        tests = {
            "random": lambda p: True,
            "rare category": lambda p: bool(p.category) and category_size[p.category] < 50,
            "sparse data": lambda p: attributes_known(p) <= 4,
            "repeated title": lambda p: title_count[p.title] >= 3,
        }
        pool = [p for p in usable if p.uniq_id not in taken and tests[group](p)]
        for product in rng.sample(pool, min(count, len(pool))):
            add(group, product, product.image_url)
    return sample
