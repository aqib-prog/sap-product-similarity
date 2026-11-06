"""Download the first picture of every product (except the 20 locked final products).

Run from the project folder:
    python experiments/download_images.py --limit 200   # a small pilot
    python experiments/download_images.py               # everything that is left

Pictures go to data/images/all/ (ignored by git, never committed), one request at a
time and at most 4 per second. The script can be stopped and started again: pictures
already downloaded and links that are known to be gone (404) are not requested twice.
The full log stays in data/images/all/log.csv. Only a short summary and the list of
failures are written to results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

from product_similarity.images import ImageFetcher
from product_similarity.service import build_engine

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data" / "images" / "all"
EARLIER = ROOT / "data" / "images" / "dev"
LOG = FOLDER / "log.csv"
SUMMARY = ROOT / "results" / "image_download_summary.json"
FAILURES = ROOT / "results" / "image_download_failures.csv"
COLUMNS = ["url", "n_products", "ok", "error", "bytes", "seconds"]
SEED = 2020


def read_log() -> dict[str, dict]:
    if not LOG.exists():
        return {}
    with open(LOG, newline="", encoding="utf-8") as handle:
        return {row["url"]: row for row in csv.DictReader(handle)}


def write_log(rows: dict[str, dict]) -> None:
    with open(LOG, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="download only this many (pilot)")
    args = parser.parse_args()

    engine = build_engine(archive=ROOT / "data" / "archive.zip")
    queries = json.loads((ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    locked = {q["id"] for q in queries if q["split"] == "final"}

    owners: dict[str, list[str]] = {}
    for product in engine.products:
        if product.image_url and product.uniq_id not in locked:
            owners.setdefault(product.image_url, []).append(product.uniq_id)
    assert not locked & {uid for ids in owners.values() for uid in ids}, (
        "a locked product slipped in"
    )

    FOLDER.mkdir(parents=True, exist_ok=True)
    fetcher = ImageFetcher(FOLDER)
    old = ImageFetcher(EARLIER) if EARLIER.exists() else None
    log = read_log()

    # Reuse pictures already downloaded for the development products.
    reused = 0
    for url in owners:
        target = fetcher.path_for(url)
        if url not in log and not target.exists() and old and old.path_for(url).exists():
            shutil.copy2(old.path_for(url), target)
            reused += 1
    print(f"{len(owners)} links to cover, {reused} reused from the earlier download")

    def settled(url: str) -> bool:
        return fetcher.path_for(url).exists() or (url in log and log[url]["error"] == "http_404")

    todo = sorted(u for u in owners if not settled(u))
    if args.limit is not None:
        random.Random(SEED).shuffle(todo)
        todo = todo[: args.limit]
    print(f"{len(todo)} to download now")

    started = time.monotonic()
    try:
        for number, url in enumerate(todo, 1):
            result = fetcher.fetch(url)
            log[url] = {
                "url": url,
                "n_products": len(owners[url]),
                "ok": result.ok,
                "error": result.error,
                "bytes": result.n_bytes,
                "seconds": round(result.seconds, 3),
            }
            if number % 100 == 0:
                write_log(log)
                elapsed = time.monotonic() - started
                left = (len(todo) - number) * elapsed / number
                print(
                    f"{number} of {len(todo)} | {elapsed / 60:.1f} min | about {left / 60:.0f} min left",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("stopped: progress saved, run again to continue")
    write_log(log)

    for url in owners:  # pictures that exist but were never logged (reused ones)
        if url not in log and fetcher.path_for(url).exists():
            log[url] = {
                "url": url,
                "n_products": len(owners[url]),
                "ok": True,
                "error": "",
                "bytes": fetcher.path_for(url).stat().st_size,
                "seconds": "",
            }
    write_log(log)

    done = [r for r in log.values() if str(r["ok"]) == "True"]
    failed = [r for r in log.values() if str(r["ok"]) != "True"]
    summary = {
        "links_to_cover": len(owners),
        "downloaded": len(done),
        "failed": dict(Counter(r["error"] for r in failed)),
        "megabytes": round(sum(int(r["bytes"] or 0) for r in done) / 1e6, 1),
        "locked_products_included": 0,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with open(FAILURES, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["url", "n_products", "error"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in ("url", "n_products", "error")} for r in failed)
    print(summary)


if __name__ == "__main__":
    main()
