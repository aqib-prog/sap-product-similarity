"""Download the first picture of the development products and their rated results.

Run from the project folder:  python experiments/download_dev_images.py
Pictures go to data/images/dev/ (ignored by git, never committed). Only the
numbers about the downloads are written to results/dev_image_downloads.csv.
The 20 locked final products are never downloaded.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from product_similarity.images import ImageFetcher
from product_similarity.service import build_engine

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "data" / "images" / "dev"
OUT = ROOT / "results" / "dev_image_downloads.csv"


def main() -> None:
    engine = build_engine(archive=ROOT / "data" / "archive.zip")
    by_id = {p.uniq_id: p for p in engine.products}
    queries = json.loads((ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    locked = {q["id"] for q in queries if q["split"] == "final"}
    dev_ids = [q["id"] for q in queries if q["split"] == "dev"]

    with (ROOT / "evaluation/judgments/dev_grades.csv").open(encoding="utf-8") as handle:
        rated_ids = [row["candidate_id"] for row in csv.DictReader(handle)]

    wanted = sorted(set(dev_ids) | set(rated_ids))
    assert not locked & set(wanted), "a locked product is in the list"
    print(f"{len(dev_ids)} dev queries + {len(set(rated_ids))} rated results = {len(wanted)}")

    fetcher = ImageFetcher(LOCAL)
    rows = []
    for number, uid in enumerate(wanted, 1):
        url = by_id[uid].image_url if uid in by_id else None
        if not url:
            rows.append({"uniq_id": uid, "url": "", "ok": False, "error": "no_image_url"})
            continue
        path = fetcher.path_for(url)
        if path.exists():
            size = path.stat().st_size
            rows.append({"uniq_id": uid, "url": url, "ok": True, "error": "", "bytes": size})
        else:
            result = fetcher.fetch(url)
            rows.append(
                {
                    "uniq_id": uid,
                    "url": url,
                    "ok": result.ok,
                    "error": result.error,
                    "bytes": result.n_bytes,
                    "seconds": round(result.seconds, 3),
                }
            )
        if number % 100 == 0:
            print(number, "of", len(wanted), flush=True)

    columns = ["uniq_id", "url", "ok", "error", "bytes", "seconds"]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: r.get(c, "") for c in columns} for r in rows)
    failed = Counter(r["error"] for r in rows if not r["ok"])
    print("ok:", sum(r["ok"] for r in rows), "| failed:", dict(failed))


if __name__ == "__main__":
    main()
