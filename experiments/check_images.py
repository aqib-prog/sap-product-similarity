"""Check a small sample of product images: do they download, and are they good?

Run from the project folder:  python experiments/check_images.py
Images are saved under data/images/sample/ (ignored by git, never committed).
Only the numbers are written to results/ and kept in git.
"""

from __future__ import annotations

import csv
import json
import statistics as st
import time
from collections import Counter
from pathlib import Path

from product_similarity import images as im
from product_similarity.data import canonicalize, load_products
from product_similarity.service import build_engine

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOCAL = ROOT / "data" / "images" / "sample"
COLUMNS = [
    "no",
    "uniq_id",
    "group",
    "url",
    "ok",
    "error",
    "http_status",
    "attempts",
    "bytes",
    "seconds",
    "width",
    "height",
    "format",
    "sharpness",
    "pixel_std",
    "white_share",
    "flags",
]


def main() -> None:
    archive = ROOT / "data" / "archive.zip"
    engine = build_engine(archive=archive)
    kept, _ = canonicalize(load_products(archive))
    urls = {r["uniq_id"]: [u for u in (r.get("medium") or "").split("|") if u] for r in kept}
    queries = json.loads((ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    locked = {q["id"] for q in queries if q["split"] == "final"}

    sample = im.select_image_sample(engine.products, urls, locked)
    plan = Counter(s["group"] for s in sample)
    print(
        "sample:",
        dict(plan),
        "| total",
        len(sample),
        "| locked products included:",
        len({s["uniq_id"] for s in sample} & locked),
    )

    fetcher = im.ImageFetcher(LOCAL)
    rows, started = [], time.monotonic()
    for number, item in enumerate(sample, start=1):
        result = fetcher.fetch(item["url"])
        row = {
            "no": number,
            **item,
            "ok": result.ok,
            "error": result.error,
            "http_status": result.http_status,
            "attempts": result.attempts,
            "bytes": result.n_bytes,
            "seconds": round(result.seconds, 3),
            "flags": [],
        }
        if result.ok:
            row.update(im.inspect_image(Path(result.path)))
            row["path"] = result.path
        else:
            row["flags"] = ["download_failed"]
        rows.append(row)
    wall = time.monotonic() - started
    im.mark_duplicates(rows)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "image_sample.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(
                [";".join(row["flags"]) if c == "flags" else row.get(c, "") for c in COLUMNS]
            )

    ok = [r for r in rows if r["ok"]]
    flagged = Counter(f for r in rows for f in r["flags"])
    failed = Counter(r["error"] for r in rows if not r["ok"])
    sizes = [r["bytes"] for r in ok]
    distinct = len({p.image_url for p in engine.products if p.image_url})
    mean_bytes = st.mean(sizes) if sizes else 0
    per_request = wall / len(rows)
    prep = st.mean(r["prep_ms"] for r in ok if "prep_ms" in r) if ok else 0
    summary = {
        "requested": len(rows),
        "downloaded": len(ok),
        "failed": dict(failed),
        "flags": dict(flagged),
        "wall_seconds": round(wall, 1),
        "bytes_min_median_mean_max": [
            min(sizes),
            int(st.median(sizes)),
            int(mean_bytes),
            max(sizes),
        ]
        if sizes
        else [],
        "dims": Counter(f"{r['width']}x{r['height']}" for r in ok).most_common(6),
        "distinct_first_images": distinct,
        "per_request_seconds": round(per_request, 2),
        "full_download_mb": round(mean_bytes * distinct / 1e6),
        "full_download_minutes": round(per_request * distinct / 60),
        "prep_ms": round(prep, 1),
    }
    (RESULTS / "image_sample_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    (LOCAL / "rows.json").write_text(json.dumps(rows, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
