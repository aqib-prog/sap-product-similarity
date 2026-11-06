"""Turn every downloaded catalogue picture into a CLIP vector.

Run from the project folder:  python experiments/extract_all_image_features.py
Vectors go to data/embeddings/all_clip.npz (ignored by git); only timings go to results/.
The picture list comes from the download log, which never contains the locked products.
Progress is saved in chunks, so a stopped run can be started again.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from product_similarity.image_features import ImageEncoder, embed_paths
from product_similarity.images import ImageFetcher

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data" / "images" / "all"
OUT_DIR = ROOT / "data" / "embeddings"
PARTS = OUT_DIR / "all_clip_parts"
SPEED = ROOT / "results" / "image_embedding_speed_all.json"
CHUNK = 1000


def main() -> None:
    with open(FOLDER / "log.csv", newline="", encoding="utf-8") as handle:
        urls = sorted(r["url"] for r in csv.DictReader(handle) if r["ok"] == "True")
    fetcher = ImageFetcher(FOLDER)
    PARTS.mkdir(parents=True, exist_ok=True)
    encoder = ImageEncoder("clip")
    started = time.monotonic()
    failed_total = 0
    for number, start in enumerate(range(0, len(urls), CHUNK)):
        part = PARTS / f"part_{number:03d}.npz"
        if part.exists():
            continue
        chunk = urls[start : start + CHUNK]
        vectors, failed, _ = embed_paths(encoder, [fetcher.path_for(u) for u in chunk])
        bad = set(failed)
        keep = [i for i in range(len(chunk)) if i not in bad]
        failed_total += len(bad)
        np.savez_compressed(part, urls=np.array([chunk[i] for i in keep]), vectors=vectors)
        done = min(start + CHUNK, len(urls))
        print(f"{done} of {len(urls)} | {(time.monotonic() - started) / 60:.1f} min", flush=True)
    files = sorted(PARTS.glob("part_*.npz"))
    all_urls, all_vec = [], []
    for f in files:
        data = np.load(f)
        all_urls += list(data["urls"])
        all_vec.append(data["vectors"])
    matrix = np.vstack(all_vec)
    np.savez_compressed(OUT_DIR / "all_clip.npz", urls=np.array(all_urls), vectors=matrix)
    SPEED.write_text(
        json.dumps(
            {
                "pictures": len(urls),
                "vectors": int(matrix.shape[0]),
                "dimensions": int(matrix.shape[1]),
                "failed_to_open": failed_total,
                "minutes": round((time.monotonic() - started) / 60, 1),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("saved", matrix.shape)


if __name__ == "__main__":
    main()
