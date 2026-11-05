"""Turn the downloaded development pictures into vectors with ResNet-50 and CLIP.

Run from the project folder:  python experiments/extract_image_features.py
The vectors are saved under data/embeddings/ (ignored by git). Only the timing
numbers are written to results/ and kept in git.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from product_similarity.image_features import MODELS, ImageEncoder, embed_paths
from product_similarity.images import ImageFetcher

ROOT = Path(__file__).resolve().parents[1]
PICTURES = ROOT / "data" / "images" / "dev"
OUT_DIR = ROOT / "data" / "embeddings"
SPEED = ROOT / "results" / "dev_image_embedding_speed.json"


def main() -> None:
    rows = list(csv.DictReader((ROOT / "results/dev_image_downloads.csv").open(encoding="utf-8")))
    queries = json.loads((ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    locked = {q["id"] for q in queries if q["split"] == "final"}
    rows = [r for r in rows if r["ok"] == "True"]
    assert not locked & {r["uniq_id"] for r in rows}, "a locked product is in the list"
    fetcher = ImageFetcher(PICTURES)
    paths = [fetcher.path_for(r["url"]) for r in rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {"pictures": len(paths)}
    for name in MODELS:
        encoder = ImageEncoder(name)
        vectors, failed, timings = embed_paths(encoder, paths)
        keep = [i for i in range(len(paths)) if i not in set(failed)]
        ids = [rows[i]["uniq_id"] for i in keep]
        np.savez_compressed(OUT_DIR / f"dev_{name}.npz", ids=np.array(ids), vectors=vectors)
        report[name] = {
            **timings,
            "dimensions": int(vectors.shape[1]),
            "failed_to_open": len(failed),
        }
        print(name, report[name], flush=True)
    SPEED.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
