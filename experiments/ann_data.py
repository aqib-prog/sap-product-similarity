"""Shared set-up for the fast-search experiments: the two vector spaces and the queries.

Used by the ann_*.py scripts (run from the project folder). The 20 locked products are
never part of a space, so they can never be returned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from product_similarity.image_features import normalise
from product_similarity.service import baseline_engine

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT / "data" / "embeddings"
SEED = 2020
RANDOM_QUERIES = 1000


@dataclass
class Space:
    name: str
    rows: np.ndarray  # product row (in the engine) of every vector
    matrix: np.ndarray  # float32, one length-1 vector per row

    @property
    def position(self) -> dict[int, int]:
        return {int(r): i for i, r in enumerate(self.rows)}


def load_spaces() -> tuple[dict[str, Space], dict[str, np.ndarray], list]:
    """Return the spaces, the query positions per space, and the engine's products."""
    queries_all = json.loads((ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    locked = {q["id"] for q in queries_all if q["split"] == "final"}
    dev = [q["id"] for q in queries_all if q["split"] == "dev"]
    engine = baseline_engine()
    products = engine.products
    not_locked = np.array([p.uniq_id not in locked for p in products])

    sentence = np.load(EMBEDDINGS / "text_minilm.npz")
    assert [str(i) for i in sentence["ids"]] == [p.uniq_id for p in products]
    sentence_vectors = sentence["vectors"]
    clip = np.load(EMBEDDINGS / "all_clip.npz")
    clip_vectors = clip["vectors"]  # read once: every access to an npz entry reloads the file
    clip_row = {str(u): i for i, u in enumerate(clip["urls"])}
    not_photos = {
        r["url"]
        for r in json.loads((ROOT / "data/images/shared/excluded.json").read_text(encoding="utf-8"))
    }
    has_picture = np.array(
        [p.image_url in clip_row and p.image_url not in not_photos for p in products]
    )

    text_rows = np.flatnonzero(not_locked)
    picture_rows = np.flatnonzero(not_locked & has_picture)
    spaces = {
        "sentence": Space(
            "sentence",
            text_rows,
            np.ascontiguousarray(sentence_vectors[text_rows], dtype=np.float32),
        ),
        "picture": Space(
            "picture",
            picture_rows,
            normalise(
                np.stack([clip_vectors[clip_row[products[i].image_url]] for i in picture_rows])
            ),
        ),
    }
    # Queries: the 30 development products, then random products that have a picture.
    dev_rows = [engine.resolve(pid) for pid in dev]
    rng = np.random.default_rng(SEED)
    pool = np.setdiff1d(picture_rows, dev_rows)
    random_rows = rng.choice(pool, RANDOM_QUERIES, replace=False)
    query_rows = {"dev30": np.array(dev_rows), "random1000": np.sort(random_rows)}
    return spaces, query_rows, products
