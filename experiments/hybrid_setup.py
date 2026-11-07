"""Shared set-up for the hybrid experiments: ratings, vectors and the evaluator.

Used by tune_hybrid_weights.py and run_hybrid_search.py (run from the project folder).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from product_similarity import evaluation as ev
from product_similarity.dedupe import picture_table
from product_similarity.hybrid import ATTRIBUTES, expand_four, random_pair_floor  # noqa: F401
from product_similarity.hybrid_eval import HybridEvaluator
from product_similarity.scoring import SimilarityEngine
from product_similarity.service import default_engine

ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = ROOT / "evaluation" / "judgments"
EMBEDDINGS = ROOT / "data" / "embeddings"
SEED = 2020
RATING_FILES = [
    "dev_grades.csv",
    "dev_grades_images.csv",
    "dev_grades_full_images.csv",
    "dev_grades_text_vectors.csv",
    "dev_grades_hybrid.csv",
    "dev_grades_final_search.csv",
    "dev_grades_final_search_fast.csv",  # added once the new fast-mode results have been rated
]


def merged_ratings() -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {}
    for name in RATING_FILES:
        path = JUDGMENTS / name
        if not path.exists():
            continue
        for no, by_id in ev.load_grades(path).items():
            for candidate, grade in by_id.items():
                assert candidate not in merged.setdefault(no, {}), "a result was rated twice"
                merged[no][candidate] = grade
    return merged


def build_evaluator() -> SimpleNamespace:
    queries_all = json.loads((ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    locked = {q["id"] for q in queries_all if q["split"] == "final"}
    dev = {q["query_no"]: q["id"] for q in queries_all if q["split"] == "dev"}
    grades = merged_ratings()

    base = default_engine()
    products = base.products
    clip = np.load(EMBEDDINGS / "all_clip.npz")
    clip_unit = clip["vectors"] / np.linalg.norm(clip["vectors"], axis=1, keepdims=True)
    row_of_link = {str(u): i for i, u in enumerate(clip["urls"])}
    not_photos = {
        r["url"]
        for r in json.loads((ROOT / "data/images/shared/excluded.json").read_text(encoding="utf-8"))
    }
    picture = np.full((len(products), clip_unit.shape[1]), np.nan, dtype=np.float32)
    for i, p in enumerate(products):
        if p.image_url in row_of_link and p.image_url not in not_photos:
            picture[i] = clip_unit[row_of_link[p.image_url]]
    text = np.load(EMBEDDINGS / "text_minilm.npz")
    assert [str(i) for i in text["ids"]] == [p.uniq_id for p in products], "vector order differs"
    sentence = text["vectors"].astype(np.float32)
    floors = {
        "image": random_pair_floor(picture, SEED),
        "text_vectors": random_pair_floor(sentence, SEED),
    }
    engine = SimilarityEngine(
        products,
        base.id_map,
        image_embeddings=picture,
        text_embeddings=sentence,
        embedding_floors=floors,
    )
    matrix, links = picture_table(products, row_of_link, clip_unit)
    started = time.monotonic()
    evaluator = HybridEvaluator(engine, dev, grades, locked, matrix, links)
    print(f"per-part scores ready in {time.monotonic() - started:.0f} s", flush=True)
    return SimpleNamespace(
        evaluator=evaluator,
        engine=engine,
        dev=dev,
        locked=locked,
        grades=grades,
        floors=floors,
        queries=sorted(dev),
    )
