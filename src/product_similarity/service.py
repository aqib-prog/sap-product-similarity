"""The public entry point: `find_similar_products`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from product_similarity.data import canonicalize, load_products
from product_similarity.preprocessing import build_products
from product_similarity.scoring import ProductNotFoundError, SimilarityEngine
from product_similarity.settings import get_settings

__all__ = ["ProductNotFoundError", "build_engine", "default_engine", "find_similar_products"]


def build_engine(
    rows: list[dict[str, Any]] | None = None,
    archive: Path | None = None,
    weights: dict[str, float] | None = None,
    image_embeddings: np.ndarray | None = None,
) -> SimilarityEngine:
    """Clean the raw rows and build an engine. Reads the archive if no rows are given."""
    if rows is None:
        rows = load_products(archive or get_settings().data_archive)
    kept, id_map = canonicalize(rows)
    products, _ = build_products(kept)
    return SimilarityEngine(products, id_map, weights, image_embeddings)


@lru_cache(maxsize=1)
def default_engine() -> SimilarityEngine:
    """The engine for the supplied dataset, built once on first use."""
    return build_engine()


def find_similar_products(product_id: str, num_similar: int) -> list[str]:
    """Return the IDs of the `num_similar` products most similar to `product_id`.

    Raises ProductNotFoundError for an unknown ID and ValueError if `num_similar`
    is not a whole number of at least 1. If fewer products exist, all are returned.
    """
    return default_engine().find_similar(product_id, num_similar)
