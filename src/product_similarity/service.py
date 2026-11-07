"""The public entry point: `find_similar_products`."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from product_similarity.data import canonicalize, load_products
from product_similarity.hybrid import MODES, HybridEngine, load_final_vectors
from product_similarity.preprocessing import build_products
from product_similarity.scoring import ProductNotFoundError, SimilarityEngine
from product_similarity.settings import get_settings
from product_similarity.vector_files import load_vector_artifact

logger = logging.getLogger(__name__)

__all__ = [
    "ProductNotFoundError",
    "build_engine",
    "baseline_engine",
    "build_final_engine",
    "default_engine",
    "find_similar_products",
]


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


def build_final_engine(
    mode: str | None = None, rows: list[dict[str, Any]] | None = None
) -> SimilarityEngine:
    """The final search: the tuned hybrid, in exact or fast mode.

    Falls back to the title-and-attributes baseline when the saved vectors are missing
    (or do not belong to these products), or when the mode is "baseline".
    """
    settings = get_settings()
    mode = mode or settings.search_mode
    if mode != "baseline" and mode not in MODES:
        raise ValueError(f"search mode must be one of {(*MODES, 'baseline')}")
    base = build_engine() if rows is None else build_engine(rows)
    if mode == "baseline":
        return base
    # The half-size vectors that ship with the service come first; the full-size ones are
    # only there on the machine where they were made.
    vectors = load_vector_artifact(base.products, settings.vectors_dir)
    if vectors is None:
        vectors = load_final_vectors(base.products, settings.embeddings_dir, settings.not_photos)
    if vectors is None:
        if settings.require_vectors:
            raise RuntimeError(
                f"No usable vectors in {settings.vectors_dir} (PS_REQUIRE_VECTORS=1), "
                "so the service will not start with the baseline search"
            )
        logger.warning("No usable vectors in %s: using the baseline search", settings.vectors_dir)
        return base
    return HybridEngine(base.products, base.id_map, vectors, mode)


@lru_cache(maxsize=1)
def baseline_engine() -> SimilarityEngine:
    """The title-and-attributes baseline, for the experiments that evaluate it."""
    return build_engine()


@lru_cache(maxsize=1)
def default_engine() -> SimilarityEngine:
    """The engine for the supplied dataset, built once on first use."""
    return build_final_engine()


def find_similar_products(product_id: str, num_similar: int) -> list[str]:
    """Return the IDs of the `num_similar` products most similar to `product_id`.

    Raises ProductNotFoundError for an unknown ID and ValueError if `num_similar`
    is not a whole number of at least 1. If fewer products exist, all are returned.
    """
    return default_engine().find_similar(product_id, num_similar)
