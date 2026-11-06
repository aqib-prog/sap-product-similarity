"""Show one copy of each item: collapse results that are the same item.

Two pictures count as the same item when they are the same link or when their vectors
are almost identical (cosine similarity at or above SAME_ITEM). The first (best) copy
of each item is kept. Copies of the query's own item are removed, because they are not
new recommendations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SAME_ITEM = 0.97


def pick_distinct(
    ranked: Sequence[int],
    vectors: np.ndarray,
    links: Sequence[str],
    query: int,
    k: int,
    threshold: float = SAME_ITEM,
) -> tuple[list[int], dict[str, int]]:
    """Walk down a ranked list of row numbers and keep the first k distinct items.

    Returns the kept rows and how many were dropped as copies of the query or of an
    item already kept.
    """
    kept: list[int] = []
    dropped = {"copy_of_query": 0, "copy_of_result": 0}
    for row in ranked:
        if row == query:
            continue
        if links[row] == links[query] or float(vectors[row] @ vectors[query]) >= threshold:
            dropped["copy_of_query"] += 1
            continue
        if any(
            links[row] == links[o] or float(vectors[row] @ vectors[o]) >= threshold for o in kept
        ):
            dropped["copy_of_result"] += 1
            continue
        kept.append(row)
        if len(kept) == k:
            break
    return kept, dropped


def picture_table(
    products: Sequence, vector_of_link: dict[str, int], vectors: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """One picture vector and one link per product, for pick_distinct.

    Products without a picture get a zero vector and a link of their own, so they are
    never treated as copies of anything.
    """
    matrix = np.zeros((len(products), vectors.shape[1]), dtype=vectors.dtype)
    links: list[str] = []
    for row, product in enumerate(products):
        if product.image_url in vector_of_link:
            matrix[row] = vectors[vector_of_link[product.image_url]]
            links.append(product.image_url)
        else:
            links.append(f"no-picture-{product.uniq_id}")
    return matrix, links
