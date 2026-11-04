"""Turn scores into a fixed, repeatable order."""

from __future__ import annotations

import numpy as np


def rank_indices(scores: np.ndarray, id_rank: np.ndarray, exclude: int) -> np.ndarray:
    """Return product positions from most to least similar, leaving out `exclude`.

    Scores are rounded to 6 decimals so tiny floating-point differences do not
    decide the order. Equal scores are ordered by `id_rank` (the position of the
    product's ID in sorted order), so the same query always gives the same list.
    Products without a score (NaN) go last.
    """
    rounded = np.round(scores.astype(float), 6)
    rounded = np.where(np.isnan(rounded), -np.inf, rounded)
    order = np.lexsort((id_rank, -rounded))
    return order[order != exclude]
