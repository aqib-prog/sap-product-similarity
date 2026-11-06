"""Exact (brute-force) vector search: the reference that fast indexes are compared with.

Offline tooling only: the web service never imports this module. All vectors are scaled
to length 1, so a dot product is the cosine similarity. Ties are broken by row number so
the answer is always the same.
"""

from __future__ import annotations

import numpy as np


def exact_top_k(
    matrix: np.ndarray, query_rows: np.ndarray, k: int = 10, block: int = 256
) -> np.ndarray:
    """The k most similar rows for each query row, best first, leaving out the query itself.

    `query_rows` are row numbers of `matrix`. Returns an array of shape (queries, k).
    """
    query_rows = np.asarray(query_rows, dtype=np.int64)
    if not 1 <= k < len(matrix):
        raise ValueError("k must be at least 1 and smaller than the number of rows")
    out = np.empty((len(query_rows), k), dtype=np.int64)
    pool = min(len(matrix) - 1, 4 * k)  # a few extra so ties are ordered consistently
    for start in range(0, len(query_rows), block):
        rows = query_rows[start : start + block]
        scores = matrix[rows] @ matrix.T
        scores[np.arange(len(rows)), rows] = -np.inf
        top = np.argpartition(-scores, pool, axis=1)[:, :pool]
        for i, candidates in enumerate(top):
            rounded = np.round(scores[i, candidates].astype(np.float64), 6)
            order = np.lexsort((candidates, -rounded))[:k]
            out[start + i] = candidates[order]
    return out


def recall_at_k(found: np.ndarray, exact: np.ndarray) -> float:
    """Share of the exact top-k that the other method also found, averaged over queries."""
    hits = [len(set(a.tolist()) & set(b.tolist())) / len(b) for a, b in zip(found, exact)]
    return float(np.mean(hits))


def recall_at_k_by_score(
    found: np.ndarray,
    matrix: np.ndarray,
    query_rows: np.ndarray,
    exact: np.ndarray,
    slack: float = 1e-6,
) -> float:
    """Recall@k that does not punish ties.

    A returned row counts as correct if it is at least as similar to the query as the exact
    k-th row. Products with identical vectors tie, and a different tie member is not a miss.
    """
    query_rows = np.asarray(query_rows)
    kth = np.einsum("ij,ij->i", matrix[query_rows], matrix[exact[:, -1]])
    hits = []
    for q, row, bar in zip(query_rows, found, kth):
        scores = matrix[row] @ matrix[q]
        hits.append(float(np.sum(scores >= bar - slack)) / exact.shape[1])
    return float(np.mean(hits))
