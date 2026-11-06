"""Score a set of hybrid weights against the human ratings, fast.

The per-part scores of each test product against every other product are computed once.
A set of weights then only needs a weighted average and a ranking, so thousands of
weight sets can be tried. The ranking follows the same rules as the other methods: one
copy of each item, and the locked products are never returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from product_similarity import evaluation as ev
from product_similarity.dedupe import pick_distinct
from product_similarity.scoring import SimilarityEngine

PARTS = (
    "title",
    "text_vectors",
    "image",
    "category",
    "brand",
    "audience",
    "price",
    "colour",
    "weight",
    "rating",
)


class HybridEvaluator:
    def __init__(
        self,
        engine: SimilarityEngine,
        queries: Mapping[str, str],
        grades: Mapping[str, Mapping[str, int]],
        excluded_ids: set[str],
        picture_matrix: np.ndarray,
        picture_links: Sequence[str],
        depth: int = 60,
    ) -> None:
        self.engine = engine
        self.grades = grades
        self.picture_matrix = picture_matrix
        self.picture_links = picture_links
        self.depth = depth
        self.query_row = {no: engine.resolve(pid) for no, pid in queries.items()}
        blocked = np.array([uid in excluded_ids for uid in engine.ids])
        self._values: dict[str, np.ndarray] = {}
        self._known: dict[str, np.ndarray] = {}
        self._allowed: dict[str, np.ndarray] = {}
        for no, row in self.query_row.items():
            sims = engine.attribute_similarities(row)
            values = np.zeros((len(PARTS), len(engine.products)), dtype=np.float32)
            known = np.zeros_like(values)
            for k, part in enumerate(PARTS):
                if part in sims:
                    column = sims[part]
                    ok = ~np.isnan(column)
                    values[k] = np.where(ok, column, 0.0)
                    known[k] = ok
            allowed = ~blocked
            allowed[row] = False
            self._values[no], self._known[no], self._allowed[no] = values, known, allowed

    def top_ids(
        self,
        weights: Mapping[str, float],
        query_no: str,
        candidates: Sequence[int] | None = None,
    ) -> list[str]:
        """Top 10 for a query. With `candidates` (product rows), only those can be returned."""
        w = np.array([weights.get(part, 0.0) for part in PARTS], dtype=np.float32)
        total = w @ self._values[query_no]
        weight_sum = w @ self._known[query_no]
        has_score = (weight_sum > 0) & self._allowed[query_no]
        if candidates is not None:
            only = np.zeros_like(has_score)
            only[np.asarray(candidates, dtype=np.int64)] = True
            has_score &= only
        scores = np.where(has_score, total / np.where(weight_sum > 0, weight_sum, 1.0), -np.inf)
        rounded = np.round(scores.astype(np.float64), 6)
        depth = min(self.depth, len(rounded) - 1)
        top = np.argpartition(-rounded, depth)[:depth]
        order = top[np.lexsort((self.engine._id_rank[top], -rounded[top]))]
        kept, _ = pick_distinct(
            order.tolist(),
            self.picture_matrix,
            self.picture_links,
            self.query_row[query_no],
            ev.K,
        )
        return [str(self.engine.ids[i]) for i in kept]

    def per_query(self, weights: Mapping[str, float], queries: Sequence[str]) -> dict[str, dict]:
        ranked = {no: self.top_ids(weights, no) for no in queries}
        return ev.score_method(ranked, self.grades)

    def score(self, weights: Mapping[str, float], queries: Sequence[str]) -> float:
        """Mean nDCG@10 over the queries that have something to find."""
        values = [
            q["ndcg"] for q in self.per_query(weights, queries).values() if q["ndcg"] is not None
        ]
        return sum(values) / len(values) if values else 0.0
