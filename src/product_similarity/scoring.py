"""Product-to-product similarity.

Each attribute gives a score between 0 and 1. The scores are mixed with weights.
When either product lacks an attribute, that attribute is skipped and the
remaining weights are rescaled, so missing data is never filled in with a guess.

The starting weights are NOT tuned. They are a placeholder until experiments
choose better ones.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from numbers import Integral

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from product_similarity.preprocessing import Product
from product_similarity.ranking import rank_indices

DEFAULT_WEIGHTS = {
    "title": 0.35,
    "category": 0.20,
    "brand": 0.10,
    "audience": 0.10,
    "price": 0.10,
    "colour": 0.05,
    "weight": 0.05,
    "rating": 0.05,
    # Switched on by giving the engine image embeddings and a weight above 0.
    "image": 0.0,
}

# How far apart two values can be before their score reaches 0.
PRICE_SPAN = 1.0  # in log10 units, so a 10x price difference scores 0
WEIGHT_SPAN = 1.0  # in log10 units
RATING_SPAN = 4.0  # stars


class ProductNotFoundError(KeyError):
    """The product ID is not in the dataset."""


def _array(values: Iterable[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _codes(values: Sequence[str | None]) -> np.ndarray:
    """Turn labels into small integers. Missing becomes -1."""
    seen: dict[str, int] = {}
    return np.array([-1 if v is None else seen.setdefault(v, len(seen)) for v in values])


def _closeness(values: np.ndarray, query: float, span: float) -> np.ndarray:
    """1 when equal, falling to 0 once the gap reaches `span`. NaN where unknown."""
    if np.isnan(query):
        return np.full(len(values), np.nan)
    return np.clip(1.0 - np.abs(values - query) / span, 0.0, 1.0)


class SimilarityEngine:
    """Scores one product against all others and returns the best matches."""

    def __init__(
        self,
        products: Sequence[Product],
        id_map: dict[str, str] | None = None,
        weights: dict[str, float] | None = None,
        image_embeddings: np.ndarray | None = None,
    ) -> None:
        if not products:
            raise ValueError("at least one product is needed")
        weights = weights or {}
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown weight names: {sorted(unknown)}")
        if any(w < 0 for w in weights.values()):
            raise ValueError("weights cannot be negative")

        self.products = list(products)
        self.weights = {**DEFAULT_WEIGHTS, **weights}
        self.id_map = dict(id_map or {})
        self.ids = np.array([p.uniq_id for p in self.products])
        self._row = {uid: i for i, uid in enumerate(self.ids)}
        self._id_rank = np.argsort(np.argsort(self.ids))

        self._tfidf = TfidfVectorizer(sublinear_tf=True).fit_transform(
            [p.title for p in self.products]
        )
        self._codes = {
            "category": _codes([p.category for p in self.products]),
            "brand": _codes([p.brand for p in self.products]),
            "audience": _codes([p.audience for p in self.products]),
        }
        self._price = _array(p.log_price for p in self.products)
        self._weight = _array(math.log10(p.weight_g) if p.weight_g else None for p in self.products)
        self._rating = _array(p.rating for p in self.products)
        self._colours, self._colour_sizes = self._colour_matrix()
        self._image = self._check_images(image_embeddings)

    # ---- setup helpers ----

    def _colour_matrix(self) -> tuple[sparse.csr_matrix, np.ndarray]:
        sets = [set(p.title_colours) | set(p.listed_colours) for p in self.products]
        vocabulary = {c: j for j, c in enumerate(sorted(set().union(*sets)))}
        rows = [i for i, s in enumerate(sets) for _ in s]
        cols = [vocabulary[c] for s in sets for c in s]
        matrix = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(sets), len(vocabulary))
        )
        return matrix, np.array([len(s) for s in sets], dtype=float)

    def _check_images(self, embeddings: np.ndarray | None) -> np.ndarray | None:
        if embeddings is None:
            return None
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.products):
            raise ValueError("image_embeddings needs one row per product")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            normalised = embeddings / norms
        # Rows that are all NaN or all zero mean "no image" and stay NaN.
        normalised[~np.isfinite(norms).ravel() | (norms.ravel() == 0)] = np.nan
        return normalised

    # ---- scoring ----

    def attribute_similarities(self, i: int) -> dict[str, np.ndarray]:
        """Per-attribute scores for product `i` against every product (NaN = unknown)."""
        n = len(self.products)
        unknown = np.full(n, np.nan)
        sims: dict[str, np.ndarray] = {"title": (self._tfidf @ self._tfidf[i].T).toarray().ravel()}
        for name, codes in self._codes.items():
            if codes[i] < 0:
                sims[name] = unknown.copy()
            else:
                same = (codes == codes[i]).astype(float)
                sims[name] = np.where(codes < 0, np.nan, same)
        sims["price"] = _closeness(self._price, self._price[i], PRICE_SPAN)
        sims["weight"] = _closeness(self._weight, self._weight[i], WEIGHT_SPAN)
        sims["rating"] = _closeness(self._rating, self._rating[i], RATING_SPAN)

        if self._colour_sizes[i] == 0:
            sims["colour"] = unknown.copy()
        else:
            shared = (self._colours @ self._colours[i].T).toarray().ravel()
            union = self._colour_sizes + self._colour_sizes[i] - shared
            sims["colour"] = np.where(
                self._colour_sizes > 0, shared / np.maximum(union, 1.0), np.nan
            )

        if self._image is not None:
            query = self._image[i]
            if np.isnan(query).any():
                sims["image"] = unknown.copy()
            else:
                sims["image"] = np.clip(self._image @ query, 0.0, 1.0)
        return sims

    def combine(self, sims: dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average over the attributes both products have."""
        n = len(self.products)
        total, weight_sum = np.zeros(n), np.zeros(n)
        for name, sim in sims.items():
            weight = self.weights[name]
            if weight == 0:
                continue
            known = ~np.isnan(sim)
            total += np.where(known, weight * np.nan_to_num(sim), 0.0)
            weight_sum += np.where(known, weight, 0.0)
        safe = np.where(weight_sum > 0, weight_sum, 1.0)
        return np.where(weight_sum > 0, total / safe, np.nan)

    def score_all(self, i: int) -> np.ndarray:
        return self.combine(self.attribute_similarities(i))

    # ---- public lookups ----

    def resolve(self, product_id: str) -> int:
        """Position of a product. A repeated (old) ID maps to the product that was kept."""
        key = self.id_map.get(product_id, product_id)
        if key not in self._row:
            raise ProductNotFoundError(product_id)
        return self._row[key]

    def find_similar_with_scores(
        self, product_id: str, num_similar: int
    ) -> list[tuple[str, float]]:
        """The `num_similar` best matches as (ID, score), best first."""
        if (
            not isinstance(num_similar, Integral)
            or isinstance(num_similar, bool)
            or num_similar < 1
        ):
            raise ValueError("num_similar must be a whole number of at least 1")
        i = self.resolve(product_id)
        scores = self.score_all(i)
        order = rank_indices(scores, self._id_rank, exclude=i)[: int(num_similar)]
        return [(str(self.ids[j]), round(float(scores[j]), 6)) for j in order]

    def find_similar(self, product_id: str, num_similar: int) -> list[str]:
        return [uid for uid, _ in self.find_similar_with_scores(product_id, num_similar)]

    @staticmethod
    def _summary(product: Product) -> dict:
        """The fields worth showing for a product."""
        return {
            "id": product.uniq_id,
            "title": product.title,
            "brand": product.brand,
            "category": product.category,
            "audience": product.audience,
            "price": product.price,
            "rating": product.rating,
            "image_url": product.image_url,
        }

    def describe(self, product_id: str, num_similar: int) -> dict:
        """The query product and its best matches with readable details and scores."""
        ranked = self.find_similar_with_scores(product_id, num_similar)
        query = self.products[self.resolve(product_id)]
        results = []
        for uid, score in ranked:
            results.append({**self._summary(self.products[self._row[uid]]), "score": score})
        return {"query": self._summary(query), "results": results}

    def explain(self, product_id: str, other_id: str) -> dict:
        """Why `other_id` scores as it does for `product_id`, attribute by attribute.

        `share` is each attribute's part of the final score, so the shares add up to it.
        """
        i, j = self.resolve(product_id), self.resolve(other_id)
        sims = self.attribute_similarities(i)
        used = {
            name: float(sim[j])
            for name, sim in sims.items()
            if self.weights[name] > 0 and not np.isnan(sim[j])
        }
        weight_sum = sum(self.weights[name] for name in used)
        parts = {
            name: {
                "similarity": round(value, 6),
                "weight": self.weights[name],
                "share": round(self.weights[name] * value / weight_sum, 6),
            }
            for name, value in used.items()
        }
        skipped = [n for n in sims if self.weights[n] > 0 and n not in used]
        score = float(self.score_all(i)[j])
        return {"score": round(score, 6), "attributes": parts, "skipped": skipped}
