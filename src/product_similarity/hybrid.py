"""The final search: the tuned hybrid over all ten parts, exact or with a fast index.

Exact mode scores every product with the hybrid. Fast mode only speeds up the two vector
parts, which are the expensive ones to compare with every product: an HNSW index for the
sentence vectors and one for the picture vectors each return the nearest products, and only
that short list is scored with all ten parts, exactly as in exact mode (a part that a product
does not have is skipped and the other weights are rescaled). Both modes show one copy of
each item, like the evaluation does.

The weights are the 4-weight mix tuned in notebook 08 (coordinate ascent on nDCG@10, with
5-fold cross-validation). The vectors are computed once, offline, with `experiments/`
scripts, so the service never loads CLIP or the sentence model.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from product_similarity.dedupe import pick_distinct, picture_table
from product_similarity.preprocessing import Product
from product_similarity.ranking import rank_indices
from product_similarity.scoring import DEFAULT_WEIGHTS, SimilarityEngine

logger = logging.getLogger(__name__)

# The tuned mix (results/hybrid_tuning.json, "four_weights"). The seven attributes share
# the "attributes" weight in the same ratios as the untuned weights.
FINAL_MIX = {
    "title": 0.028571428571428574,
    "text_vectors": 0.4571428571428572,
    "image": 0.2857142857142857,
    "attributes": 0.2285714285714286,
}
ATTRIBUTES = ("category", "brand", "audience", "price", "colour", "weight", "rating")
SEED = 2020
MODES = ("exact", "fast")


def expand_four(group: Mapping[str, float]) -> dict[str, float]:
    """Turn the 4-weight mix into one weight per part."""
    share = {k: DEFAULT_WEIGHTS[k] for k in ATTRIBUTES}
    total = sum(share.values())
    weights = {k: group["attributes"] * v / total for k, v in share.items()}
    weights.update({k: group[k] for k in ("title", "text_vectors", "image")})
    return weights


def random_pair_floor(vectors: np.ndarray, seed: int = SEED, pairs: int = 20000) -> float:
    """Mean cosine of random pairs: the similarity of two unrelated products."""
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(~np.isnan(vectors).any(axis=1))
    a, b = rng.choice(valid, pairs), rng.choice(valid, pairs)
    keep = a != b
    return float(np.mean(np.sum(vectors[a[keep]] * vectors[b[keep]], axis=1)))


@dataclass
class FinalVectors:
    """The precomputed vectors, one row per product (NaN where a product has none)."""

    picture: np.ndarray
    sentence: np.ndarray
    floors: dict[str, float]
    copies_matrix: np.ndarray  # picture vectors used to spot copies of the same item
    copies_links: list[str]


def load_final_vectors(
    products: Sequence[Product],
    embeddings_dir: Path,
    not_photos: Path | None = None,
) -> FinalVectors | None:
    """Read the saved picture (CLIP) and sentence vectors, or None if they are not there
    or do not belong to these products."""
    clip_file, text_file = embeddings_dir / "all_clip.npz", embeddings_dir / "text_minilm.npz"
    if not clip_file.exists() or not text_file.exists():
        return None
    text = np.load(text_file)
    if [str(i) for i in text["ids"]] != [p.uniq_id for p in products]:
        logger.warning("The saved sentence vectors do not match the products")
        return None
    clip = np.load(clip_file)
    unit = clip["vectors"] / np.linalg.norm(clip["vectors"], axis=1, keepdims=True)
    row_of_link = {str(u): i for i, u in enumerate(clip["urls"])}
    excluded: set[str] = set()
    if not_photos is not None and not_photos.exists():
        excluded = {r["url"] for r in json.loads(not_photos.read_text(encoding="utf-8"))}
    picture = np.full((len(products), unit.shape[1]), np.nan, dtype=np.float32)
    for i, p in enumerate(products):
        if p.image_url in row_of_link and p.image_url not in excluded:
            picture[i] = unit[row_of_link[p.image_url]]
    sentence = text["vectors"].astype(np.float32)
    floors = {
        "image": random_pair_floor(picture),
        "text_vectors": random_pair_floor(sentence),
    }
    matrix, links = picture_table(products, row_of_link, unit)
    return FinalVectors(picture, sentence, floors, matrix, links)


class HybridEngine(SimilarityEngine):
    """The tuned hybrid, as `SimilarityEngine`, with an exact and a fast mode."""

    # The vectors are stored in the normal 4-byte format (half the size of the default).
    _embedding_dtype = np.float32

    def __init__(
        self,
        products: Sequence[Product],
        id_map: dict[str, str] | None,
        vectors: FinalVectors,
        mode: str = "exact",
        candidates: int = 100,
        ef_search: int = 64,
        weights: dict[str, float] | None = None,
        hidden_ids: Collection[str] = (),
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        super().__init__(
            products,
            id_map,
            weights or expand_four(FINAL_MIX),
            image_embeddings=vectors.picture,
            text_embeddings=vectors.sentence,
            embedding_floors=vectors.floors,
        )
        self.mode = mode
        self.candidates = candidates
        self.ef_search = ef_search
        self._copies_matrix = vectors.copies_matrix
        self._copies_links = vectors.copies_links
        hidden = set(hidden_ids)  # products that are never returned (used by the checks)
        self._hidden = np.array([uid in hidden for uid in self.ids], dtype=bool)
        self._indexes: dict[str, tuple[object, np.ndarray]] = {}
        self._build_lock = threading.Lock()
        if mode == "fast":
            self._build_indexes()

    # ---- the fast mode: one HNSW index for each vector part ----

    def _vector_parts(self):
        return (("text_vectors", self._text_vectors), ("image", self._image))

    def _build_indexes(self) -> None:
        with self._build_lock:
            if not self._indexes:
                self._build_indexes_now()

    def _build_indexes_now(self) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError("fast mode needs faiss-cpu: pip install -e '.[ann]'") from exc
        # One thread while building, so that the same data always gives the same index.
        threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)
        try:
            for name, matrix in self._vector_parts():
                # Only products that have this vector are in its index. A product without a
                # picture is still found through its sentence vector.
                rows = np.flatnonzero(~self._hidden & ~np.isnan(matrix).any(axis=1))
                data = np.ascontiguousarray(matrix[rows], dtype=np.float32)
                index = faiss.IndexHNSWFlat(data.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = self.ef_search
                index.add(data)
                self._indexes[name] = (index, rows)
        finally:
            faiss.omp_set_num_threads(threads)

    def index_mb(self) -> float:
        """Size of the two HNSW indexes (fast mode), in MB."""
        if not self._indexes:
            return 0.0
        import faiss

        return round(
            sum(len(faiss.serialize_index(ix)) for ix, _ in self._indexes.values()) / 1e6, 1
        )

    def memory_mb(self) -> float:
        """Memory held for the vectors: the stored vectors, plus the indexes in fast mode."""
        arrays = [self._image, self._text_vectors, self._copies_matrix]
        return round(sum(a.nbytes for a in arrays) / 1e6 + self.index_mb(), 1)

    def candidate_rows(self, i: int, count: int | None = None) -> np.ndarray:
        """The shortlist for product `i`: its `count` nearest products by sentence vector
        and its `count` nearest by picture vector (fast mode).

        A vector part that the product itself does not have is left out of the shortlist,
        just as the hybrid skips that part and rescales the other weights.
        """
        if not self._indexes:
            raise RuntimeError("candidates need fast mode")
        count = count or self.candidates
        found = []
        for name, matrix in self._vector_parts():
            if np.isnan(matrix[i]).any():
                continue
            index, rows = self._indexes[name]
            hits = index.search(matrix[i : i + 1].astype(np.float32), count + 1)[1][0]
            found.append(rows[hits[hits >= 0]])
        if not found:
            return np.empty(0, dtype=np.int64)
        rows = np.unique(np.concatenate(found))
        return rows[rows != i]

    # ---- ranking ----

    def _ranked_rows(self, i: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Product positions from best to worst for the given mode, and every score."""
        if mode == "fast":
            self._build_indexes()  # only the first fast search builds them
        rows = self.candidate_rows(i) if mode == "fast" else None
        if rows is None or len(rows) == 0:  # exact mode, or nothing to shortlist from
            scores = self.score_all(i)
            scores[self._hidden] = np.nan
            order = rank_indices(scores, self._id_rank, exclude=i)
            return order[~self._hidden[order]], scores
        rows = rows[~self._hidden[rows]]
        scores_here = self.combine(self.attribute_similarities(i, rows))
        scores = np.full(len(self.products), np.nan)
        scores[rows] = scores_here
        order = rank_indices(scores_here, self._id_rank[rows], exclude=-1)
        return rows[order][np.isfinite(scores_here[order])], scores

    def find_similar(self, product_id: str, num_similar: int, mode: str | None = None) -> list[str]:
        return [uid for uid, _ in self.find_similar_with_scores(product_id, num_similar, mode)]

    def describe(self, product_id: str, num_similar: int, mode: str | None = None) -> dict:
        ranked = self.find_similar_with_scores(product_id, num_similar, mode)
        query = self.products[self.resolve(product_id)]
        results = [{**self._summary(self.products[self._row[u]]), "score": sc} for u, sc in ranked]
        return {"query": self._summary(query), "results": results}

    def find_similar_with_scores(
        self, product_id: str, num_similar: int, mode: str | None = None
    ) -> list[tuple[str, float]]:
        """The best matches as (ID, score), one copy of each item, best first.

        `mode` ("exact" or "fast") overrides the engine's mode for this search only.
        """
        mode = mode or self.mode
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if (
            not isinstance(num_similar, Integral)
            or isinstance(num_similar, bool)
            or num_similar < 1
        ):
            raise ValueError("num_similar must be a whole number of at least 1")
        k = int(num_similar)
        i = self.resolve(product_id)
        order, scores = self._ranked_rows(i, mode)
        copies, links = self._copies_matrix, self._copies_links
        depth = max(60, 6 * k)
        while True:
            kept, _ = pick_distinct(order[:depth].tolist(), copies, links, i, k)
            if len(kept) == k or depth >= len(order):
                break
            depth *= 2
        return [(str(self.ids[j]), round(float(scores[j]), 6)) for j in kept]
