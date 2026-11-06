"""Turn product titles into lists of numbers with a pretrained sentence model.

Offline tooling only: the web service never imports this module. The model is used
as it is (no training) and is loaded on first use, so importing this file does not
need torch.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from product_similarity.image_features import normalise

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEncoder:
    """A pretrained model that maps a sentence to 384 numbers."""

    def __init__(self, name: str = TEXT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = SentenceTransformer(name, device="cpu")

    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        """Vectors of length 1, one row per text, so a dot product is the cosine."""
        vectors = self.model.encode(
            list(texts), batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        return normalise(vectors)


def embed_titles(encoder: TextEncoder, titles: Sequence[str]) -> tuple[np.ndarray, dict]:
    """Encode all titles and report the time it took. Empty titles get a space."""
    started = time.monotonic()
    vectors = encoder.encode([t if t and t.strip() else " " for t in titles])
    seconds = time.monotonic() - started
    return vectors, {
        "titles": len(titles),
        "dimensions": int(vectors.shape[1]),
        "seconds": round(seconds, 1),
        "ms_per_title": round(1000 * seconds / max(len(titles), 1), 3),
    }
