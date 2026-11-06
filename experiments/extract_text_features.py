"""Turn every product title into a vector with a pretrained sentence model.

Run from the project folder:  python experiments/extract_text_features.py
The model (about 90 MB) is downloaded once from Hugging Face. Vectors go to
data/embeddings/text_minilm.npz (ignored by git); only the timings are written to results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sentence_transformers

from product_similarity.service import default_engine
from product_similarity.text_features import TEXT_MODEL, TextEncoder, embed_titles

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "embeddings" / "text_minilm.npz"
SPEED = ROOT / "results" / "text_embedding_speed.json"


def main() -> None:
    engine = default_engine()
    products = engine.products
    encoder = TextEncoder()
    vectors, report = embed_titles(encoder, [p.title or "" for p in products])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, ids=np.array([p.uniq_id for p in products]), vectors=vectors)
    report.update(
        model=TEXT_MODEL,
        sentence_transformers=sentence_transformers.__version__,
        empty_titles=sum(1 for p in products if not (p.title or "").strip()),
        longest_title_words=max(len((p.title or "").split()) for p in products),
    )
    SPEED.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
