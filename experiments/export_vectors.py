"""Save the picture and sentence vectors as the half-size files that ship with the service.

Run from the project folder:  python experiments/export_vectors.py
Reads data/embeddings/all_clip.npz and data/embeddings/text_minilm.npz (made offline) and
writes artifacts/vectors/: picture_float16.npz, sentence_float16.npz and manifest.json. Every
number is stored in 2 bytes instead of 4, so each file stays under 50 MB (GitHub warns above
that). check_vector_files.py then checks that the results do not change.
"""

from __future__ import annotations

import json
from pathlib import Path

from product_similarity.vector_files import export_vector_artifact

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = export_vector_artifact(
        ROOT / "data" / "embeddings",
        ROOT / "artifacts" / "vectors",
        ROOT / "data" / "images" / "shared" / "excluded.json",
    )
    for name, info in manifest["files"].items():
        print(f"{name}: {info['bytes'] / 1e6:.1f} MB")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
