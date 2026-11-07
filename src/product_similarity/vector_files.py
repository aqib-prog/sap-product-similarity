"""The picture and sentence vectors as they ship with the service: half-size numbers.

The vectors are made once, offline (CLIP for the pictures, MiniLM for the titles). This
module saves them in two files with every number stored in 2 bytes (float16) instead of 4,
so each file stays under 50 MB, and reads them back. A manifest records the SHA-256 of each
file, so a damaged or wrong file is noticed when the service starts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from product_similarity.hybrid import FinalVectors, assemble_final_vectors
from product_similarity.preprocessing import Product

logger = logging.getLogger(__name__)

FORMAT = "product-similarity-vectors-v1"
PICTURE_FILE = "picture_float16.npz"
SENTENCE_FILE = "sentence_float16.npz"
MANIFEST_FILE = "manifest.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_vector_artifact(
    embeddings_dir: Path, out_dir: Path, not_photos: Path | None = None
) -> dict:
    """Write the two half-size files and the manifest. Returns the manifest."""
    clip = np.load(embeddings_dir / "all_clip.npz")
    text = np.load(embeddings_dir / "text_minilm.npz")
    unit = clip["vectors"] / np.linalg.norm(clip["vectors"], axis=1, keepdims=True)
    excluded: set[str] = set()
    if not_photos is not None and not_photos.exists():
        excluded = {r["url"] for r in json.loads(not_photos.read_text(encoding="utf-8"))}
    urls = np.asarray([str(u) for u in clip["urls"]])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / PICTURE_FILE,
        urls=urls,
        not_photo=np.array([u in excluded for u in urls]),
        vectors=unit.astype(np.float16),
    )
    np.savez_compressed(
        out_dir / SENTENCE_FILE,
        ids=np.asarray([str(i) for i in text["ids"]]),
        vectors=text["vectors"].astype(np.float16),
    )
    manifest = {
        "format": FORMAT,
        "number_format": "float16 (picture vectors scaled to length 1 before saving)",
        "files": {
            name: {
                "sha256": sha256_of(out_dir / name),
                "bytes": (out_dir / name).stat().st_size,
            }
            for name in (PICTURE_FILE, SENTENCE_FILE)
        },
        "rows": {"picture": int(len(urls)), "sentence": int(len(text["ids"]))},
        "dimensions": {"picture": int(unit.shape[1]), "sentence": int(text["vectors"].shape[1])},
        "source": "CLIP (clip-vit-base-patch32) and MiniLM (all-MiniLM-L6-v2), computed offline",
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_vector_artifact(products: Sequence[Product], vectors_dir: Path) -> FinalVectors | None:
    """Read the half-size files (as normal 4-byte numbers), or None if they are not there
    or do not belong to these products. A file that does not match the manifest raises."""
    manifest_path = vectors_dir / MANIFEST_FILE
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise RuntimeError(f"unknown vector file format in {manifest_path}")
    for name, info in manifest["files"].items():
        path = vectors_dir / name
        if not path.exists():
            raise RuntimeError(f"vector file missing: {path}")
        if sha256_of(path) != info["sha256"]:
            raise RuntimeError(f"vector file {path} does not match its manifest (SHA-256)")
    picture = np.load(vectors_dir / PICTURE_FILE)
    sentence = np.load(vectors_dir / SENTENCE_FILE)
    return assemble_final_vectors(
        products,
        picture["urls"],
        picture["vectors"].astype(np.float32),
        set(picture["urls"][picture["not_photo"]]),
        sentence["ids"],
        sentence["vectors"],
    )
