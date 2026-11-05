"""Turn product pictures into lists of numbers with pretrained models.

Offline tooling only: the web service never imports this module. The models are
used as they are (no training). ResNet-50 and CLIP are loaded on first use, so
importing this file does not need torch.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MODELS = ("resnet50", "clip")
CLIP_NAME = "openai/clip-vit-base-patch32"


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product is the cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(lengths == 0, 1.0, lengths)


def top_k(vectors: np.ndarray, query_index: int, k: int = 10) -> list[int]:
    """Row numbers of the k closest rows to one row, excluding itself.

    Ties are broken by row number so the answer is always the same.
    """
    scores = vectors @ vectors[query_index]
    scores[query_index] = -np.inf
    order = sorted(range(len(scores)), key=lambda i: (-round(float(scores[i]), 6), i))
    return order[:k]


class ImageEncoder:
    """One pretrained model that maps a picture to a vector."""

    def __init__(self, name: str) -> None:
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}; choose from {MODELS}")
        import torch

        self.name = name
        self.torch = torch
        self.device = "cpu"
        torch.set_grad_enabled(False)
        if name == "resnet50":
            from torchvision import models

            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
            model.fc = torch.nn.Identity()  # keep the 2048 numbers before the classifier
            self.model = model.eval()
            self.transform = weights.transforms()
        else:
            from transformers import CLIPModel, CLIPProcessor

            self.model = CLIPModel.from_pretrained(CLIP_NAME).eval()
            self.processor = CLIPProcessor.from_pretrained(CLIP_NAME)

    def encode(self, images: Sequence) -> np.ndarray:
        """Vectors (one row per PIL image), already scaled to length 1."""
        torch = self.torch
        if self.name == "resnet50":
            batch = torch.stack([self.transform(image) for image in images])
            out = self.model(batch)
        else:
            inputs = self.processor(images=list(images), return_tensors="pt")
            out = self.model.get_image_features(**inputs)
        return normalise(out.numpy())


def load_rgb(path: Path):
    """Open a picture as RGB. Raises if the file cannot be decoded."""
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def embed_paths(
    encoder: ImageEncoder, paths: Sequence[Path], batch_size: int = 16
) -> tuple[np.ndarray, list[int], dict]:
    """Vectors for the pictures that open, the row numbers that failed, and timings."""
    vectors: list[np.ndarray] = []
    failed: list[int] = []
    pending: list = []
    started = time.monotonic()
    for number, path in enumerate(paths):
        try:
            pending.append((number, load_rgb(path)))
        except Exception:  # noqa: BLE001 - a bad picture is a result, not a crash
            failed.append(number)
        if len(pending) == batch_size:
            vectors.append(encoder.encode([image for _, image in pending]))
            pending = []
    if pending:
        vectors.append(encoder.encode([image for _, image in pending]))
    seconds = time.monotonic() - started
    done = len(paths) - len(failed)
    timings = {
        "images": done,
        "seconds_total": round(seconds, 2),
        "ms_per_image": round(1000 * seconds / max(done, 1), 1),
    }
    return (np.concatenate(vectors) if vectors else np.zeros((0, 0), np.float32)), failed, timings
