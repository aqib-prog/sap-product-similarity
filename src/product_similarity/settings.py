"""Central configuration: file locations that the rest of the package relies on."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Paths used by the data loader and the artifact build.

    Each path can be overridden with an environment variable, which keeps
    Docker and Kubernetes configuration outside the code.
    """

    data_archive: Path = Path(os.environ.get("PS_DATA_ARCHIVE", "data/archive.zip"))
    artifact_dir: Path = Path(os.environ.get("PS_ARTIFACT_DIR", "artifacts"))
    # Where the saved picture and sentence vectors are (made offline by `experiments/`).
    embeddings_dir: Path = Path(os.environ.get("PS_EMBEDDINGS_DIR", "data/embeddings"))
    # Pictures that are not product photos (a shop logo, a drawing); optional.
    not_photos: Path = Path(os.environ.get("PS_NOT_PHOTOS", "data/images/shared/excluded.json"))
    # "exact" or "fast" (the tuned hybrid), or "baseline" (title and attributes only).
    search_mode: str = os.environ.get("PS_SEARCH_MODE", "exact")


def get_settings() -> Settings:
    """Return settings read from the environment at call time."""
    return Settings(
        data_archive=Path(os.environ.get("PS_DATA_ARCHIVE", "data/archive.zip")),
        artifact_dir=Path(os.environ.get("PS_ARTIFACT_DIR", "artifacts")),
        embeddings_dir=Path(os.environ.get("PS_EMBEDDINGS_DIR", "data/embeddings")),
        not_photos=Path(os.environ.get("PS_NOT_PHOTOS", "data/images/shared/excluded.json")),
        search_mode=os.environ.get("PS_SEARCH_MODE", "exact"),
    )
