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
    # The half-size copies of the vectors that ship with the service (see vector_files.py).
    vectors_dir: Path = Path(os.environ.get("PS_VECTORS_DIR", "artifacts/vectors"))
    # Stop with an error when no vectors are found, instead of using the baseline search.
    require_vectors: bool = os.environ.get("PS_REQUIRE_VECTORS", "") == "1"
    # The vectors as they were made offline (full-size numbers); only on a development machine.
    embeddings_dir: Path = Path(os.environ.get("PS_EMBEDDINGS_DIR", "data/embeddings"))
    # Pictures that are not product photos (a shop logo, a drawing); optional.
    not_photos: Path = Path(os.environ.get("PS_NOT_PHOTOS", "data/images/shared/excluded.json"))
    # How many answers the in-process cache keeps (0 turns it off).
    cache_size: int = int(os.environ.get("PS_CACHE_SIZE", "1024"))
    # "exact" or "fast" (the tuned hybrid), or "baseline" (title and attributes only).
    search_mode: str = os.environ.get("PS_SEARCH_MODE", "exact")


def get_settings() -> Settings:
    """Return settings read from the environment at call time."""
    return Settings(
        data_archive=Path(os.environ.get("PS_DATA_ARCHIVE", "data/archive.zip")),
        artifact_dir=Path(os.environ.get("PS_ARTIFACT_DIR", "artifacts")),
        vectors_dir=Path(os.environ.get("PS_VECTORS_DIR", "artifacts/vectors")),
        require_vectors=os.environ.get("PS_REQUIRE_VECTORS", "") == "1",
        embeddings_dir=Path(os.environ.get("PS_EMBEDDINGS_DIR", "data/embeddings")),
        not_photos=Path(os.environ.get("PS_NOT_PHOTOS", "data/images/shared/excluded.json")),
        cache_size=int(os.environ.get("PS_CACHE_SIZE", "1024")),
        search_mode=os.environ.get("PS_SEARCH_MODE", "exact"),
    )
