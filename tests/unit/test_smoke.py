"""Stage 0 checks: the package imports and configuration behaves."""

from pathlib import Path

import product_similarity
from product_similarity.settings import get_settings


def test_package_has_version():
    assert product_similarity.__version__ == "0.1.0"


def test_default_paths():
    settings = get_settings()
    assert settings.data_archive == Path("data/archive.zip")
    assert settings.artifact_dir == Path("artifacts")


def test_paths_can_be_overridden(monkeypatch):
    monkeypatch.setenv("PS_ARTIFACT_DIR", "/tmp/custom")
    assert get_settings().artifact_dir == Path("/tmp/custom")
