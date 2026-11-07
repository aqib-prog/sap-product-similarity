"""The half-size vector files: saving, loading, and refusing damaged or wrong files."""

import json

import numpy as np
import pytest

from product_similarity.data import canonicalize
from product_similarity.preprocessing import build_products
from product_similarity.vector_files import (
    MANIFEST_FILE,
    PICTURE_FILE,
    export_vector_artifact,
    load_vector_artifact,
)


@pytest.fixture
def products(make_row):
    rows = [
        make_row(uid, f"Cotton shirt {uid}", medium=f"https://example.com/{uid}.jpg")
        for uid in ("a", "b", "c")
    ]
    kept, _ = canonicalize(rows)
    built, _ = build_products(kept)
    return built


@pytest.fixture
def folders(tmp_path, products):
    """Full-size vectors in `source` (as made offline) and an empty `out` folder."""
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    rng = np.random.default_rng(3)
    clip = rng.normal(size=(3, 6)).astype(np.float32) * 5  # not scaled to length 1
    text = rng.normal(size=(3, 4)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    urls = [p.image_url for p in products]
    np.savez(source / "all_clip.npz", urls=np.asarray(urls), vectors=clip)
    np.savez(
        source / "text_minilm.npz", ids=np.asarray([p.uniq_id for p in products]), vectors=text
    )
    return source, out


def test_round_trip_keeps_the_numbers_close(products, folders):
    source, out = folders
    manifest = export_vector_artifact(source, out)
    assert set(manifest["files"]) == {"picture_float16.npz", "sentence_float16.npz"}
    vectors = load_vector_artifact(products, out)
    assert vectors is not None
    assert vectors.picture.dtype == np.float32 and vectors.sentence.dtype == np.float32
    original = np.load(source / "text_minilm.npz")["vectors"]
    np.testing.assert_allclose(vectors.sentence, original, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(vectors.picture, axis=1), 1.0, atol=1e-3)


def test_the_files_are_half_the_size_of_float32(folders):
    source, out = folders
    export_vector_artifact(source, out)
    assert np.load(out / PICTURE_FILE)["vectors"].dtype == np.float16


def test_a_photo_marked_as_not_a_photo_gets_no_picture_vector(products, folders, tmp_path):
    source, out = folders
    excluded = tmp_path / "excluded.json"
    excluded.write_text(json.dumps([{"url": products[1].image_url, "why": "logo"}]))
    export_vector_artifact(source, out, excluded)
    vectors = load_vector_artifact(products, out)
    assert np.isnan(vectors.picture[1]).all()
    assert not np.isnan(vectors.picture[0]).any()


def test_missing_folder_gives_none(products, tmp_path):
    assert load_vector_artifact(products, tmp_path / "nothing") is None


def test_a_damaged_file_is_refused(products, folders):
    source, out = folders
    export_vector_artifact(source, out)
    with (out / PICTURE_FILE).open("ab") as handle:
        handle.write(b"damage")
    with pytest.raises(RuntimeError, match="SHA-256"):
        load_vector_artifact(products, out)


def test_a_missing_file_is_refused(products, folders):
    source, out = folders
    export_vector_artifact(source, out)
    (out / PICTURE_FILE).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        load_vector_artifact(products, out)


def test_an_unknown_format_is_refused(products, folders):
    source, out = folders
    export_vector_artifact(source, out)
    manifest = json.loads((out / MANIFEST_FILE).read_text())
    manifest["format"] = "something-else"
    (out / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="format"):
        load_vector_artifact(products, out)


def test_vectors_of_other_products_are_not_used(products, folders, make_row):
    source, out = folders
    export_vector_artifact(source, out)
    kept, _ = canonicalize([make_row("z", "Other", medium="https://example.com/z.jpg")])
    other, _ = build_products(kept)
    assert load_vector_artifact(other, out) is None
