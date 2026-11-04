"""Loader and validator tests using a tiny hand-made archive."""

import json
import zipfile

import pytest

from product_similarity.data import (
    archive_sha256,
    check_integrity,
    load_products,
    parse_product,
)
from product_similarity.validation import validate_products

ID_A = "a" * 32
ID_B = "b" * 32


def _row(uniq_id, asin="B000000001", **extra):
    base = {
        "uniq_id": uniq_id,
        "asin": asin,
        "product_name": "Test Shirt",
        "sales_price": "1,299.00",
        "rating": "4.5",
        "weight": "999999999",
        "medium": "https://example.test/a.jpg",
    }
    base.update(extra)
    return base


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "sample.zip"
    lines = [json.dumps(_row(ID_A)), json.dumps(_row(ID_B, asin="B000000002", sales_price=None))]
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("sample.ldjson", "\n".join(lines) + "\n")
    return path


def test_parse_product_converts_numbers_and_keeps_weight_raw():
    row = parse_product(json.dumps(_row(ID_A)))
    assert row["sales_price"] == 1299.0
    assert row["rating"] == 4.5
    assert row["weight"] == "999999999"


def test_parse_product_handles_missing_and_invalid_numbers():
    row = parse_product(json.dumps(_row(ID_A, sales_price=None, rating="n/a")))
    assert row["sales_price"] is None
    assert row["rating"] is None


def test_load_products_reads_archive_in_place(archive):
    rows = load_products(archive)
    assert [r["uniq_id"] for r in rows] == [ID_A, ID_B]


def test_load_products_rejects_archive_without_ldjson(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    with pytest.raises(ValueError):
        load_products(path)


def test_integrity_check_detects_other_files(archive):
    assert len(archive_sha256(archive)) == 64
    assert check_integrity(archive) is False


def test_validation_clean_data(archive):
    report = validate_products(load_products(archive))
    assert report.is_usable
    assert report.weight_sentinel_rows == 2
    assert report.rows_without_price == 1
    assert report.rows_without_image == 0


def test_validation_flags_duplicates_and_bad_ids():
    rows = [_row(ID_A), _row(ID_A), _row("XYZ", product_name="")]
    report = validate_products(rows)
    assert report.duplicate_uniq_ids == 1
    assert report.malformed_uniq_ids == 1
    assert report.missing_required["product_name"] == 1
    assert report.duplicate_asin_extra_rows == 2
    assert not report.is_usable
