"""Read the Amazon Fashion dataset directly from the source archive."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

# SHA-256 of the data/archive.zip supplied with the assignment.
EXPECTED_ARCHIVE_SHA256 = "6cd0dfa83a311533c6c991cc6586d13cc4e94e7d6fb799d2d00e5aa2a4bf8f10"

# Fields stored as strings in the source file that are really numbers.
_NUMERIC_FIELDS = ("sales_price", "rating", "no__of_reviews", "no__of_offers", "no__of_sellers")


def archive_sha256(archive: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(archive, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_integrity(archive: Path) -> bool:
    """True when the archive is byte-identical to the one supplied by SAP."""
    return archive_sha256(archive) == EXPECTED_ARCHIVE_SHA256


def _to_number(value: Any) -> float | None:
    """Convert a numeric string such as '1,234.00' to a float, or None if empty or invalid."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def parse_product(line: str) -> dict[str, Any]:
    """Parse one JSON line and convert known numeric string fields to numbers.

    Every other field, including `weight` and its sentinel value, is kept as
    the source supplied it. Cleaning belongs to the preprocessing step.
    """
    row = json.loads(line)
    for name in _NUMERIC_FIELDS:
        if name in row:
            row[name] = _to_number(row[name])
    return row


def load_products(archive: Path) -> list[dict[str, Any]]:
    """Read all products from the first .ldjson file inside the archive.

    The archive is read in place. Nothing is extracted to disk.
    """
    with zipfile.ZipFile(archive) as zf:
        members = [n for n in zf.namelist() if n.endswith(".ldjson")]
        if not members:
            raise ValueError(f"No .ldjson file found in {archive}")
        with zf.open(members[0]) as handle:
            return [parse_product(line.decode("utf-8")) for line in handle if line.strip()]


def canonicalize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Keep one row per ASIN and map every `uniq_id` to its kept row.

    Rows with the same ASIN are recrawls of one product. The kept row is the
    most recently crawled one; if crawl times tie, the lowest `uniq_id` wins.
    A row without an ASIN is its own product. The input rows are not changed.

    Returns the kept rows (in their original order) and a mapping that sends
    every original `uniq_id`, kept or not, to the `uniq_id` of its kept row.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get("asin") or row["uniq_id"], []).append(row)

    kept_ids: dict[str, str] = {}
    id_map: dict[str, str] = {}
    for key, group in groups.items():
        ordered = sorted(group, key=lambda r: r["uniq_id"])
        best = max(ordered, key=lambda r: r.get("crawl_timestamp") or "")
        kept_ids[key] = best["uniq_id"]
        for row in group:
            id_map[row["uniq_id"]] = best["uniq_id"]

    kept = {uid for uid in kept_ids.values()}
    return [row for row in rows if row["uniq_id"] in kept], id_map
