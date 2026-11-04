"""Check loaded products and report problems without changing the data."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

WEIGHT_SENTINEL = "999999999"
REQUIRED_FIELDS = ("uniq_id", "asin", "product_name")
_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


@dataclass
class ValidationReport:
    """Counts of data problems found in a list of products."""

    n_rows: int = 0
    duplicate_uniq_ids: int = 0
    malformed_uniq_ids: int = 0
    missing_required: dict[str, int] = field(default_factory=dict)
    duplicate_asin_extra_rows: int = 0
    weight_sentinel_rows: int = 0
    rows_without_image: int = 0
    rows_without_price: int = 0

    @property
    def is_usable(self) -> bool:
        """True when IDs are unique and well formed and required fields are present."""
        return (
            self.n_rows > 0
            and self.duplicate_uniq_ids == 0
            and self.malformed_uniq_ids == 0
            and not any(self.missing_required.values())
        )


def validate_products(rows: list[dict[str, Any]]) -> ValidationReport:
    """Return a ValidationReport. Does not modify or drop any row."""
    report = ValidationReport(n_rows=len(rows))
    ids = Counter(row.get("uniq_id") for row in rows)
    report.duplicate_uniq_ids = sum(count - 1 for count in ids.values() if count > 1)
    report.malformed_uniq_ids = sum(
        1 for row in rows if not _ID_PATTERN.fullmatch(str(row.get("uniq_id", "")))
    )
    report.missing_required = {
        name: sum(1 for row in rows if not row.get(name)) for name in REQUIRED_FIELDS
    }
    asins = Counter(row.get("asin") for row in rows if row.get("asin"))
    report.duplicate_asin_extra_rows = sum(count - 1 for count in asins.values() if count > 1)
    report.weight_sentinel_rows = sum(1 for row in rows if row.get("weight") == WEIGHT_SENTINEL)
    report.rows_without_image = sum(1 for row in rows if not row.get("medium"))
    report.rows_without_price = sum(1 for row in rows if row.get("sales_price") is None)
    return report
