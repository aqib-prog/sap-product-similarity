"""Cleaning and derived fields for product records.

Nothing here edits the raw rows. Each function reads a value and returns a
cleaned one, and a missing or unusable value stays missing (None).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from product_similarity.validation import WEIGHT_SENTINEL

# A garment outside this range is treated as a data error, not a real weight.
MIN_WEIGHT_G = 1.0
MAX_WEIGHT_G = 5000.0
_GRAMS_PER_UNIT = {
    "g": 1.0,
    "gm": 1.0,
    "gms": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kgs": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
}

BRAND_PLACEHOLDERS = {"", "unknown", "n/a", "na", "none", "null", "-"}
CATCH_ALL_CATEGORIES = {"Fashion"}

# Fixed colour vocabulary. Longer phrases are listed first so they win.
_COLOUR_WORDS = {
    "navy blue": "navy",
    "royal blue": "blue",
    "sky blue": "blue",
    "light blue": "blue",
    "dark blue": "blue",
    "navy": "navy",
    "black": "black",
    "white": "white",
    "blue": "blue",
    "red": "red",
    "green": "green",
    "yellow": "yellow",
    "pink": "pink",
    "grey": "grey",
    "gray": "grey",
    "brown": "brown",
    "beige": "beige",
    "maroon": "maroon",
    "purple": "purple",
    "orange": "orange",
    "gold": "gold",
    "silver": "silver",
    "cream": "cream",
    "peach": "peach",
    "olive": "olive",
    "teal": "teal",
    "turquoise": "turquoise",
    "khaki": "khaki",
    "mustard": "mustard",
    "wine": "wine",
}
_COLOUR_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _COLOUR_WORDS), key=len, reverse=True)) + r")\b"
)

_CATEGORY_AUDIENCE = (
    ("Mens", "men"),
    ("Womens", "women"),
    ("Boys", "boys"),
    ("Girls", "girls"),
    ("Baby", "baby"),
    ("Kids", "kids"),
)
_AUDIENCE_WORDS = (
    (re.compile(r"\b(mens?|man)\b"), "men"),
    (re.compile(r"\b(womens?|woman|ladies|lady)\b"), "women"),
    (re.compile(r"\b(boys?)\b"), "boys"),
    (re.compile(r"\b(girls?)\b"), "girls"),
    (re.compile(r"\b(baby|babies|infants?|toddlers?|newborn)\b"), "baby"),
    (re.compile(r"\b(kids?|children)\b"), "kids"),
)
_REVIEW_PATTERN = re.compile(r"([\d.]+) out of 5 stars(?:\s+([\d,]+))?")


# ---- weight ---------------------------------------------------------------


def parse_weight(raw: Any) -> tuple[float | None, str | None]:
    """Return (grams, problem). Exactly one of the two is None.

    The placeholder 999999999 is missing, not a weight. Units are converted to
    grams, and values outside the plausible range are rejected with a reason.
    """
    if raw is None or str(raw).strip() == WEIGHT_SENTINEL:
        return None, "placeholder"
    text = str(raw).replace("\xa0", " ").strip()
    match = re.fullmatch(r"([\d.,]+)\s*([A-Za-z]+)", text)
    if not match:
        return None, "not_a_weight"
    unit = match.group(2).lower()
    if unit not in _GRAMS_PER_UNIT:
        return None, "unknown_unit"
    try:
        grams = float(match.group(1).replace(",", "")) * _GRAMS_PER_UNIT[unit]
    except ValueError:
        return None, "not_a_weight"
    if not MIN_WEIGHT_G <= grams <= MAX_WEIGHT_G:
        return None, "out_of_range"
    return grams, None


# ---- text -----------------------------------------------------------------


def _repair_mojibake(text: str) -> str:
    """Undo text that was decoded with the wrong encoding, such as 'â€™' for an apostrophe.

    Titles are broken in two ways in this dataset (cp1252 and latin-1), and a few
    are broken twice, so up to two passes are tried. Text that does not repair
    cleanly is returned unchanged.
    """
    for _ in range(2):
        if not any(mark in text for mark in ("Ã", "Â", "â")):
            break
        for encoding in ("cp1252", "latin-1"):
            try:
                text = text.encode(encoding).decode("utf-8")
                break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        else:
            break
    return text


def clean_title(raw: str) -> str:
    """Fix spacing and broken characters. Sizes, SKUs and pack counts are kept."""
    text = _repair_mojibake(unicodedata.normalize("NFC", str(raw)))
    return " ".join(text.replace("\xa0", " ").split())


# ---- brand ----------------------------------------------------------------


def clean_brand(raw: Any) -> str | None:
    """Trim spacing and turn placeholders such as 'Unknown' into None."""
    if raw is None:
        return None
    text = " ".join(str(raw).replace("\xa0", " ").split())
    return None if text.lower() in BRAND_PLACEHOLDERS else text


def _brand_key(brand: str) -> str:
    return re.sub(r"[^a-z0-9]", "", brand.lower())


def build_brand_map(brands: Iterable[str]) -> dict[str, str]:
    """Map each brand spelling to the most common spelling of the same name.

    Spellings that differ only in case, spaces or punctuation are merged
    ('Print Mafias' and 'print mafias'). Ties go to the alphabetically first.
    """
    counts = Counter(brands)
    groups: dict[str, list[str]] = defaultdict(list)
    for spelling in counts:
        key = _brand_key(spelling)
        if key:
            groups[key].append(spelling)
    mapping = {}
    for spellings in groups.values():
        best = sorted(spellings, key=lambda s: (-counts[s], s))[0]
        for spelling in spellings:
            mapping[spelling] = best
    return mapping


def build_brand_lookup(brands: Iterable[str]) -> dict[str, str]:
    """Map a punctuation-free lowercase key to a brand, for title matching."""
    return {_brand_key(b): b for b in brands if _brand_key(b)}


def recover_brand_from_title(title: str, lookup: dict[str, str]) -> str | None:
    """Return a known brand when the title starts with it, else None.

    Tries the first four words, longest first. `lookup` comes from
    `build_brand_lookup`. Not used unless switched on.
    """
    words = title.split()
    for size in range(min(4, len(words)), 0, -1):
        found = lookup.get(_brand_key("".join(words[:size])))
        if found:
            return found
    return None


# ---- category -------------------------------------------------------------


def primary_category(row: dict[str, Any]) -> str | None:
    """The leaf category from the sales-rank field, or None for missing or catch-all."""
    ranks = row.get("sales_rank_in_child_category")
    if not isinstance(ranks, dict) or not ranks:
        return None
    name = next(iter(ranks))
    return None if name in CATCH_ALL_CATEGORIES else name


def build_node_category_map(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    """For each browse node, the category most products in it have."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        category, node = primary_category(row), row.get("browsenode")
        if category and node:
            votes[node][category] += 1
    return {
        node: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for node, c in votes.items()
    }


# ---- reviews, price, image ------------------------------------------------


def parse_page_reviews(row: dict[str, Any]) -> tuple[float | None, int | None, str]:
    """Read (rating, review count, state) from the page's review sentence.

    State is 'none' (nobody has reviewed it), 'counted' or 'unknown'.
    """
    details = row.get("product_details__k_v_pairs")
    text = details.get("Customer_Reviews", "") if isinstance(details, dict) else ""
    if text.startswith("Be the first"):
        return None, 0, "none"
    match = _REVIEW_PATTERN.match(text)
    if not match or match.group(2) is None:
        return None, None, "unknown"
    return float(match.group(1)), int(match.group(2).replace(",", "")), "counted"


def price_fields(row: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return (price, log10 price). Both None when the price is missing or not positive."""
    price = row.get("sales_price")
    if price is None or price <= 0:
        return None, None
    return price, math.log10(price)


def first_image_url(row: dict[str, Any]) -> str | None:
    """The first medium-size image URL, or None."""
    for url in (row.get("medium") or "").split("|"):
        if url.strip():
            return url.strip()
    return None


# ---- colour and audience --------------------------------------------------


def extract_colours(text: str | None) -> tuple[str, ...]:
    """Colours from a fixed vocabulary found in the text, sorted and without repeats."""
    if not text:
        return ()
    found = {_COLOUR_WORDS[m] for m in _COLOUR_PATTERN.findall(text.lower())}
    return tuple(sorted(found))


def derive_audience(title: str, category: str | None) -> str | None:
    """men, women, boys, girls, baby, kids, unisex (several groups named) or None.

    The category name decides first; the title is the fallback. When a title
    names several groups: men and women together mean unisex, baby wins over
    boys or girls, and boys or girls win over the general word kids.
    """
    for prefix, audience in _CATEGORY_AUDIENCE:
        if category and category.startswith(prefix):
            return audience
    text = title.lower().replace("'", "").replace("\u2019", "")
    found = {audience for pattern, audience in _AUDIENCE_WORDS if pattern.search(text)}
    if len(found) <= 1:
        return found.pop() if found else None
    if {"men", "women"} <= found:
        return "unisex"
    if "baby" in found:
        return "baby"
    if found & {"boys", "girls"}:
        found.discard("kids")
    return found.pop() if len(found) == 1 else "unisex"


# ---- the product record ---------------------------------------------------


@dataclass(frozen=True)
class Product:
    """One cleaned product. Missing values are None."""

    uniq_id: str
    asin: str
    title: str
    brand: str | None
    category: str | None
    category_source: str | None
    audience: str | None
    price: float | None
    log_price: float | None
    weight_g: float | None
    rating: float | None
    page_rating: float | None
    review_count: int | None
    review_state: str
    title_colours: tuple[str, ...]
    listed_colours: tuple[str, ...]
    image_url: str | None


@dataclass
class PreprocessReport:
    """Counts of what the pipeline changed or could not use."""

    n_products: int = 0
    brands_before: int = 0
    brands_after: int = 0
    brand_placeholders: int = 0
    brand_recovered: int = 0
    weight_problems: dict[str, int] = field(default_factory=dict)
    category_from_child: int = 0
    category_from_node: int = 0
    category_missing: int = 0
    audience_found: int = 0
    title_colour_found: int = 0
    listed_colour_found: int = 0


def build_products(
    rows: list[dict[str, Any]], recover_brands: bool = False
) -> tuple[list[Product], PreprocessReport]:
    """Turn canonical raw rows into cleaned Product records plus a report.

    Pass rows that have already been through `data.canonicalize`.
    Brand recovery from titles is off unless asked for.
    """
    report = PreprocessReport(n_products=len(rows))
    cleaned = [clean_brand(r.get("brand")) for r in rows]
    report.brand_placeholders = sum(
        1 for r, c in zip(rows, cleaned) if r.get("brand") and c is None
    )
    named = [b for b in cleaned if b]
    report.brands_before = len(set(named))
    brand_map = build_brand_map(named)
    lookup = build_brand_lookup(set(brand_map.values()))
    node_map = build_node_category_map(rows)
    weight_problems: Counter = Counter()

    products = []
    for row, brand in zip(rows, cleaned):
        title = clean_title(row["product_name"])
        brand = brand_map.get(brand) if brand else None
        if brand is None and recover_brands:
            brand = recover_brand_from_title(title, lookup)
            report.brand_recovered += brand is not None

        category, source = primary_category(row), "child"
        if category is None:
            category = node_map.get(row.get("browsenode"))
            source = "node" if category else None
        if source == "child":
            report.category_from_child += 1
        elif source == "node":
            report.category_from_node += 1
        else:
            report.category_missing += 1

        grams, problem = parse_weight(row.get("weight"))
        if problem and problem != "placeholder":
            weight_problems[problem] += 1
        price, log_price = price_fields(row)
        page_rating, review_count, review_state = parse_page_reviews(row)
        audience = derive_audience(title, category)
        title_colours = extract_colours(title)
        listed_colours = extract_colours((row.get("colour") or "").replace("|", " "))
        report.audience_found += audience is not None
        report.title_colour_found += bool(title_colours)
        report.listed_colour_found += bool(listed_colours)

        products.append(
            Product(
                uniq_id=row["uniq_id"],
                asin=row.get("asin") or "",
                title=title,
                brand=brand,
                category=category,
                category_source=source,
                audience=audience,
                price=price,
                log_price=log_price,
                weight_g=grams,
                rating=row.get("rating"),
                page_rating=page_rating,
                review_count=review_count,
                review_state=review_state,
                title_colours=title_colours,
                listed_colours=listed_colours,
                image_url=first_image_url(row),
            )
        )
    report.brands_after = len({p.brand for p in products if p.brand})
    report.weight_problems = dict(weight_problems)
    return products, report
