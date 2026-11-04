"""Score ranked result lists against human ratings.

A rating is 0 (not a match), 1 (acceptable) or 2 (strong match). A result that
nobody rated counts as 0, and the number of such results is reported.
"""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

K = 10
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 2020


def load_grades(path: Path) -> dict[str, dict[str, int]]:
    """Read a downloaded ratings file into {query_no: {candidate_id: grade}}.

    Rows with an empty grade are left out (unrated). Anything other than 0, 1
    or 2 is an error.
    """
    grades: dict[str, dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            query = row["query_no"]
            grades.setdefault(query, {})
            value = (row["grade"] or "").strip()
            if value == "":
                continue
            if value not in ("0", "1", "2"):
                raise ValueError(f"invalid grade {value!r} for {query} {row['candidate_id']}")
            grades[query][row["candidate_id"]] = int(value)
    return grades


def dcg(grades: Sequence[int]) -> float:
    """Discounted cumulative gain: good results count more near the top."""
    return sum((2**g - 1) / math.log2(position + 2) for position, g in enumerate(grades))


def ndcg_at_k(ranked: Sequence[int], pool: Sequence[int], k: int = K) -> float | None:
    """nDCG@k: the list's gain divided by the best gain the pool could give.

    `ranked` holds the grades of the returned results in order, and `pool` holds
    every grade known for the query. Returns None when the pool has nothing to find.
    """
    ideal = dcg(sorted(pool, reverse=True)[:k])
    if ideal == 0:
        return None
    return dcg(list(ranked)[:k]) / ideal


def precision_at_k(ranked: Sequence[int], k: int = K, minimum: int = 1) -> float:
    """Share of the top k results with a grade of at least `minimum`."""
    return sum(1 for g in list(ranked)[:k] if g >= minimum) / k


def bootstrap_mean(
    values: Sequence[float], rounds: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float]:
    """Mean with a 95% range, from resampling the products with replacement."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(rounds))
    return sum(values) / n, means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def paired_difference(
    first: Mapping[str, float],
    second: Mapping[str, float],
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Mean of (first - second) over shared products, with a 95% range.

    If the range excludes zero, the difference is unlikely to be luck.
    """
    shared = sorted(set(first) & set(second))
    return bootstrap_mean([first[q] - second[q] for q in shared], rounds, seed)


def score_method(
    ranked_ids: Mapping[str, Sequence[str]],
    grades: Mapping[str, Mapping[str, int]],
    k: int = K,
) -> dict[str, dict]:
    """Per-product scores for one method.

    `ranked_ids` maps each query to its returned candidate IDs, best first.
    """
    scores: dict[str, dict] = {}
    for query, returned in ranked_ids.items():
        known = grades.get(query, {})
        ranked = [known.get(candidate, 0) for candidate in list(returned)[:k]]
        scores[query] = {
            "ndcg": ndcg_at_k(ranked, list(known.values()), k),
            "precision": precision_at_k(ranked, k, 1),
            "strong_precision": precision_at_k(ranked, k, 2),
            "unrated": sum(1 for candidate in list(returned)[:k] if candidate not in known),
        }
    return scores


def summarise(per_query: Mapping[str, Mapping]) -> dict[str, float]:
    """Mean scores with 95% ranges for one method."""
    summary: dict[str, float] = {}
    for name, label in (
        ("ndcg", "ndcg_at_10"),
        ("precision", "precision_at_10"),
        ("strong_precision", "strong_precision_at_10"),
    ):
        values = [q[name] for q in per_query.values() if q[name] is not None]
        mean, low, high = bootstrap_mean(values)
        summary[label] = mean
        summary[f"{label}_low"] = low
        summary[f"{label}_high"] = high
    summary["products_scored"] = sum(1 for q in per_query.values() if q["ndcg"] is not None)
    summary["unrated_in_top_10"] = sum(q["unrated"] for q in per_query.values())
    return summary
