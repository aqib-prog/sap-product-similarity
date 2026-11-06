"""Choose the weights of the hybrid search from the human ratings.

Coordinate ascent changes one weight at a time and keeps the value that gives the best
score, then moves on to the next weight, and goes around again until nothing improves
(Metzler and Croft, 2007). It is started from several points and the best result is
kept. Cross-validation checks that the weights also work on products they were not
tuned on. Nothing here knows about products: it only needs a function that turns a set
of weights into a score, so it is easy to test.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

Weights = dict[str, float]
ScoreFn = Callable[[Weights], float]

GRID = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)


@dataclass
class TuneResult:
    weights: Weights
    score: float
    evaluations: int
    rounds: int
    history: list[float] = field(default_factory=list)


def normalise(weights: Mapping[str, float]) -> Weights:
    """Scale the weights so that they add up to 1."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("at least one weight must be above 0")
    return {name: value / total for name, value in weights.items()}


def coordinate_ascent(
    score: ScoreFn,
    start: Mapping[str, float],
    grid: Sequence[float] = GRID,
    max_rounds: int = 10,
    tolerance: float = 1e-9,
) -> TuneResult:
    """Improve `start` one weight at a time until a full round changes nothing."""
    weights = dict(start)
    best = score(weights)
    evaluations = 1
    history = [best]
    rounds = 0
    for _ in range(max_rounds):
        rounds += 1
        changed = False
        for name in weights:
            candidates = sorted({*grid, weights[name]})
            results = []
            for value in candidates:
                trial = {**weights, name: value}
                if sum(trial.values()) <= 0:
                    continue
                results.append((score(trial), value))
                evaluations += 1
            # Best score first; on a tie keep the smaller weight (the simpler mix).
            top_score, top_value = min(results, key=lambda r: (-r[0], r[1]))
            if top_score > best + tolerance and top_value != weights[name]:
                weights[name], best, changed = top_value, top_score, True
        history.append(best)
        if not changed:
            break
    return TuneResult(weights, best, evaluations, rounds, history)


def random_start(names: Sequence[str], seed: int, grid: Sequence[float] = GRID) -> Weights:
    """A random starting point on the grid (never all zeros)."""
    rng = random.Random(seed)
    while True:
        weights = {name: rng.choice(grid) for name in names}
        if sum(weights.values()) > 0:
            return weights


def tune(score: ScoreFn, starts: Sequence[Mapping[str, float]], **options) -> TuneResult:
    """Run coordinate ascent from every start and keep the best result."""
    results = [coordinate_ascent(score, start, **options) for start in starts]
    best = max(results, key=lambda r: r.score)
    best.evaluations = sum(r.evaluations for r in results)
    return best


def make_folds(items: Sequence[str], n_folds: int, seed: int) -> list[list[str]]:
    """Split the items into n_folds groups of nearly equal size, the same way every time."""
    if not 2 <= n_folds <= len(items):
        raise ValueError("n_folds must be between 2 and the number of items")
    shuffled = sorted(items)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i::n_folds] for i in range(n_folds)]


def cross_validate(
    score_on: Callable[[Weights, Sequence[str]], float],
    queries: Sequence[str],
    starts: Sequence[Mapping[str, float]],
    reference: Mapping[str, float],
    n_folds: int = 5,
    seed: int = 2020,
    **options,
) -> list[dict]:
    """Tune on all folds but one and score on the fold that was left out.

    `reference` (for example the untuned weights) is scored on the same left-out
    products, so the tuned weights are compared with it fairly.
    """
    rows = []
    folds = make_folds(queries, n_folds, seed)
    for number, test in enumerate(folds, start=1):
        train = [q for q in queries if q not in set(test)]
        result = tune(lambda w, train=train: score_on(w, train), starts, **options)
        rows.append(
            {
                "fold": number,
                "train_products": len(train),
                "test_products": len(test),
                "train_score": result.score,
                "test_score": score_on(result.weights, test),
                "reference_test_score": score_on(dict(reference), test),
                "weights": normalise(result.weights),
            }
        )
    return rows


def importance(score: ScoreFn, weights: Mapping[str, float]) -> list[dict]:
    """How much the score drops when each part is removed, biggest drop first."""
    full = score(dict(weights))
    rows = []
    for name, value in weights.items():
        if value <= 0:
            continue
        without = {**weights, name: 0.0}
        if sum(without.values()) <= 0:
            continue
        rows.append({"part": name, "weight": value, "score_without": score(without)})
    for row in rows:
        row["drop"] = full - row["score_without"]
    return sorted(rows, key=lambda r: -r["drop"])
