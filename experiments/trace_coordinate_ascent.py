"""Show coordinate ascent step by step on the four weights, with real numbers.

It runs the same tuning as `tune_hybrid_weights.py` (the same ratings and the same five
starting points), and records what happens: round 1 from the first starting point, and where
each of the five starting points ends. The README uses these numbers to explain the method.

    python experiments/trace_coordinate_ascent.py
"""

from __future__ import annotations

import json

from hybrid_setup import ROOT, SEED, build_evaluator, expand_four

from product_similarity import tuning

N_RANDOM_STARTS = 2  # the same as tune_hybrid_weights.py
GROUPS = ("title", "text_vectors", "image", "attributes")
FIRST_GUESS = {"title": 0.35, "text_vectors": 0.0, "image": 0.0, "attributes": 0.65}


def round_one(score, start: dict[str, float]) -> list[dict]:
    """Round 1 of coordinate ascent: for each weight, the best value on the grid."""
    weights, best, steps = dict(start), score(start), []
    for name in weights:
        results = []
        for value in sorted({*tuning.GRID, weights[name]}):
            trial = {**weights, name: value}
            if sum(trial.values()) > 0:
                results.append((score(trial), value))
        top_score, top_value = min(results, key=lambda r: (-r[0], r[1]))
        if top_score > best and top_value != weights[name]:
            weights[name], best = top_value, top_score
        steps.append({"weight": name, "best_value": weights[name], "ndcg_at_10": round(best, 4)})
    return steps


def main() -> None:
    evaluator = build_evaluator()
    queries = evaluator.queries
    score = lambda g: evaluator.evaluator.score(expand_four(g), queries)  # noqa: E731
    starts = [
        FIRST_GUESS,
        {"title": 0.2, "text_vectors": 0.2, "image": 0.2, "attributes": 0.4},
        dict.fromkeys(GROUPS, 0.25),
        *[tuning.random_start(GROUPS, seed=SEED + k) for k in range(N_RANDOM_STARTS)],
    ]
    runs = []
    for number, start in enumerate(starts, start=1):
        result = tuning.coordinate_ascent(score, start)
        runs.append(
            {
                "start_no": number,
                "start": start,
                "final": result.weights,
                "ndcg_at_10": result.score,
                "rounds": result.rounds,
            }
        )
        print(
            f"start {number}: nDCG@10 {result.score:.4f} after {result.rounds} rounds", flush=True
        )
    best = max(runs, key=lambda r: r["ndcg_at_10"])
    report = {
        "round_one_from_the_first_guess": round_one(score, FIRST_GUESS),
        "starts": runs,
        "best_start_no": best["start_no"],
        "best_weights": best["final"],
        "best_shares": tuning.normalise(best["final"]),
    }
    path = ROOT / "results" / "coordinate_ascent_trace.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
