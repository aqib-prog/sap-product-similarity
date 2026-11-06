"""Tune the hybrid weights on the 30 development products and check them.

Run from the project folder:  python experiments/tune_hybrid_weights.py
Parts: TF-IDF title, sentence-vector title, CLIP picture, category, brand, audience,
price, colour, weight and rating. The weights are chosen by coordinate ascent on
nDCG@10 against all development ratings so far, started from several points. A 5-fold
cross-validation checks that they also work on products they were not tuned on, and
removing each part in turn shows how much each one matters. The 20 locked products are
never used or returned.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from hybrid_setup import SEED, build_evaluator, expand_four

from product_similarity import evaluation as ev
from product_similarity import tuning
from product_similarity.hybrid_eval import PARTS
from product_similarity.scoring import DEFAULT_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EMBEDDINGS = ROOT / "data" / "embeddings"
N_RANDOM_STARTS = 2


def main() -> None:
    setup = build_evaluator()
    evaluator, floors, queries = setup.evaluator, setup.floors, setup.queries
    print("embedding floors (similarity of random pairs):", floors, flush=True)

    score_all = lambda w: evaluator.score(w, queries)  # noqa: E731
    score_on = lambda w, subset: evaluator.score(w, subset)  # noqa: E731

    default = {part: DEFAULT_WEIGHTS[part] for part in PARTS}
    t0 = time.monotonic()
    base_score = score_all(default)
    print(
        f"untuned weights: nDCG@10 {base_score:.3f} (one score takes "
        f"{time.monotonic() - t0:.2f} s)",
        flush=True,
    )

    starts = [
        default,
        {**default, "text_vectors": 0.2, "image": 0.2},
        dict.fromkeys(PARTS, 0.1),
        *[tuning.random_start(PARTS, seed=SEED + k) for k in range(N_RANDOM_STARTS)],
    ]

    print("tuning on all 30 products ...", flush=True)
    t0 = time.monotonic()
    final = tuning.tune(score_all, starts)
    final_weights = tuning.normalise(final.weights)
    print(
        f"tuned: nDCG@10 {final.score:.3f} after {final.evaluations} tries, "
        f"{time.monotonic() - t0:.0f} s",
        flush=True,
    )
    print({k: round(v, 3) for k, v in final_weights.items()}, flush=True)

    print("5-fold cross-validation ...", flush=True)
    folds = tuning.cross_validate(score_on, queries, starts, reference=default, seed=SEED)
    for row in folds:
        print(
            f"fold {row['fold']}: tuned on {row['train_products']} -> train {row['train_score']:.3f}, "
            f"test {row['test_score']:.3f}, untuned test {row['reference_test_score']:.3f}",
            flush=True,
        )

    # A simpler mix with only 4 free weights: the attributes keep their untuned ratios.
    expand = expand_four

    group_names = ("title", "text_vectors", "image", "attributes")
    group_reference = {"title": 0.35, "text_vectors": 0.0, "image": 0.0, "attributes": 0.65}
    group_starts = [
        group_reference,
        {"title": 0.2, "text_vectors": 0.2, "image": 0.2, "attributes": 0.4},
        dict.fromkeys(group_names, 0.25),
        *[tuning.random_start(group_names, seed=SEED + k) for k in range(N_RANDOM_STARTS)],
    ]
    group_score = lambda g: evaluator.score(expand(g), queries)  # noqa: E731
    group_score_on = lambda g, subset: evaluator.score(expand(g), subset)  # noqa: E731
    print("simpler mix with 4 weights ...", flush=True)
    group_final = tuning.tune(group_score, group_starts)
    group_cv = tuning.cross_validate(
        group_score_on, queries, group_starts, reference=group_reference, seed=SEED
    )
    group_mean = lambda key: float(np.mean([r[key] for r in group_cv]))  # noqa: E731
    print(
        f"4 weights: tuned on all {group_final.score:.3f}; cross-validation train "
        f"{group_mean('train_score'):.3f}, test {group_mean('test_score'):.3f}, "
        f"untuned test {group_mean('reference_test_score'):.3f}",
        flush=True,
    )
    print({k: round(v, 3) for k, v in tuning.normalise(group_final.weights).items()}, flush=True)

    parts_matter = tuning.importance(score_all, final_weights)
    per_query = evaluator.per_query(final_weights, queries)
    summary = ev.summarise(per_query)
    top10 = {no: evaluator.top_ids(final_weights, no) for no in queries}
    mean = lambda key: float(np.mean([r[key] for r in folds]))  # noqa: E731
    report = {
        "parts": list(PARTS),
        "embedding_floors": floors,
        "untuned_weights": default,
        "untuned_ndcg_at_10": base_score,
        "tuned_weights": final_weights,
        "tuned_ndcg_at_10": final.score,
        "tuning_tries": final.evaluations,
        "tuned_precision_at_10": summary["precision_at_10"],
        "unrated_in_top_10": summary["unrated_in_top_10"],
        "cross_validation": {
            "folds": folds,
            "mean_train": mean("train_score"),
            "mean_test": mean("test_score"),
            "mean_untuned_test": mean("reference_test_score"),
        },
        "importance": parts_matter,
        "four_weights": {
            "weights": tuning.normalise(group_final.weights),
            "ndcg_at_10": group_final.score,
            "mean_train": group_mean("train_score"),
            "mean_test": group_mean("test_score"),
            "mean_untuned_test": group_mean("reference_test_score"),
            "folds": group_cv,
        },
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "hybrid_tuning.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (EMBEDDINGS / "dev_hybrid_top10.json").write_text(json.dumps(top10), encoding="utf-8")
    cv = report["cross_validation"]
    print(
        f"cross-validation: train {cv['mean_train']:.3f}, test {cv['mean_test']:.3f}, "
        f"untuned test {cv['mean_untuned_test']:.3f}",
        flush=True,
    )
    for row in parts_matter:
        print(f"  without {row['part']:13} drop {row['drop']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
