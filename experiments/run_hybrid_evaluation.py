"""Score the hybrid search next to the single methods against all ratings.

Run from the project folder:  python experiments/run_hybrid_evaluation.py
The answer key is every development rating so far. The hybrid scores are out-of-fold:
each product is scored with weights that were tuned on the other products (5-fold
cross-validation), so they are honest scores on products the weights have not seen.
All methods show one copy of each item. Add --no-mlflow to skip MLflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

from hybrid_setup import RATING_FILES, SEED, build_evaluator, expand_four

from product_similarity import evaluation as ev
from product_similarity import tuning
from product_similarity.dedupe import pick_distinct
from product_similarity.judging import comparison_methods
from product_similarity.tracking import log_run, mlflow_available

ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = ROOT / "evaluation" / "judgments"
RESULTS = ROOT / "results"
EXPERIMENT = "dev-ratings-hybrid"
LABELS = {
    "hybrid_4_weights": "Hybrid, 4 weights",
    "hybrid_10_weights": "Hybrid, 10 weights",
    "image_clip_all": "CLIP (pictures)",
    "text_vectors": "Sentence vectors (titles)",
    "baseline": "Title (TF-IDF) and attributes (my baseline)",
    "title_only": "TF-IDF, title only",
    "attributes_only": "Attributes only",
}
LIMITATIONS = [
    "One person rated the results, so the labels reflect one judgment.",
    "Only 30 development products were used, so small differences cannot be told apart.",
    "The 20 locked final products have not been rated or used.",
]
TAGS = {
    "evaluation_set": "development (30 products)",
    "final_set": "locked, not used",
    "rater": "single person",
    "hybrid_scores": "out-of-fold: 5-fold cross-validation, weights tuned on the other products",
    "limitations": " | ".join(LIMITATIONS),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    setup = build_evaluator()
    evaluator, engine, dev, grades, queries = (
        setup.evaluator,
        setup.engine,
        setup.dev,
        setup.grades,
        setup.queries,
    )
    report = json.loads((RESULTS / "hybrid_tuning.json").read_text(encoding="utf-8"))
    row_of = {str(uid): i for i, uid in enumerate(engine.ids)}

    def one_copy_each(ids: list[str], pid: str) -> list[str]:
        rows = [row_of[i] for i in ids]
        kept, _ = pick_distinct(
            rows, evaluator.picture_matrix, evaluator.picture_links, engine.resolve(pid), ev.K
        )
        return [str(engine.ids[r]) for r in kept]

    ranked: dict[str, dict[str, list[str]]] = {}
    for name, method in comparison_methods(engine.products, engine.id_map).items():
        ranked[name] = {
            no: one_copy_each(method.find_similar(pid, 60), pid) for no, pid in dev.items()
        }
    for name, path in (
        ("image_clip_all", "dev_full_clip_top10.json"),
        ("text_vectors", "dev_text_top10.json"),
    ):
        top = json.loads((ROOT / "data/embeddings" / path).read_text(encoding="utf-8"))
        ranked[name] = {no: top[no] for no in dev}

    per_method = {name: ev.score_method(lists, grades) for name, lists in ranked.items()}

    # Hybrid: score every product with the weights tuned without it (out-of-fold).
    folds = tuning.make_folds(queries, 5, SEED)
    fold_weights = {
        "hybrid_10_weights": [row["weights"] for row in report["cross_validation"]["folds"]],
        "hybrid_4_weights": [
            expand_four(row["weights"]) for row in report["four_weights"]["folds"]
        ],
    }
    for name, weight_sets in fold_weights.items():
        scored: dict[str, dict] = {}
        for test, weights in zip(folds, weight_sets):
            scored.update(evaluator.per_query(weights, test))
        per_method[name] = {no: scored[no] for no in queries}
    final = {
        "hybrid_10_weights": report["tuned_weights"],
        "hybrid_4_weights": expand_four(report["four_weights"]["weights"]),
    }
    in_sample = {
        name: ev.summarise(evaluator.per_query(w, queries))["ndcg_at_10"]
        for name, w in final.items()
    }

    summaries = {name: ev.summarise(per_method[name]) for name in LABELS}
    RESULTS.mkdir(exist_ok=True)
    board = RESULTS / "dev_hybrid_scoreboard.csv"
    columns = ["method", *next(iter(summaries.values())), "ndcg_tuned_on_same_30"]
    with open(board, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for name, s in summaries.items():
            extra = round(in_sample[name], 6) if name in in_sample else ""
            writer.writerow(
                [name, *[round(v, 6) if isinstance(v, float) else v for v in s.values()], extra]
            )

    ndcg = {
        n: {q: s["ndcg"] for q, s in r.items() if s["ndcg"] is not None}
        for n, r in per_method.items()
    }
    names = list(LABELS)
    pairs = RESULTS / "dev_hybrid_pairwise.csv"
    pair_rows = []
    with open(pairs, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["first", "second", "ndcg_difference", "low", "high", "difference_is_real"])
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                mean, low, high = ev.paired_difference(ndcg[a], ndcg[b])
                real = low > 0 or high < 0
                writer.writerow([a, b, round(mean, 4), round(low, 4), round(high, 4), real])
                pair_rows.append((a, b, mean, low, high, real))
    per_query = RESULTS / "dev_hybrid_per_query.csv"
    with open(per_query, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "query_no", "ndcg_at_10", "precision_at_10"])
        for name, rows in per_method.items():
            for q, s in rows.items():
                writer.writerow(
                    [
                        name,
                        q,
                        "" if s["ndcg"] is None else round(s["ndcg"], 4),
                        round(s["precision"], 4),
                    ]
                )

    lines = [
        "# Development evaluation: hybrid search and the single methods",
        "",
        "Seven methods scored against the same human ratings of 30 development products. "
        "The hybrid scores are out-of-fold: each product is scored with weights tuned on the "
        "other products. Every method shows one copy of each item.",
        "",
        "| Method | nDCG@10 | 95% range | Precision@10 |",
        "|---|---|---|---|",
    ]
    for name, s in summaries.items():
        lines.append(
            f"| {LABELS[name]} | {s['ndcg_at_10']:.3f} | {s['ndcg_at_10_low']:.2f} to "
            f"{s['ndcg_at_10_high']:.2f} | {s['precision_at_10']:.2f} |"
        )
    lines += [
        "",
        "## Differences (nDCG@10, same products)",
        "",
        "| Comparison | Difference | 95% range | Verdict |",
        "|---|---|---|---|",
    ]
    for a, b, mean, low, high, real in pair_rows:
        lines.append(
            f"| {LABELS[a]} minus {LABELS[b]} | {mean:+.3f} | {low:+.3f} to {high:+.3f} | "
            f"{'real' if real else 'could be luck'} |"
        )
    tuned = report["four_weights"]["weights"]
    lines += [
        "",
        "## Final weights (tuned on all 30 products)",
        "",
        "| Part | 4-weight mix | 10-weight mix |",
        "|---|---|---|",
    ]
    four = expand_four(tuned)
    for part in report["parts"]:
        lines.append(f"| {part} | {four[part]:.3f} | {report['tuned_weights'][part]:.3f} |")
    lines += ["", "## Limitations", "", *[f"- {item}" for item in LIMITATIONS], ""]
    report_path = RESULTS / "dev_hybrid_evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    for name, s in summaries.items():
        print(
            f"{name:18} nDCG@10 {s['ndcg_at_10']:.3f} [{s['ndcg_at_10_low']:.3f}, {s['ndcg_at_10_high']:.3f}] P@10 {s['precision_at_10']:.3f}"
            + (f" | tuned on same 30: {in_sample[name]:.3f}" if name in in_sample else "")
        )
    for a, b, mean, low, high, real in pair_rows:
        if "hybrid" in a:
            print(f"{a} - {b}: {mean:+.3f} [{low:+.3f}, {high:+.3f}] {'real' if real else 'luck?'}")

    if args.no_mlflow or not mlflow_available():
        return
    import mlflow

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16]  # noqa: E731
    for name, s in summaries.items():
        metrics = {k: v for k, v in s.items() if isinstance(v, (int, float))}
        params = {
            "method": name,
            "k": ev.K,
            "dev_products": len(dev),
            "git_commit": commit,
            "mlflow": mlflow.__version__,
            "ratings_sha256": "|".join(
                sha(JUDGMENTS / f)[:8] for f in RATING_FILES if (JUDGMENTS / f).exists()
            ),
        }
        if name in final:
            params.update({f"weight_{k}": round(v, 4) for k, v in final[name].items()})
        log_run(
            EXPERIMENT, name, params, metrics, [board, pairs, per_query, report_path], tags=TAGS
        )
    print(f"MLflow: {len(summaries)} runs recorded in experiment '{EXPERIMENT}' (mlflow_images.db)")


if __name__ == "__main__":
    main()
