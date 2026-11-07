"""Score the current methods against the development ratings and record the run.

Run from the project folder:  python experiments/run_evaluation.py
Add --no-mlflow to skip MLflow. Results are written to results/ and kept in git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

from product_similarity import evaluation as ev
from product_similarity.judging import comparison_methods
from product_similarity.service import baseline_engine
from product_similarity.tracking import log_run, mlflow_available

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

LIMITATIONS = [
    "One person rated the results, so the labels reflect one judgment.",
    "Only 30 development products were rated, so small differences cannot be told apart.",
    "The 20 locked final products have not been rated or used.",
]
NOTES = [
    "Each method's top 10 is scored against the ratings (nDCG@10 and precision@10, with 95% ranges from resampling the products).",
]
TAGS = {
    "evaluation_set": "development (30 products)",
    "final_set": "locked, not used",
    "rater": "single person",
    "limitations": " | ".join(LIMITATIONS),
}


def write_report(summaries: dict, pairs: list, path: Path) -> None:
    """A readable summary of the results and their limits, kept in git."""
    lines = [
        "# Development evaluation: first scoreboard",
        "",
        "Three methods scored against human ratings of 30 development products (nDCG@10 and precision@10, with 95% ranges).",
        "",
        "| Method | nDCG@10 | 95% range | Precision@10 | Strong matches in top 10 |",
        "|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | {s['ndcg_at_10']:.3f} | {s['ndcg_at_10_low']:.2f} to {s['ndcg_at_10_high']:.2f} "
            f"| {s['precision_at_10']:.2f} | {s['strong_precision_at_10']:.0%} |"
        )
    lines += [
        "",
        "## Differences between methods (nDCG@10, same products)",
        "",
        "| Comparison | Difference | 95% range | Verdict |",
        "|---|---|---|---|",
    ]
    for a, b, mean, low, high in pairs:
        verdict = "real" if low > 0 or high < 0 else "could be luck"
        lines.append(f"| {a} minus {b} | {mean:+.3f} | {low:+.3f} to {high:+.3f} | {verdict} |")
    lines += ["", "## How the scores are computed", ""] + [f"- {item}" for item in NOTES]
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in LIMITATIONS] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    engine = baseline_engine()
    methods = comparison_methods(engine.products, engine.id_map)
    queries = json.loads((ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    dev = {q["query_no"]: q["id"] for q in queries if q["split"] == "dev"}
    grades_file = ROOT / "evaluation/judgments/dev_grades.csv"
    grades = ev.load_grades(grades_file)

    per_method, summaries = {}, {}
    for name, method in methods.items():
        ranked = {no: method.find_similar(pid, ev.K) for no, pid in dev.items()}
        per_method[name] = ev.score_method(ranked, grades)
        summaries[name] = ev.summarise(per_method[name])

    RESULTS.mkdir(exist_ok=True)
    board = RESULTS / "dev_scoreboard.csv"
    with open(board, "w", newline="", encoding="utf-8") as handle:
        columns = ["method", *next(iter(summaries.values()))]
        writer = csv.writer(handle)
        writer.writerow(columns)
        for name, summary in summaries.items():
            writer.writerow(
                [name, *[round(v, 4) if isinstance(v, float) else v for v in summary.values()]]
            )

    ndcg = {
        n: {q: s["ndcg"] for q, s in r.items() if s["ndcg"] is not None}
        for n, r in per_method.items()
    }
    pairs = RESULTS / "dev_pairwise.csv"
    names = list(methods)
    pair_rows = []
    with open(pairs, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["first", "second", "ndcg_difference", "low", "high", "difference_is_real"])
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                mean, low, high = ev.paired_difference(ndcg[a], ndcg[b])
                writer.writerow(
                    [a, b, round(mean, 4), round(low, 4), round(high, 4), low > 0 or high < 0]
                )
                pair_rows.append((a, b, mean, low, high))

    per_query = RESULTS / "dev_per_query.csv"
    with open(per_query, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "query_no", "ndcg_at_10", "precision_at_10", "unrated"])
        for name, rows in per_method.items():
            for query, s in rows.items():
                nd = "" if s["ndcg"] is None else round(s["ndcg"], 4)
                writer.writerow([name, query, nd, round(s["precision"], 4), s["unrated"]])

    report = RESULTS / "dev_evaluation_report.md"
    write_report(summaries, pair_rows, report)

    for name, summary in summaries.items():
        print(
            f"{name:16} nDCG@10 {summary['ndcg_at_10']:.3f} "
            f"[{summary['ndcg_at_10_low']:.3f}, {summary['ndcg_at_10_high']:.3f}]  "
            f"P@10 {summary['precision_at_10']:.3f}"
        )

    if args.no_mlflow or not mlflow_available():
        print("MLflow skipped (use the 'experiments' extra to enable it).")
        return
    import mlflow

    for name, summary in summaries.items():
        weights = methods[name].weights
        log_run(
            "dev-ratings",
            name,
            {
                "method": name,
                "weights": json.dumps({k: v for k, v in weights.items() if v}),
                "k": ev.K,
                "dev_products": len(dev),
                "ratings_sha256": sha256(grades_file)[:16],
                "archive_sha256": sha256(ROOT / "data/archive.zip")[:16],
                "git_commit": git_commit(),
                "mlflow": mlflow.__version__,
            },
            summary,
            [board, pairs, per_query, report],
            tags=TAGS,
        )
    print("MLflow: 3 runs recorded in experiment 'dev-ratings' (mlflow.db)")


if __name__ == "__main__":
    main()
