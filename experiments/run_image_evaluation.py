"""Score text and image similarity together against all development ratings.

Run from the project folder:  python experiments/run_image_evaluation.py
The answer key is the first ratings plus the ratings of the new image results,
so every method is scored against the same key. The first ratings file is only
read, never changed. Add --no-mlflow to skip MLflow.
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
JUDGMENTS = ROOT / "evaluation" / "judgments"
RESULTS = ROOT / "results"
IMAGE_METHODS = {"image_resnet": "resnet50", "image_clip": "clip"}

LIMITATIONS = [
    "One person rated the results, so the labels reflect one judgment.",
    "Only 30 development products were used, so small differences cannot be told apart.",
    "Image search only looked at 708 downloaded pictures; text search looked at all 30,000 products.",
    "The first ratings were collected for results picked by the text methods, so they may favour text.",
    "The 20 locked final products have not been rated or used.",
]
TAGS = {
    "evaluation_set": "development (30 products)",
    "final_set": "locked, not used",
    "rater": "single person",
    "limitations": " | ".join(LIMITATIONS),
}


def merge(first: dict, second: dict) -> dict:
    merged = {q: dict(v) for q, v in first.items()}
    for query, rated in second.items():
        for candidate, grade in rated.items():
            assert candidate not in merged.setdefault(query, {}), "a result was rated twice"
            merged[query][candidate] = grade
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    queries = json.loads((ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    dev = {q["query_no"]: q["id"] for q in queries if q["split"] == "dev"}
    first = ev.load_grades(JUDGMENTS / "dev_grades.csv")
    second_file = JUDGMENTS / "dev_grades_images.csv"
    second = ev.load_grades(second_file)
    grades = merge(first, second)

    # One result was left blank. Score it both ways to show how much it could matter.
    with open(JUDGMENTS / "dev_sheet_images.csv", newline="", encoding="utf-8-sig") as handle:
        sheet = [(r["query_no"], r["candidate_id"]) for r in csv.DictReader(handle)]
    blank = [(q, c) for q, c in sheet if c not in second.get(q, {})]
    as_zero = {q: dict(v) for q, v in grades.items()}
    as_two = {q: dict(v) for q, v in grades.items()}
    for q, c in blank:
        as_zero[q][c], as_two[q][c] = 0, 2

    engine = baseline_engine()
    text = comparison_methods(engine.products, engine.id_map)
    top10 = json.loads((ROOT / "data/embeddings/dev_image_top10.json").read_text(encoding="utf-8"))
    ranked: dict[str, dict[str, list[str]]] = {
        name: {no: m.find_similar(pid, ev.K) for no, pid in dev.items()} for name, m in text.items()
    }
    for name, key in IMAGE_METHODS.items():
        ranked[name] = {no: top10[key][no] for no in dev}

    per_method, summaries = {}, {}
    for name, lists in ranked.items():
        per_method[name] = ev.score_method(lists, as_zero)
        summaries[name] = ev.summarise(per_method[name])
        summaries[name]["ndcg_if_blank_is_2"] = ev.summarise(ev.score_method(lists, as_two))[
            "ndcg_at_10"
        ]
        summaries[name]["unrated_in_top_10"] = sum(
            1 for no, lst in lists.items() for c in lst[: ev.K] if c not in grades.get(no, {})
        )
    first_only = {n: ev.summarise(ev.score_method(ranked[n], first))["ndcg_at_10"] for n in text}

    RESULTS.mkdir(exist_ok=True)
    board = RESULTS / "dev_image_scoreboard.csv"
    columns = ["method", *next(iter(summaries.values())), "ndcg_first_ratings_only"]
    with open(board, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for name, s in summaries.items():
            extra = round(first_only[name], 4) if name in first_only else ""
            writer.writerow(
                [name, *[round(v, 4) if isinstance(v, float) else v for v in s.values()], extra]
            )

    ndcg = {
        n: {q: s["ndcg"] for q, s in r.items() if s["ndcg"] is not None}
        for n, r in per_method.items()
    }
    names = list(ranked)
    pairs = RESULTS / "dev_image_pairwise.csv"
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

    per_query = RESULTS / "dev_image_per_query.csv"
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
        "# Development evaluation: text and image similarity",
        "",
        "Five methods scored against the same human ratings of 30 development products "
        "(first ratings plus the ratings of the new image results).",
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
        "## Differences (nDCG@10, same products)",
        "",
        "| Comparison | Difference | 95% range | Verdict |",
        "|---|---|---|---|",
    ]
    for a, b, mean, low, high, real in pair_rows:
        lines.append(
            f"| {a} minus {b} | {mean:+.3f} | {low:+.3f} to {high:+.3f} | {'real' if real else 'could be luck'} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        f"- {len(blank)} new result was left unrated and is scored as 0 here. Scored as 2 instead, "
        "the image scores change by at most "
        f"{max(abs(summaries[n]['ndcg_if_blank_is_2'] - summaries[n]['ndcg_at_10']) for n in IMAGE_METHODS):.3f}.",
        "- Text scores are lower than in the first report because the answer key now also contains the new ratings.",
        "- Results in the top 10 that nobody rated (scored as 0): "
        + ", ".join(f"{n} {summaries[n]['unrated_in_top_10']}" for n in summaries)
        + ".",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in LIMITATIONS],
        "",
    ]
    report = RESULTS / "dev_image_evaluation_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    for name, s in summaries.items():
        print(
            f"{name:16} nDCG@10 {s['ndcg_at_10']:.3f} [{s['ndcg_at_10_low']:.3f}, {s['ndcg_at_10_high']:.3f}] "
            f"P@10 {s['precision_at_10']:.3f} strong {s['strong_precision_at_10']:.2f} "
            f"unrated {s['unrated_in_top_10']} | blank=2: {s['ndcg_if_blank_is_2']:.3f}"
            + (f" | first ratings only: {first_only[name]:.3f}" if name in first_only else "")
        )
    for a, b, mean, low, high, real in pair_rows:
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
        log_run(
            "dev-ratings-with-images",
            name,
            {
                "method": name,
                "k": ev.K,
                "dev_products": len(dev),
                "ratings_sha256": sha(JUDGMENTS / "dev_grades.csv"),
                "image_ratings_sha256": sha(second_file),
                "git_commit": commit,
                "mlflow": mlflow.__version__,
            },
            metrics,
            [board, pairs, per_query, report],
            tags=TAGS,
        )
    print("MLflow: 5 runs recorded in experiment 'dev-ratings-with-images' (mlflow_images.db)")


if __name__ == "__main__":
    main()
