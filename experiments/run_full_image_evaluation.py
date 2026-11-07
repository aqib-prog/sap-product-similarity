"""Score text search and CLIP image search over the whole catalogue against all ratings.

Run from the project folder:  python experiments/run_full_image_evaluation.py
The answer key is every development rating so far (text results, the first image
results, and the new results of CLIP over the whole catalogue), so all methods are
scored against the same key. Every method shows one copy of each item (see
product_similarity/dedupe.py). The rating files are only read, never changed.
Add --no-mlflow to skip MLflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from product_similarity import evaluation as ev
from product_similarity.dedupe import SAME_ITEM, pick_distinct
from product_similarity.judging import comparison_methods
from product_similarity.service import baseline_engine
from product_similarity.tracking import log_run, mlflow_available

ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = ROOT / "evaluation" / "judgments"
RESULTS = ROOT / "results"
RATING_FILES = ["dev_grades.csv", "dev_grades_images.csv", "dev_grades_full_images.csv"]
EXPERIMENT = "dev-ratings-full-images"
DEPTH = 60  # how far down a text ranking we look to find 10 different items

LIMITATIONS = [
    "One person rated the results, so the labels reflect one judgment.",
    "Only 30 development products were used, so small differences cannot be told apart.",
    "The 20 locked final products have not been rated or used.",
]
TAGS = {
    "evaluation_set": "development (30 products)",
    "final_set": "locked, not used",
    "rater": "single person",
    "one_copy_per_item": f"same picture link or picture cosine >= {SAME_ITEM}",
    "limitations": " | ".join(LIMITATIONS),
}


def merge(files: list[str]) -> dict:
    merged: dict[str, dict[str, int]] = {}
    for name in files:
        for query, rated in ev.load_grades(JUDGMENTS / name).items():
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
    grades = merge(RATING_FILES)

    engine = baseline_engine()
    data = np.load(ROOT / "data" / "embeddings" / "all_clip.npz")
    vec_of = {str(u): i for i, u in enumerate(data["urls"])}
    unit = data["vectors"] / np.linalg.norm(data["vectors"], axis=1, keepdims=True)
    # One row per catalogue product. Products without a picture get a zero vector and a
    # link of their own, so they are never treated as copies of anything.
    products = engine.products
    row_of = {p.uniq_id: i for i, p in enumerate(products)}
    matrix = np.zeros((len(products), unit.shape[1]), dtype=unit.dtype)
    links = []
    for i, p in enumerate(products):
        if p.image_url in vec_of:
            matrix[i] = unit[vec_of[p.image_url]]
            links.append(p.image_url)
        else:
            links.append(f"no-picture-{p.uniq_id}")

    def one_copy_each(ids: list[str], query_id: str) -> list[str]:
        rows = [row_of[i] for i in ids]
        kept, _ = pick_distinct(rows, matrix, links, engine.resolve(query_id), ev.K)
        return [products[r].uniq_id for r in kept]

    text = comparison_methods(engine.products, engine.id_map)
    ranked: dict[str, dict[str, list[str]]] = {}
    raw_text: dict[str, dict[str, list[str]]] = {}
    for name, m in text.items():
        raw_text[name] = {no: m.find_similar(pid, ev.K) for no, pid in dev.items()}
        ranked[name] = {
            no: one_copy_each(m.find_similar(pid, DEPTH), pid) for no, pid in dev.items()
        }
    full = json.loads(
        (ROOT / "data/embeddings/dev_full_clip_top10.json").read_text(encoding="utf-8")
    )
    ranked["image_clip_all"] = {no: full[no] for no in dev}
    # CLIP over the first 708 pictures only (the earlier test), as a reference point.
    small = json.loads((ROOT / "data/embeddings/dev_image_top10.json").read_text(encoding="utf-8"))
    ranked["image_clip_708"] = {no: small["clip"][no] for no in dev}

    per_method, summaries = {}, {}
    for name, lists in ranked.items():
        per_method[name] = ev.score_method(lists, grades)
        summaries[name] = ev.summarise(per_method[name])
    raw_ndcg = {n: ev.summarise(ev.score_method(raw_text[n], grades))["ndcg_at_10"] for n in text}
    copies_removed = {
        n: sum(len(set(raw_text[n][no]) - set(ranked[n][no])) for no in dev) for n in text
    }

    RESULTS.mkdir(exist_ok=True)
    board = RESULTS / "dev_full_image_scoreboard.csv"
    columns = ["method", *next(iter(summaries.values())), "ndcg_without_one_copy_rule"]
    with open(board, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for name, s in summaries.items():
            extra = round(raw_ndcg[name], 6) if name in raw_ndcg else ""
            writer.writerow(
                [name, *[round(v, 6) if isinstance(v, float) else v for v in s.values()], extra]
            )

    ndcg = {
        n: {q: s["ndcg"] for q, s in r.items() if s["ndcg"] is not None}
        for n, r in per_method.items()
    }
    names = list(ranked)
    pairs = RESULTS / "dev_full_image_pairwise.csv"
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

    per_query = RESULTS / "dev_full_image_per_query.csv"
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
        "# Development evaluation: text search and CLIP over the whole catalogue",
        "",
        "Five methods scored against the same human ratings of 30 development products "
        "(all ratings so far). Every method shows one copy of each item.",
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
        "- One copy per item: "
        + ", ".join(f"{n} lost {copies_removed[n]} copies" for n in text)
        + " (out of 300 results each). Without the rule the text nDCG@10 would be "
        + ", ".join(f"{n} {raw_ndcg[n]:.3f}" for n in text)
        + ".",
        "- image_clip_708 is the earlier test that searched only 708 pictures; image_clip_all "
        "searched every downloaded picture.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in LIMITATIONS],
        "",
    ]
    report = RESULTS / "dev_full_image_evaluation_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    for name, s in summaries.items():
        print(
            f"{name:16} nDCG@10 {s['ndcg_at_10']:.3f} [{s['ndcg_at_10_low']:.3f}, {s['ndcg_at_10_high']:.3f}] "
            f"P@10 {s['precision_at_10']:.3f} strong {s['strong_precision_at_10']:.2f}"
            + (f" | no-dedupe: {raw_ndcg[name]:.3f}" if name in raw_ndcg else "")
        )
    print("copies removed:", copies_removed)
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
            EXPERIMENT,
            name,
            {
                "method": name,
                "k": ev.K,
                "dev_products": len(dev),
                "ratings_sha256": sha(JUDGMENTS / RATING_FILES[0]),
                "image_ratings_sha256": sha(JUDGMENTS / RATING_FILES[1]),
                "full_image_ratings_sha256": sha(JUDGMENTS / RATING_FILES[2]),
                "git_commit": commit,
                "mlflow": mlflow.__version__,
            },
            metrics,
            [board, pairs, per_query, report],
            tags=TAGS,
        )
    print(f"MLflow: {len(summaries)} runs recorded in experiment '{EXPERIMENT}' (mlflow_images.db)")


if __name__ == "__main__":
    main()
