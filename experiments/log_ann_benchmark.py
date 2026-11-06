"""Record the fast-search benchmark in MLflow, one run for each index setting.

Run from the project folder:  python experiments/log_ann_benchmark.py
Reads results/ann_exact_baseline.json and results/ann_benchmark.json. Runs tagged
`headline = yes` are the exact baseline and, for each method, its fastest setting that
still finds at least 95% of the exact top 10; they are the ones shown side by side.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from product_similarity.tracking import log_run, mlflow_available

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPERIMENT = "ann-benchmark"
SETTINGS = (
    "M",
    "efConstruction",
    "efSearch",
    "nlist",
    "nprobe",
    "n_trees",
    "search_k",
    "dimensions",
)
RULE = "1,000 random products (seed 2020, no locked product), 5 runs each, fastest kept, median; one thread"


def headline(rows: list[dict]) -> set[int]:
    chosen: set[int] = set()
    for method in {r["index"] for r in rows}:
        group = [i for i, r in enumerate(rows) if r["index"] == method]
        good = [i for i in group if rows[i]["recall_at_10"] >= 0.95]
        pool = good or group
        chosen.add(
            min(pool, key=lambda i: rows[i]["ms_per_query"])
            if good
            else max(pool, key=lambda i: rows[i]["recall_at_10"])
        )
    return chosen


def main() -> None:
    if not mlflow_available():
        raise SystemExit("MLflow is not installed")
    import faiss
    import mlflow

    base = json.loads((RESULTS / "ann_exact_baseline.json").read_text(encoding="utf-8"))
    bench = json.loads((RESULTS / "ann_benchmark.json").read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    shared = {
        "git_commit": commit,
        "mlflow": mlflow.__version__,
        "faiss": faiss.__version__,
        "k": 10,
        "threads": 1,
    }
    tags = {
        "rule": RULE,
        "final_set": "locked, not used",
        "reference": "exact top 10 by cosine similarity",
    }
    count = 0
    for space, info in bench["spaces"].items():
        timing = base["spaces"][space]["queries"]["random1000"]
        log_run(
            EXPERIMENT,
            f"{space}_numpy_exact",
            {
                **shared,
                "space": space,
                "index": "numpy_exact",
                "products": info["products"],
                "dimensions": info["dimensions"],
            },
            {
                "ms_per_query": timing["one_query_median_ms"],
                "recall_at_10": 1.0,
                "index_mb": base["spaces"][space]["memory_mb"],
                "build_seconds": 0.0,
            },
            [RESULTS / "ann_exact_baseline.json"],
            tags={**tags, "headline": "yes"},
        )
        count += 1
        rows = info["indexes"]
        top = headline(rows)
        for i, row in enumerate(rows):
            label = "_".join([row["index"], *[f"{k}{row[k]}" for k in SETTINGS if k in row]])
            params = {
                **shared,
                "space": space,
                "products": info["products"],
                "dimensions": info["dimensions"],
                **{k: v for k, v in row.items() if k in SETTINGS or k == "index"},
            }
            metrics = {
                k: row[k] for k in ("ms_per_query", "recall_at_10", "index_mb", "build_seconds")
            }
            log_run(
                EXPERIMENT,
                f"{space}_{label}",
                params,
                metrics,
                [RESULTS / "ann_benchmark.json"],
                tags={**tags, "headline": "yes" if i in top else "no"},
            )
            count += 1
    print(f"MLflow: {count} runs recorded in experiment '{EXPERIMENT}' (mlflow_images.db)")

    # The fast index used inside the hybrid (see ann_hybrid_check.py).
    check = json.loads((RESULTS / "ann_hybrid_check.json").read_text(encoding="utf-8"))
    hybrid_tags = {
        "final_set": "locked, not used",
        "reference": "normal hybrid over all products",
        "queries": "30 development products",
    }
    log_run(
        "ann-hybrid-check",
        "hybrid_exact",
        {**shared, "pipeline": "score all products with the 4-weight hybrid"},
        {
            "ndcg_at_10": check["exact_hybrid_ndcg_at_10"],
            "overlap_at_10": 1.0,
            "ms_per_query": check["exact_hybrid_ms_per_query"],
        },
        [RESULTS / "ann_hybrid_check.json"],
        tags=hybrid_tags,
    )
    for row in check["results"]:
        if row["candidates"] != 200:
            continue
        log_run(
            "ann-hybrid-check",
            f"hybrid_{row['candidates_from']}_{row['candidates']}",
            {
                **shared,
                "pipeline": f"{row['candidates_from']} picks {row['candidates']} candidates, then the 4-weight hybrid picks 10",
                "candidates": row["candidates"],
            },
            {
                "ndcg_at_10": row["ndcg_at_10"],
                "overlap_at_10": row["overlap_with_exact_hybrid_at_10"],
                "ms_per_query": row["candidate_search_ms"],
            },
            [RESULTS / "ann_hybrid_check.json"],
            tags=hybrid_tags,
        )
    print("MLflow: 3 runs recorded in experiment 'ann-hybrid-check'")


if __name__ == "__main__":
    main()
