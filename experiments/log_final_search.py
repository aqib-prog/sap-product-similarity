"""Record the exact and fast modes of the final search in MLflow, one run for each.

Run from the project folder:  python experiments/log_final_search.py
Reads results/fast_vs_exact_check.json. Runs tagged `headline = yes` are the exact mode and
the fast mode with 100 nearest products from each index; they are the ones shown side by side.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from product_similarity.tracking import log_run, mlflow_available

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPERIMENT = "final-search"
RULE = "30 development products, 5 runs each, fastest kept, median; one thread"


def main() -> None:
    if not mlflow_available():
        raise SystemExit("MLflow is not installed")
    import faiss
    import mlflow

    check = json.loads((RESULTS / "fast_vs_exact_check.json").read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    shared = {
        "git_commit": commit,
        "mlflow": mlflow.__version__,
        "faiss": faiss.__version__,
        "k": 10,
        "threads": 1,
        "products": check["products"],
    }
    tags = {
        "rule": RULE,
        "final_set": "locked, not used",
        "reference": "exact mode of the real search",
        "truth": "my ratings",
    }
    files = [RESULTS / "fast_vs_exact_check.json"]
    block = check["comparison"]
    exact = block["exact"]
    log_run(
        EXPERIMENT,
        "exact",
        {**shared, "mode": "exact", "can_be_results": block["products_that_can_be_results"]},
        {
            "ndcg_at_10": exact["ndcg_at_10"],
            "precision_at_10": exact["precision_at_10"],
            "recall_at_10_vs_exact": 1.0,
            "lost_of_300": 0,
            "ms_per_query": exact["ms_per_query"],
            "memory_mb": exact["memory_mb"],
            "build_seconds": 0.0,
        },
        files,
        tags={**tags, "headline": "yes"},
    )
    count = 1
    for row in block["by_candidates"]:
        log_run(
            EXPERIMENT,
            f"fast_{row['candidates']}_per_index",
            {
                **shared,
                "mode": "fast",
                "nearest_from_each_index": row["candidates"],
                "can_be_results": block["products_that_can_be_results"],
            },
            {
                "ndcg_at_10": row["ndcg_at_10"],
                "precision_at_10": row["precision_at_10"],
                "recall_at_10_vs_exact": row["recall_at_10_vs_exact"],
                "lost_of_300": row["lost_from_final_top10_total"],
                "ms_per_query": row["ms_per_query"],
                "memory_mb": block["fast"]["memory_mb"],
                "build_seconds": block["fast"]["build_seconds"],
                "unrated_in_top_10": row["unrated_in_top_10"],
            },
            files,
            tags={**tags, "headline": "yes" if row["candidates"] == 100 else "no"},
        )
        count += 1
    print(f"MLflow: {count} runs recorded in experiment '{EXPERIMENT}' (mlflow_images.db)")


if __name__ == "__main__":
    main()
