"""Record the cache measurements in MLflow, one run for each.

Run from the project folder:  python experiments/log_cache.py
Reads results/cache_benchmark.json: a request that is not in the cache (miss), a request that
is (hit), and the traffic simulation for each cache size. All runs are shown side by side.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from product_similarity.tracking import log_run, mlflow_available

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPERIMENT = "caching"


def main() -> None:
    if not mlflow_available():
        raise SystemExit("MLflow is not installed")
    import mlflow

    data = json.loads((RESULTS / "cache_benchmark.json").read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    shared = {
        "git_commit": commit,
        "mlflow": mlflow.__version__,
        "search": data["search"],
        "num_similar": data["num_similar"],
        "cache": "in-process, least recently used",
    }
    tags = {"rule": data["timing_rule"], "final_set": "locked, not used"}
    files = [RESULTS / "cache_benchmark.json"]
    pair = data["miss_against_hit"]
    log_run(
        EXPERIMENT,
        "request_miss",
        {**shared, "kind": "one request, not in the cache", "cache_size": 1024},
        {"ms_per_request": pair["miss_median_ms"], "hit_rate": 0.0},
        files,
        tags=tags,
    )
    log_run(
        EXPERIMENT,
        "request_hit",
        {**shared, "kind": "one request, in the cache", "cache_size": 1024},
        {"ms_per_request": pair["hit_median_ms"], "hit_rate": 1.0},
        files,
        tags=tags,
    )
    for row in data["traffic_simulation"]["results"]:
        log_run(
            EXPERIMENT,
            f"traffic_cache_{row['cache_size']}",
            {
                **shared,
                "kind": "2,000 requests, a few products asked often",
                "cache_size": row["cache_size"],
            },
            {"ms_per_request": row["average_ms_per_request"], "hit_rate": row["hit_rate"]},
            files,
            tags=tags,
        )
    print("MLflow: 6 runs recorded in experiment 'caching' (mlflow_images.db)")


if __name__ == "__main__":
    main()
