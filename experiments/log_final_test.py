"""Record the final test (20 locked products) in MLflow, one run for each mode.

Run from the project folder:  python experiments/log_final_test.py
Reads results/final_test_scores.json. The two runs are shown side by side.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from product_similarity.tracking import log_run, mlflow_available

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "final-test"


def main() -> None:
    if not mlflow_available():
        raise SystemExit("MLflow is not installed")
    import faiss
    import mlflow

    data = json.loads((ROOT / "results" / "final_test_scores.json").read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    shared = {
        "git_commit": commit,
        "mlflow": mlflow.__version__,
        "faiss": faiss.__version__,
        "products": data["products"],
        "threads": 1,
    }
    tags = {
        "final_set": "the 20 locked products, rated once, used once",
        "truth": "my ratings",
        "rule": "20 products, 5 runs each, fastest kept, median; one thread",
    }
    for name, m in data["modes"].items():
        log_run(
            EXPERIMENT,
            name,
            {**shared, "mode": name, "nearest_from_each_index": 100 if name == "fast" else ""},
            {
                "ndcg_at_10": m["ndcg_at_10"],
                "precision_at_10": m["precision_at_10"],
                "ms_per_search": m["ms_per_search"],
                "results_rated_2": m["rated_2"],
                "results_rated_1": m["rated_1"],
                "results_rated_0": m["rated_0"],
            },
            [ROOT / "results" / "final_test_scores.json"],
            tags=tags,
        )
    print("MLflow: 2 runs recorded in experiment 'final-test' (mlflow_images.db)")


if __name__ == "__main__":
    main()
