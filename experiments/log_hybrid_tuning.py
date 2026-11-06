"""Record the hybrid weight tuning in MLflow.

Run from the project folder:  python experiments/log_hybrid_tuning.py [report.json]
Three runs: the untuned weights, the tuned 10-weight mix and the tuned 4-weight mix. For each,
`ndcg_at_10` is the score on all 30 development products and `ndcg_at_10_unseen` is the
5-fold cross-validation score on products the weights were not tuned on.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from product_similarity.tracking import log_run, mlflow_available

ROOT = Path(__file__).resolve().parents[1]
# The first-round report by default; pass another path to record a later round.
REPORT = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "hybrid_tuning_first_round.json"
)
EXPERIMENT = "dev-hybrid-tuning"
TAGS = {
    "evaluation_set": "development (30 products)",
    "final_set": "locked, not used",
    "rater": "single person",
    "method": "coordinate ascent on nDCG@10, 5-fold cross-validation",
    "limitations": "Only 30 products. Results of the tuned mixes that nobody rated count as 0.",
}


def main() -> None:
    if not mlflow_available():
        raise SystemExit("MLflow is not installed")
    import mlflow

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    cv, four = report["cross_validation"], report["four_weights"]
    shared = {
        "folds": 5,
        "dev_products": 30,
        "git_commit": commit,
        "mlflow": mlflow.__version__,
        "report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest()[:16],
    }
    runs = [
        (
            "untuned_baseline",
            report["untuned_weights"],
            report["untuned_ndcg_at_10"],
            cv["mean_untuned_test"],
        ),
        ("tuned_10_weights", report["tuned_weights"], report["tuned_ndcg_at_10"], cv["mean_test"]),
        ("tuned_4_weights", four["weights"], four["ndcg_at_10"], four["mean_test"]),
    ]
    for name, weights, on_all, unseen in runs:
        params = {**shared, **{f"weight_{k}": round(v, 4) for k, v in weights.items()}}
        log_run(
            EXPERIMENT,
            name,
            params,
            {"ndcg_at_10": on_all, "ndcg_at_10_unseen": unseen},
            [REPORT],
            tags=TAGS,
        )
    print(f"MLflow: {len(runs)} runs recorded in experiment '{EXPERIMENT}' (mlflow_images.db)")


if __name__ == "__main__":
    main()
