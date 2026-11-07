"""Record the PCA experiment in MLflow, one run for each setting.

Run from the project folder:  python experiments/log_ann_pca.py
Reads results/ann_exact_baseline.json, results/ann_benchmark.json and results/ann_pca.json.
Runs tagged `headline = yes` are the two methods without PCA (exact search and HNSW) and
HNSW on the PCA vectors with 256, 128 and 64 numbers; they are the ones shown side by side.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from product_similarity.tracking import log_run, mlflow_available

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPERIMENT = "ann-pca"
SETTINGS = ("M", "efConstruction", "efSearch", "nlist", "nprobe", "n_trees", "search_k")
RULE = "1,000 random products (seed 2020, no locked product), 5 runs each, fastest kept, median; one thread"


def best_hnsw(rows: list[dict]) -> dict:
    """The fastest HNSW setting within half a point of the best recall."""
    hnsw = [r for r in rows if r["index"] == "faiss_hnsw"]
    top = max(r["recall_at_10"] for r in hnsw)
    return min((r for r in hnsw if r["recall_at_10"] >= top - 0.005), key=lambda r: r["ms_per_query"])


def main() -> None:
    if not mlflow_available():
        raise SystemExit("MLflow is not installed")
    import faiss
    import mlflow

    base = json.loads((RESULTS / "ann_exact_baseline.json").read_text(encoding="utf-8"))
    bench = json.loads((RESULTS / "ann_benchmark.json").read_text(encoding="utf-8"))
    pca = json.loads((RESULTS / "ann_pca.json").read_text(encoding="utf-8"))
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
        "reference": "exact top 10 on the full vectors",
    }
    files = [RESULTS / "ann_pca.json"]
    count = 0
    for space, info in pca["spaces"].items():
        full = bench["spaces"][space]
        timing = base["spaces"][space]["queries"]["random1000"]
        hnsw_full = next(
            r
            for r in full["indexes"]
            if r["index"] == "faiss_hnsw" and r["M"] == 16 and r["efSearch"] == 16
        )
        log_run(
            EXPERIMENT,
            f"{space}_full_numpy_exact",
            {**shared, "space": space, "index": "numpy_exact", "dimensions": info["dimensions"], "pca": "no"},
            {
                "ms_per_query": timing["one_query_median_ms"],
                "recall_at_10": 1.0,
                "index_mb": base["spaces"][space]["memory_mb"],
                "build_seconds": 0.0,
            },
            files,
            tags={**tags, "headline": "yes"},
        )
        log_run(
            EXPERIMENT,
            f"{space}_full_hnsw",
            {
                **shared,
                "space": space,
                "index": "faiss_hnsw",
                "dimensions": info["dimensions"],
                "pca": "no",
                "M": 16,
                "efSearch": 16,
            },
            {
                k: hnsw_full[k]
                for k in ("ms_per_query", "recall_at_10", "index_mb", "build_seconds")
            },
            files,
            tags={**tags, "headline": "yes"},
        )
        count += 2
        for dims, block in info["by_dimensions"].items():
            chosen = best_hnsw(block["indexes"])
            for row in block["indexes"]:
                label = "_".join([row["index"], *[f"{k}{row[k]}" for k in SETTINGS if k in row]])
                params = {
                    **shared,
                    "space": space,
                    "pca": "yes",
                    "dimensions": dims,
                    **{k: v for k, v in row.items() if k in SETTINGS or k == "index"},
                }
                metrics = {
                    k: row[k] for k in ("ms_per_query", "recall_at_10", "index_mb", "build_seconds")
                }
                metrics["projection_ms"] = block["projection_ms"]
                log_run(
                    EXPERIMENT,
                    f"{space}_pca{dims}_{label}",
                    params,
                    metrics,
                    files,
                    tags={**tags, "headline": "yes" if row is chosen else "no"},
                )
                count += 1
    print(f"MLflow: {count} runs recorded in experiment '{EXPERIMENT}' (mlflow_images.db)")


if __name__ == "__main__":
    main()
