"""Record experiments in MLflow (optional; the web service never imports this)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def log_run(
    experiment: str,
    run_name: str,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    files: list[Path] | None = None,
    tags: Mapping[str, str] | None = None,
) -> str:
    """Save one run (settings, scores and files) and return its run ID."""
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        for path in files or []:
            mlflow.log_artifact(str(path))
        return run.info.run_id
