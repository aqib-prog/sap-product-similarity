"""MLflow logging (skipped when MLflow is not installed)."""

import pytest

pytest.importorskip("mlflow")

from product_similarity.tracking import log_run, mlflow_available  # noqa: E402


def test_mlflow_is_reported_available():
    assert mlflow_available() is True


def test_a_run_records_settings_scores_and_files(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'test.db'}")
    note = tmp_path / "note.txt"
    note.write_text("hello")
    run_id = log_run(
        "unit-test",
        "run-a",
        {"method": "demo", "k": 10},
        {"ndcg_at_10": 0.5},
        [note],
        tags={"limitations": "one rater"},
    )
    run = mlflow.get_run(run_id)
    assert run.data.params == {"method": "demo", "k": "10"}
    assert run.data.metrics["ndcg_at_10"] == 0.5
    assert run.data.tags["limitations"] == "one rater"
    client = mlflow.tracking.MlflowClient()
    assert [a.path for a in client.list_artifacts(run_id)] == ["note.txt"]
