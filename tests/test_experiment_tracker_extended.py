"""
Testes estendidos para ExperimentTracker (MLflow).
"""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import src.mlflow_tracking.experiment_tracker as tracker_module


def _build_tracker(monkeypatch):
    """Cria tracker com MLflow e client totalmente mockados."""
    experiment = SimpleNamespace(experiment_id="exp-1")
    run = SimpleNamespace(info=SimpleNamespace(run_id="run-1", artifact_uri="artifacts://run-1"))

    mlflow_mock = Mock()
    mlflow_mock.set_tracking_uri = Mock()
    mlflow_mock.get_experiment_by_name = Mock(return_value=experiment)
    mlflow_mock.set_experiment = Mock()
    mlflow_mock.start_run = Mock(return_value=run)
    mlflow_mock.end_run = Mock()
    mlflow_mock.log_params = Mock()
    mlflow_mock.log_metrics = Mock()
    mlflow_mock.log_metric = Mock()
    mlflow_mock.log_artifact = Mock()
    mlflow_mock.log_artifacts = Mock()
    mlflow_mock.log_dict = Mock()
    mlflow_mock.log_figure = Mock()
    mlflow_mock.log_input = Mock()
    mlflow_mock.set_tag = Mock()
    mlflow_mock.set_tags = Mock()
    mlflow_mock.sklearn = SimpleNamespace(
        log_model=Mock(return_value=SimpleNamespace(model_uri="models:/test-model/1"))
    )
    mlflow_mock.data = SimpleNamespace(from_pandas=Mock(return_value="dataset"))

    client_mock = Mock()

    monkeypatch.setattr(tracker_module, "mlflow", mlflow_mock)
    monkeypatch.setattr(tracker_module, "MlflowClient", Mock(return_value=client_mock))

    tracker = tracker_module.ExperimentTracker(
        experiment_name="exp-test",
        tracking_uri="sqlite:///mlruns.db",
        artifact_location="artifacts://bucket/path"
    )
    return tracker, mlflow_mock, client_mock


class TestExperimentTrackerExtended:
    def test_get_or_create_experiment_creates_when_missing(self, monkeypatch):
        experiment = SimpleNamespace(experiment_id="exp-created")
        mlflow_mock = Mock()
        mlflow_mock.set_tracking_uri = Mock()
        mlflow_mock.get_experiment_by_name = Mock(return_value=None)
        mlflow_mock.create_experiment = Mock(return_value="exp-created")
        mlflow_mock.get_experiment = Mock(return_value=experiment)
        mlflow_mock.set_experiment = Mock()

        monkeypatch.setattr(tracker_module, "mlflow", mlflow_mock)
        monkeypatch.setattr(tracker_module, "MlflowClient", Mock(return_value=Mock()))

        tracker = tracker_module.ExperimentTracker(
            experiment_name="exp-test",
            tracking_uri="sqlite:///mlruns.db",
            artifact_location="artifacts://bucket/path"
        )

        assert tracker.experiment.experiment_id == "exp-created"
        mlflow_mock.create_experiment.assert_called_once_with(
            "exp-test",
            artifact_location="artifacts://bucket/path"
        )

    def test_run_lifecycle_and_logging_helpers(self, monkeypatch):
        tracker, mlflow_mock, _ = _build_tracker(monkeypatch)

        tracker.start_run(tags={"env": "test"}, description="descricao-run")
        start_kwargs = mlflow_mock.start_run.call_args.kwargs
        assert start_kwargs["tags"]["project"] == "passos-magicos"
        assert start_kwargs["tags"]["env"] == "test"
        assert start_kwargs["tags"]["mlflow.note.content"] == "descricao-run"
        assert tracker.get_run_id() == "run-1"
        assert tracker.get_artifact_uri() == "artifacts://run-1"

        tracker.log_params({"alpha": 0.1, "complex_value": [1, 2]})
        logged_params = mlflow_mock.log_params.call_args.args[0]
        assert logged_params["alpha"] == 0.1
        assert logged_params["complex_value"] == "[1, 2]"

        tracker.log_metrics({"f1": 0.9}, step=2)
        tracker.log_metric("roc_auc", 0.93, step=3)
        tracker.log_artifact("local/file.txt", "artifacts")
        tracker.log_artifacts("local/dir", "artifacts")
        tracker.log_dict({"k": "v"}, "config.json")
        tracker.log_figure(Mock(), "figure.png")
        tracker.log_training_dataset(
            pd.DataFrame({"feature": [1, 2], "target": [0, 1]}),
            name="train_ds",
            targets="target"
        )
        tracker.set_tag("team", "mle")
        tracker.set_tags({"scope": "test", "phase": "qa"})

        tracker.end_run(status="FAILED")
        mlflow_mock.end_run.assert_called_once_with(status="FAILED")
        assert tracker.get_run_id() is None
        assert tracker.get_artifact_uri() is None

    def test_log_model_and_signature_helper(self, monkeypatch):
        tracker, mlflow_mock, _ = _build_tracker(monkeypatch)

        model = Mock()
        model_info = tracker.log_model(model=model, artifact_path="model")
        assert model_info.model_uri == "models:/test-model/1"
        default_requirements = mlflow_mock.sklearn.log_model.call_args.kwargs["pip_requirements"]
        assert "scikit-learn>=1.3.0" in default_requirements

        tracker.log_model(
            model=model,
            artifact_path="model-custom",
            pip_requirements=["xgboost==2.0.0"]
        )
        assert mlflow_mock.sklearn.log_model.call_args.kwargs["pip_requirements"] == ["xgboost==2.0.0"]

        tracker.log_model = Mock(return_value=SimpleNamespace(model_uri="models:/test-model/2"))
        model.predict.return_value = np.array([1, 0])
        X_sample = np.array([[1.0, 2.0], [3.0, 4.0]])

        with patch("mlflow.models.signature.infer_signature", return_value="sig") as infer_signature:
            tracker.log_sklearn_model_with_signature(
                model=model,
                X_sample=X_sample,
                y_sample=np.array([1, 0]),
                artifact_path="model-sig",
                registered_model_name="test-model"
            )

        infer_signature.assert_called_once()
        tracker.log_model.assert_called_once()
        assert np.array_equal(tracker.log_model.call_args.kwargs["input_example"], X_sample[:1])

    def test_list_runs_best_run_and_compare(self, monkeypatch):
        tracker, _, client_mock = _build_tracker(monkeypatch)

        run_obj = SimpleNamespace(
            data=SimpleNamespace(params={"model": "rf"}, metrics={"f1": 0.9}, tags={"env": "test"}),
            info=SimpleNamespace(status="FINISHED", start_time=123456)
        )

        client_mock.search_runs.return_value = [run_obj]
        runs = tracker.list_runs()
        assert runs == [run_obj]
        client_mock.search_runs.assert_called_with(
            experiment_ids=["exp-1"],
            max_results=100,
            order_by=["start_time DESC"]
        )

        best_asc = tracker.get_best_run(metric="f1", ascending=True)
        assert best_asc == run_obj

        client_mock.search_runs.return_value = []
        best_desc = tracker.get_best_run(metric="f1", ascending=False)
        assert best_desc is None

        client_mock.get_run.side_effect = [run_obj, run_obj]
        comparison = tracker.compare_runs(["run-a", "run-b"])
        assert set(comparison.keys()) == {"run-a", "run-b"}
        assert comparison["run-a"]["metrics"]["f1"] == 0.9
