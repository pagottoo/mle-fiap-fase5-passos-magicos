"""
Testes estendidos para integração MLflow no ModelTrainer.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import src.models.trainer as trainer_module
from src.config import MODEL_CONFIG
from src.models.trainer import ModelTrainer


def _balanced_xy():
    X = np.array([
        [0.1, 1.0], [0.2, 1.1], [0.3, 1.2], [0.4, 1.3],
        [0.5, 1.4], [0.6, 1.5], [0.7, 1.6], [0.8, 1.7],
        [0.9, 1.8], [1.0, 1.9], [1.1, 2.0], [1.2, 2.1],
        [1.3, 2.2], [1.4, 2.3], [1.5, 2.4], [1.6, 2.5],
        [1.7, 2.6], [1.8, 2.7], [1.9, 2.8], [2.0, 2.9]
    ])
    y = np.array([0, 1] * 10)
    return X, y


class TestModelTrainerMlflowExtended:
    def test_start_and_end_run_with_mlflow(self):
        trainer = ModelTrainer(enable_mlflow=False)
        tracker_mock = Mock()
        tracker_mock.get_run_id.return_value = "run-123"

        trainer.mlflow_enabled = True
        trainer.tracker = tracker_mock
        trainer.model_params = {"n_estimators": 100}

        trainer.start_run(description="execucao de teste")

        assert trainer.run_id == "run-123"
        assert tracker_mock.start_run.called
        start_kwargs = tracker_mock.start_run.call_args.kwargs
        assert start_kwargs["description"] == "execucao de teste"
        assert start_kwargs["run_name"].startswith("random_forest-")
        tracker_mock.log_params.assert_called_once()

        trainer.end_run(status="FAILED")
        tracker_mock.end_run.assert_called_once_with(status="FAILED")

    def test_cross_validate_train_and_evaluate_log_metrics(self):
        X, y = _balanced_xy()
        trainer = ModelTrainer(enable_mlflow=False)
        tracker_mock = Mock()

        trainer.mlflow_enabled = True
        trainer.tracker = tracker_mock

        cv_results = trainer.cross_validate(X, y, cv_folds=2)
        assert "accuracy" in cv_results
        assert tracker_mock.log_metric.call_count >= 10

        trainer.train(X, y)
        tracker_mock.log_params.assert_called()
        assert trainer.feature_importance

        metrics = trainer.evaluate(X, y)
        assert "f1_score" in metrics
        tracker_mock.log_metrics.assert_called_once()

    def test_save_model_replaces_existing_latest_link(self, monkeypatch, tmp_path):
        X, y = _balanced_xy()
        trainer = ModelTrainer(enable_mlflow=False)
        trainer.train(X, y)
        trainer.evaluate(X, y)

        monkeypatch.setattr(trainer_module, "MODELS_DIR", tmp_path)

        latest_path = tmp_path / f"{MODEL_CONFIG['model_name']}_latest.joblib"
        latest_path.write_text("conteudo-antigo")

        model_path = trainer.save_model(
            preprocessor={"ok": True},
            feature_engineer={"ok": True},
            model_name=MODEL_CONFIG["model_name"]
        )

        assert model_path.exists()
        assert latest_path.is_symlink()
        assert latest_path.resolve() == model_path.resolve()

    def test_log_model_to_mlflow_branches(self):
        X, y = _balanced_xy()
        trainer = ModelTrainer(enable_mlflow=False)

        trainer.mlflow_enabled = False
        assert trainer.log_model_to_mlflow(X_sample=X[:2], y_sample=y[:2]) is None

        tracker_mock = Mock()
        tracker_mock.log_sklearn_model_with_signature.return_value = SimpleNamespace(model_uri="models:/ponto/9")

        trainer.mlflow_enabled = True
        trainer.tracker = tracker_mock
        trainer.model = Mock()
        trainer.metrics = {"f1_score": 0.91}
        trainer.cv_results = {"accuracy": {"mean": 0.88, "std": 0.02}}
        trainer.feature_importance = {0: 0.6, 1: 0.4}

        model_uri = trainer.log_model_to_mlflow(X_sample=X[:2], y_sample=y[:2])
        assert model_uri == "models:/ponto/9"
        assert tracker_mock.log_dict.call_count == 3

        tracker_mock.log_sklearn_model_with_signature.side_effect = RuntimeError("erro")
        assert trainer.log_model_to_mlflow(X_sample=X[:2], y_sample=y[:2]) is None

    def test_promote_model_to_production_branches(self):
        trainer = ModelTrainer(enable_mlflow=False)

        trainer.mlflow_enabled = False
        trainer.model_registry = None
        assert trainer.promote_model_to_production(model_name="ponto") is False

        registry_mock = Mock()
        trainer.mlflow_enabled = True
        trainer.model_registry = registry_mock

        registry_mock.get_latest_versions.return_value = []
        assert trainer.promote_model_to_production(model_name="ponto") is False

        registry_mock.get_latest_versions.return_value = [SimpleNamespace(version="5")]
        registry_mock.promote_to_production.return_value = None
        assert trainer.promote_model_to_production(model_name="ponto") is True
        registry_mock.promote_to_production.assert_called_with("ponto", 5)

        registry_mock.promote_to_production.side_effect = RuntimeError("erro")
        assert trainer.promote_model_to_production(model_name="ponto", version=7) is False
