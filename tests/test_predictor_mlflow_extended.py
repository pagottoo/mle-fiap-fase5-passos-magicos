"""
Testes estendidos para fluxos MLflow e fallback do ModelPredictor.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pytest

import src.models.predictor as predictor_module
from src.models.predictor import ModelPredictor


class _DummyModel:
    def __init__(self):
        self.last_X = None

    def predict(self, X):
        self.last_X = X
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])


class TestModelPredictorMlflowExtended:
    def test_load_from_mlflow_success(self, monkeypatch):
        dummy_model = _DummyModel()
        version = SimpleNamespace(version="7", run_id="run-7")
        run = SimpleNamespace(data=SimpleNamespace(params={"model_type": "rf"}, metrics={"f1": 0.91}))

        mlflow_mock = Mock()
        mlflow_mock.sklearn = SimpleNamespace(load_model=Mock(return_value=dummy_model))
        mlflow_mock.tracking = SimpleNamespace(MlflowClient=Mock(return_value=Mock(get_run=Mock(return_value=run))))

        registry_mock = Mock()
        registry_mock.get_latest_versions.return_value = [version]

        monkeypatch.setattr(predictor_module, "MLFLOW_AVAILABLE", True)
        monkeypatch.setattr(predictor_module, "mlflow", mlflow_mock, raising=False)
        monkeypatch.setattr(predictor_module, "ModelRegistry", Mock(return_value=registry_mock), raising=False)

        predictor = ModelPredictor(
            model_path=Path("nao-usado.joblib"),
            mlflow_model_name="ponto-virada",
            mlflow_stage="Staging"
        )

        assert predictor.loaded_from == "mlflow"
        assert predictor.artifacts["model_type"] == "rf"
        assert predictor.artifacts["version"] == "mlflow-v7"
        assert predictor.artifacts["trained_at"] == "mlflow"

    def test_load_model_fallback_to_file_when_mlflow_fails(self, monkeypatch):
        monkeypatch.setattr(predictor_module, "MLFLOW_AVAILABLE", True)
        monkeypatch.setattr(
            predictor_module.ModelPredictor,
            "_load_from_mlflow",
            Mock(side_effect=RuntimeError("mlflow indisponivel"))
        )

        def fake_load_from_file(self):
            self.loaded_from = "local"
            self.artifacts = {
                "model_type": "rf",
                "version": "1.0.0",
                "trained_at": "now",
                "metrics": {}
            }
            self.model = _DummyModel()

        monkeypatch.setattr(predictor_module.ModelPredictor, "_load_from_file", fake_load_from_file)

        predictor = ModelPredictor(mlflow_model_name="ponto-virada", mlflow_stage="Production")
        assert predictor.loaded_from == "local"
        assert predictor.artifacts["version"] == "1.0.0"

    def test_load_from_mlflow_raises_when_no_versions(self, monkeypatch):
        predictor = ModelPredictor.__new__(ModelPredictor)
        predictor.mlflow_model_name = "ponto-virada"
        predictor.mlflow_stage = "Production"
        predictor.loaded_from = None

        mlflow_mock = Mock()
        mlflow_mock.sklearn = SimpleNamespace(load_model=Mock(return_value=_DummyModel()))
        mlflow_mock.tracking = SimpleNamespace(MlflowClient=Mock(return_value=Mock(get_run=Mock())))

        registry_mock = Mock()
        registry_mock.get_latest_versions.return_value = []

        monkeypatch.setattr(predictor_module, "mlflow", mlflow_mock, raising=False)
        monkeypatch.setattr(predictor_module, "ModelRegistry", Mock(return_value=registry_mock), raising=False)

        with pytest.raises(ValueError, match="Nenhuma versão encontrada"):
            predictor._load_from_mlflow()

    def test_load_from_file_uses_default_latest_path(self, monkeypatch, tmp_path):
        artifacts = {
            "model": "dummy-model",
            "preprocessor": None,
            "feature_engineer": None,
            "model_type": "rf",
            "metrics": {"f1": 0.9},
            "version": "1.0.0",
            "trained_at": "2026-02-18"
        }
        default_model_path = tmp_path / "ponto_virada_model_latest.joblib"
        joblib.dump(artifacts, default_model_path)

        monkeypatch.setattr(predictor_module, "MODELS_DIR", tmp_path)

        predictor = ModelPredictor.__new__(ModelPredictor)
        predictor.model_path = None
        predictor.mlflow_model_name = None
        predictor.mlflow_stage = "Production"
        predictor.artifacts = None
        predictor.model = None
        predictor.preprocessor = None
        predictor.feature_engineer = None
        predictor.loaded_from = None

        predictor._load_from_file()

        assert predictor.model_path == default_model_path
        assert predictor.loaded_from == "local"
        assert predictor.artifacts["version"] == "1.0.0"

    def test_predict_uses_numpy_values_for_mlflow_loaded_model(self):
        predictor = ModelPredictor.__new__(ModelPredictor)
        predictor.model = _DummyModel()
        predictor.preprocessor = None
        predictor.feature_engineer = None
        predictor.loaded_from = "mlflow"
        predictor.artifacts = {"version": "mlflow-v1"}
        predictor.mlflow_model_name = "ponto-virada"
        predictor.mlflow_stage = "Production"

        result = predictor.predict({"INDE": 7.2, "IPV": 8.0})

        assert result["prediction"] == 1
        assert result["loaded_from"] == "mlflow"
        assert isinstance(predictor.model.last_X, np.ndarray)

    def test_get_feature_names_empty_and_from_mlflow_factory(self):
        predictor = ModelPredictor.__new__(ModelPredictor)
        predictor.feature_engineer = None
        assert predictor.get_feature_names() == []

        with patch.object(ModelPredictor, "__init__", return_value=None) as init_mock:
            created = ModelPredictor.from_mlflow(model_name="modelo-a", stage="Staging")
            init_mock.assert_called_once_with(mlflow_model_name="modelo-a", mlflow_stage="Staging")
            assert isinstance(created, ModelPredictor)
