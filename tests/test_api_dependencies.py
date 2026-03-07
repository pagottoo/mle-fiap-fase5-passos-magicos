"""
Unit tests for API dependency wiring and runtime config.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.main as main
import api.dependencies as deps


def test_create_predictor_local_default(monkeypatch):
    monkeypatch.delenv("MODEL_SOURCE", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    calls = {}

    class _FakePredictor:
        @classmethod
        def from_mlflow(cls, **kwargs):
            raise AssertionError("from_mlflow should not be called")

        def __init__(self, model_path=None):
            calls["model_path"] = model_path

    monkeypatch.setattr(deps, "ModelPredictor", _FakePredictor)

    predictor = deps._create_predictor_from_runtime_config()
    assert isinstance(predictor, _FakePredictor)
    assert calls["model_path"] is None


def test_create_predictor_with_model_path(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    calls = {}

    class _FakePredictor:
        @classmethod
        def from_mlflow(cls, **kwargs):
            raise AssertionError("from_mlflow should not be called")

        def __init__(self, model_path=None):
            calls["model_path"] = model_path

    monkeypatch.setattr(deps, "ModelPredictor", _FakePredictor)

    predictor = deps._create_predictor_from_runtime_config()
    assert isinstance(predictor, _FakePredictor)
    assert calls["model_path"] == Path(model_path)


def test_create_predictor_mlflow_success(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "pm-model")
    monkeypatch.setenv("MLFLOW_MODEL_STAGE", "Staging")

    sentinel = object()

    class _FakePredictor:
        @classmethod
        def from_mlflow(cls, **kwargs):
            assert kwargs["model_name"] == "pm-model"
            assert kwargs["stage"] == "Staging"
            return sentinel

        def __init__(self, model_path=None):
            raise AssertionError("constructor should not be called")

    monkeypatch.setattr(deps, "ModelPredictor", _FakePredictor)
    assert deps._create_predictor_from_runtime_config() is sentinel


def test_create_predictor_mlflow_failure_without_fallback(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.setenv("MODEL_FALLBACK_LOCAL", "false")
    monkeypatch.delenv("MODEL_PATH", raising=False)

    class _FakePredictor:
        @classmethod
        def from_mlflow(cls, **kwargs):
            raise RuntimeError("mlflow unavailable")

        def __init__(self, model_path=None):
            raise AssertionError("constructor should not be called")

    monkeypatch.setattr(deps, "ModelPredictor", _FakePredictor)

    with pytest.raises(RuntimeError, match="mlflow unavailable"):
        deps._create_predictor_from_runtime_config()


def test_create_predictor_mlflow_failure_with_fallback(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.setenv("MODEL_FALLBACK_LOCAL", "true")
    monkeypatch.delenv("MODEL_PATH", raising=False)

    calls = {}

    class _FakePredictor:
        @classmethod
        def from_mlflow(cls, **kwargs):
            raise RuntimeError("mlflow unavailable")

        def __init__(self, model_path=None):
            calls["model_path"] = model_path

    monkeypatch.setattr(deps, "ModelPredictor", _FakePredictor)
    predictor = deps._create_predictor_from_runtime_config()
    assert isinstance(predictor, _FakePredictor)
    assert calls["model_path"] is None


def test_init_app_state_success(monkeypatch):
    fake_predictor = SimpleNamespace(get_model_info=lambda: {"version": "x"})
    fake_feature_store = SimpleNamespace()

    monkeypatch.setattr(deps, "_create_predictor_from_runtime_config", lambda: fake_predictor)
    monkeypatch.setattr(deps, "FeatureStore", lambda: fake_feature_store)

    model_loaded = {}
    fs_loaded = {}
    monkeypatch.setattr(
        deps.metrics_collector,
        "set_model_loaded",
        lambda value: model_loaded.setdefault("value", value),
    )
    monkeypatch.setattr(
        deps.metrics_collector,
        "set_feature_store_loaded",
        lambda value: fs_loaded.setdefault("value", value),
    )

    predictor, feature_store, metrics = deps.init_app_state()
    assert predictor is fake_predictor
    assert feature_store is fake_feature_store
    assert metrics is deps.metrics_collector
    assert model_loaded["value"] is True
    assert fs_loaded["value"] is True


def test_init_app_state_failures(monkeypatch):
    monkeypatch.setattr(
        deps,
        "_create_predictor_from_runtime_config",
        lambda: (_ for _ in ()).throw(RuntimeError("no model")),
    )
    monkeypatch.setattr(
        deps,
        "FeatureStore",
        lambda: (_ for _ in ()).throw(RuntimeError("no feature store")),
    )

    model_loaded = {}
    fs_loaded = {}
    monkeypatch.setattr(
        deps.metrics_collector,
        "set_model_loaded",
        lambda value: model_loaded.setdefault("value", value),
    )
    monkeypatch.setattr(
        deps.metrics_collector,
        "set_feature_store_loaded",
        lambda value: fs_loaded.setdefault("value", value),
    )

    predictor, feature_store, metrics = deps.init_app_state()
    assert predictor is None
    assert feature_store is None
    assert metrics is deps.metrics_collector
    assert model_loaded["value"] is False
    assert fs_loaded["value"] is False


def test_dependency_getters():
    sentinel_predictor = object()
    sentinel_feature_store = object()

    deps.predictor = sentinel_predictor
    deps.feature_store = sentinel_feature_store

    assert deps.get_predictor() is sentinel_predictor
    assert deps.get_feature_store() is sentinel_feature_store
    assert deps.get_metrics_collector() is deps.metrics_collector


def test_reload_model_error_branch(monkeypatch):
    monkeypatch.setattr(
        "api.routes.admin.init_app_state",
        lambda: (_ for _ in ()).throw(RuntimeError("reload failure")),
    )

    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post("/admin/reload-model")

    assert response.status_code == 500
    assert "Falha ao recarregar modelo" in response.json()["detail"]
