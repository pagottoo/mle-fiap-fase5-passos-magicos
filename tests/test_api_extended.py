"""
Testes extras para aumentar cobertura de api/main.py.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as main


class _DummyPredictor:
    def predict(self, data):
        return {
            "prediction": 1,
            "label": "Ponto de Virada Provável",
            "probability_no_turning_point": 0.1,
            "probability_turning_point": 0.9,
            "confidence": 0.9,
            "model_version": "1.0.0",
        }

    def get_model_info(self):
        return {
            "model_type": "random_forest",
            "version": "1.0.0",
            "trained_at": "20260218_000000",
            "metrics": {"f1_score": 0.9},
        }

    def get_feature_names(self):
        return ["INDE_scaled", "IAA_scaled"]


class _DummyRegistry:
    def __init__(self, group):
        self._group = group

    def get_group(self, name):
        return self._group if name == "g1" else None


class _DummyFeatureStore:
    def __init__(self):
        self.registry = _DummyRegistry(
            SimpleNamespace(
                name="g1",
                description="Grupo de teste",
                features=["inde", "ipv"],
                entity="aluno",
            )
        )

    def get_status(self):
        return {"registry": {"features": 2, "groups": 1}}

    def list_features(self):
        return ["inde", "ipv"]

    def get_feature_definition(self, name):
        mapping = {
            "inde": SimpleNamespace(
                name="inde",
                dtype="numeric",
                description="INDE",
                source="INDE",
                transformation="scale",
                tags=["indice"],
            ),
            "ipv": SimpleNamespace(
                name="ipv",
                dtype="numeric",
                description="IPV",
                source="IPV",
                transformation="scale",
                tags=["indice"],
            ),
        }
        return mapping.get(name)

    def list_groups(self):
        return ["g1"]

    def get_feature_vector(self, table_name, aluno_id):
        if aluno_id == 404:
            return {}
        return {"INDE": 7.0, "IAA": 6.0, "IEG": 6.0, "IPS": 6.0, "IDA": 7.0, "IPP": 6.0, "IPV": 7.0, "IAN": 5.0, "FASE": "F1", "PEDRA": "Ametista", "BOLSISTA": "Sim"}

    def get_serving_features(self, table_name, ids):
        if ids == [999]:
            return pd.DataFrame()
        rows = []
        for i in ids:
            rows.append(
                {
                    "aluno_id": i,
                    "INDE": 7.0,
                    "IAA": 6.0,
                    "IEG": 6.0,
                    "IPS": 6.0,
                    "IDA": 7.0,
                    "IPP": 6.0,
                    "IPV": 7.0,
                    "IAN": 5.0,
                    "FASE": "F1",
                    "PEDRA": "Ametista",
                    "BOLSISTA": "Sim",
                }
            )
        return pd.DataFrame(rows)


@pytest.fixture
def client_ok(monkeypatch):
    dummy_predictor = _DummyPredictor()
    dummy_fs = _DummyFeatureStore()

    monkeypatch.setattr(main, "ModelPredictor", lambda: dummy_predictor)
    monkeypatch.setattr(main, "FeatureStore", lambda: dummy_fs)

    with TestClient(main.app, raise_server_exceptions=False) as client:
        main.metrics_collector.reset()
        yield client, dummy_predictor, dummy_fs


@pytest.fixture
def client_startup_failure(monkeypatch):
    def _raise_model():
        raise FileNotFoundError("sem modelo")

    def _raise_fs():
        raise RuntimeError("sem feature store")

    monkeypatch.setattr(main, "ModelPredictor", _raise_model)
    monkeypatch.setattr(main, "FeatureStore", _raise_fs)

    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client


class TestApiExtended:
    def test_startup_failure_paths(self, client_startup_failure):
        health = client_startup_failure.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"

        model_info = client_startup_failure.get("/model/info")
        assert model_info.status_code == 503

        features_status = client_startup_failure.get("/features/status")
        assert features_status.status_code == 503

    def test_alert_status_and_history_success(self, client_ok):
        client, _, _ = client_ok
        manager = SimpleNamespace(
            get_status=lambda: {
                "channels": [{"name": "ConsoleChannel", "configured": True}],
                "total_channels": 1,
                "configured_channels": 1,
                "alerts_in_history": 0,
                "recent_alerts": [],
            },
            get_history=lambda limit=50: [{"title": "x"}],
        )

        with patch("src.monitoring.alerts.get_alert_manager", return_value=manager):
            r1 = client.get("/alerts/status")
            assert r1.status_code == 200
            assert r1.json()["status"] == "configured"

            r2 = client.get("/alerts/history?limit=10")
            assert r2.status_code == 200
            assert r2.json()["count"] == 1

    def test_alert_status_and_history_error_paths(self, client_ok):
        client, _, _ = client_ok
        with patch("src.monitoring.alerts.get_alert_manager", side_effect=RuntimeError("boom")):
            r1 = client.get("/alerts/status")
            assert r1.status_code == 200
            assert r1.json()["status"] == "error"

            r2 = client.get("/alerts/history")
            assert r2.status_code == 500

    def test_alert_test_and_manual_send(self, client_ok):
        client, _, _ = client_ok
        manager = SimpleNamespace(
            send_alert=lambda **kwargs: {"ConsoleChannel": True},
            get_status=lambda: {"channels": [{"name": "ConsoleChannel", "configured": True}]},
        )

        with patch("src.monitoring.alerts.get_alert_manager", return_value=manager):
            r1 = client.post("/alerts/test", json={"message": "ok"})
            assert r1.status_code == 200
            assert r1.json()["success"] == {"ConsoleChannel": True}

            r2 = client.post(
                "/alerts/send",
                json={
                    "type": "api_error",
                    "severity": "ERROR",
                    "title": "Falha",
                    "message": "Algo deu errado",
                },
            )
            assert r2.status_code == 200
            assert r2.json()["success"] == {"ConsoleChannel": True}

    def test_reload_model_authorization_and_success(self, client_ok, monkeypatch):
        client, _, _ = client_ok

        monkeypatch.setenv("ADMIN_RELOAD_TOKEN", "top-secret")
        unauthorized = client.post("/admin/reload-model")
        assert unauthorized.status_code == 401

        reloaded_predictor = _DummyPredictor()
        with patch.object(main, "_create_predictor_from_runtime_config", return_value=reloaded_predictor):
            authorized = client.post(
                "/admin/reload-model",
                headers={"X-Admin-Token": "top-secret"}
            )

        assert authorized.status_code == 200
        body = authorized.json()
        assert body["status"] == "reloaded"
        assert body["model_loaded"] is True
        assert body["model_version"] == "1.0.0"

    def test_feature_store_endpoints(self, client_ok):
        client, _, _ = client_ok
        r1 = client.get("/features/status")
        assert r1.status_code == 200
        assert "registry" in r1.json()

        r2 = client.get("/features/registry")
        assert r2.status_code == 200
        assert r2.json()["count"] == 2

        r3 = client.get("/features/groups")
        assert r3.status_code == 200
        assert r3.json()["count"] == 1

    def test_predict_by_aluno_success_not_found_and_error(self, client_ok):
        client, predictor, _ = client_ok

        ok = client.get("/predict/aluno/1")
        assert ok.status_code == 200

        not_found = client.get("/predict/aluno/404")
        assert not_found.status_code == 404

        with patch.object(predictor, "predict", side_effect=RuntimeError("erro interno")):
            fail = client.get("/predict/aluno/1")
            assert fail.status_code == 500

    def test_predict_batch_by_ids_paths(self, client_ok):
        client, _, _ = client_ok
        ok = client.get("/predict/alunos?aluno_ids=1,2,3")
        assert ok.status_code == 200
        assert ok.json()["count"] == 3

        invalid = client.get("/predict/alunos?aluno_ids=a,b")
        assert invalid.status_code == 400

        empty = client.get("/predict/alunos?aluno_ids=999")
        # Comportamento atual: HTTPException(404) é capturada e convertida em 500 no endpoint.
        assert empty.status_code == 500

    def test_predict_batch_without_model(self, client_startup_failure):
        resp = client_startup_failure.post("/predict/batch", json={"students": []})
        assert resp.status_code == 503

    def test_global_exception_handler_response(self, monkeypatch):
        # Forçar exceção não tratada em /model/info (sem método get_model_info)
        broken_predictor = object()
        fs = _DummyFeatureStore()
        monkeypatch.setattr(main, "ModelPredictor", lambda: broken_predictor)
        monkeypatch.setattr(main, "FeatureStore", lambda: fs)

        with TestClient(main.app, raise_server_exceptions=False) as client:
            resp = client.get("/model/info")
            assert resp.status_code == 500
            body = resp.json()
            assert body["detail"] == "Internal server error"
