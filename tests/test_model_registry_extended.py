"""
Testes extras para aumentar cobertura de src/mlflow_tracking/model_registry.py.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException

from src.mlflow_tracking.model_registry import ModelRegistry, ModelStage
import src.mlflow_tracking.model_registry as model_registry_module


@pytest.fixture
def registry_with_mocks(tmp_path):
    mock_client = MagicMock()
    with patch("src.mlflow_tracking.model_registry.mlflow.set_tracking_uri"):
        with patch("src.mlflow_tracking.model_registry.MlflowClient", return_value=mock_client):
            registry = ModelRegistry(tracking_uri=str(tmp_path / "mlruns"))
    registry.client = mock_client
    return registry, mock_client


class TestModelRegistryExtended:
    def test_register_model_existing_name(self, registry_with_mocks):
        registry, client = registry_with_mocks
        client.get_registered_model.return_value = SimpleNamespace(name="m")
        model_version = SimpleNamespace(version="3")

        with patch("src.mlflow_tracking.model_registry.mlflow.register_model", return_value=model_version):
            mv = registry.register_model(
                model_uri="runs:/123/model",
                name="m",
                tags={"team": "mlops"},
                description="desc",
            )

        assert mv.version == "3"
        client.set_model_version_tag.assert_called_once_with("m", "3", "team", "mlops")
        client.update_model_version.assert_called_once()

    def test_register_model_creates_registered_model_if_missing(self, registry_with_mocks):
        registry, client = registry_with_mocks
        client.get_registered_model.side_effect = MlflowException("not found")
        model_version = SimpleNamespace(version="1")

        with patch("src.mlflow_tracking.model_registry.mlflow.register_model", return_value=model_version):
            registry.register_model(model_uri="runs:/abc/model", name="novo")

        client.create_registered_model.assert_called_once()

    def test_transition_model_stage_invalid(self, registry_with_mocks):
        registry, _ = registry_with_mocks
        with pytest.raises(ValueError):
            registry.transition_model_stage("m", 1, "INVALID")

    def test_transition_model_stage_valid(self, registry_with_mocks):
        registry, client = registry_with_mocks
        client.transition_model_version_stage.return_value = SimpleNamespace(version="2", current_stage="Production")

        result = registry.transition_model_stage("m", 2, ModelStage.PRODUCTION)
        assert result.version == "2"
        client.transition_model_version_stage.assert_called_once()

    def test_promote_helpers(self, registry_with_mocks):
        registry, client = registry_with_mocks
        client.transition_model_version_stage.return_value = SimpleNamespace(version="4")

        registry.promote_to_staging("m", 4)
        registry.promote_to_production("m", 4)
        registry.archive_model("m", 4)
        assert client.transition_model_version_stage.call_count == 3

    def test_get_latest_versions_handles_exception(self, registry_with_mocks):
        registry, client = registry_with_mocks
        client.get_latest_versions.side_effect = MlflowException("missing")

        versions = registry.get_latest_versions("not-found")
        assert versions == []

    def test_get_production_and_staging_version(self, registry_with_mocks):
        registry, _ = registry_with_mocks
        vprod = SimpleNamespace(version="8")
        vstag = SimpleNamespace(version="9")

        with patch.object(registry, "get_latest_versions", side_effect=[[vprod], [vstag], []]):
            assert registry.get_production_version("m").version == "8"
            assert registry.get_staging_version("m").version == "9"
            assert registry.get_production_version("m2") is None

    def test_load_model_uses_version_stage_or_latest(self, registry_with_mocks):
        registry, _ = registry_with_mocks
        load_model = MagicMock(return_value="MODEL")
        fake_mlflow = SimpleNamespace(
            sklearn=SimpleNamespace(load_model=load_model),
            set_tracking_uri=MagicMock(),
        )
        with patch.object(model_registry_module, "mlflow", fake_mlflow):
            assert registry.load_model(name="m", version=5) == "MODEL"
            load_model.assert_called_with("models:/m/5")

            registry.load_model(name="m", stage=ModelStage.STAGING)
            load_model.assert_called_with("models:/m/Staging")

            with patch.object(registry, "get_production_version", return_value=SimpleNamespace(version="11")):
                registry.load_model(name="m")
                load_model.assert_called_with("models:/m/Production")

            with patch.object(registry, "get_production_version", return_value=None):
                registry.load_model(name="m")
                load_model.assert_called_with("models:/m/latest")

    def test_load_production_and_staging_shortcuts(self, registry_with_mocks):
        registry, _ = registry_with_mocks
        with patch.object(registry, "load_model", return_value="ok") as load_model:
            assert registry.load_production_model("m") == "ok"
            assert registry.load_staging_model("m") == "ok"
            assert load_model.call_count == 2

    def test_model_version_details_and_compare(self, registry_with_mocks):
        registry, client = registry_with_mocks
        mv1 = SimpleNamespace(
            name="m",
            version="1",
            current_stage="Production",
            status="READY",
            source="runs:/1/model",
            run_id="run-1",
            creation_timestamp=1,
            last_updated_timestamp=2,
            description="d1",
            tags={"a": "b"},
        )
        mv2 = SimpleNamespace(
            name="m",
            version="2",
            current_stage="Staging",
            status="READY",
            source="runs:/2/model",
            run_id="run-2",
            creation_timestamp=3,
            last_updated_timestamp=4,
            description="d2",
            tags={},
        )
        client.get_model_version.side_effect = [mv1, mv2]
        client.get_run.side_effect = [
            SimpleNamespace(data=SimpleNamespace(metrics={"f1": 0.8}, params={"p": "1"})),
            SimpleNamespace(data=SimpleNamespace(metrics={"f1": 0.9}, params={"p": "2"})),
        ]

        details = registry.get_model_version_details("m", 1)
        assert details["name"] == "m"
        assert details["tags"]["a"] == "b"

        with patch.object(registry, "get_model_version_details", side_effect=[details, {**details, "run_id": "run-2", "stage": "Staging"}]):
            comparison = registry.compare_model_versions("m", 1, 2)
        assert comparison["version1"]["metrics"]["f1"] == 0.8
        assert comparison["version2"]["params"]["p"] == "2"

    def test_delete_search_uri_and_tags(self, registry_with_mocks):
        registry, client = registry_with_mocks
        client.search_model_versions.return_value = [SimpleNamespace(version="1")]
        client.search_registered_models.return_value = [SimpleNamespace(name="m")]
        client.get_latest_versions.return_value = [SimpleNamespace(current_stage="Production", version="1", status="READY")]

        registry.delete_model_version("m", 1)
        registry.delete_registered_model("m")
        versions = registry.search_model_versions("name='m'", 10)
        assert len(versions) == 1

        assert registry.get_model_uri("m", version=2) == "models:/m/2"
        assert registry.get_model_uri("m", stage="Production") == "models:/m/Production"
        assert registry.get_model_uri("m") == "models:/m/latest"

        registry.set_model_version_tag("m", 1, "k", "v")
        client.set_model_version_tag.assert_called_with("m", "1", "k", "v")

        status = registry.get_registry_status()
        assert status["total_models"] == 1
        assert status["models"][0]["name"] == "m"
