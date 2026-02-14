"""
Testes estendidos para backends do Feature Store (offline/online).
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.feature_store.offline_store import OfflineStore
from src.feature_store.online_store import OnlineStore


@pytest.fixture
def sample_features_df():
    return pd.DataFrame({
        "aluno_id": [1, 2, 3, 4],
        "INDE": [7.5, 6.0, 8.0, 5.5],
        "IPV": [8.0, 6.5, 9.0, 5.0],
        "PEDRA": ["Ametista", "Quartzo", "Ametista", "Topázio"],
        "target": [1, 0, 1, 0]
    })


class TestOfflineStoreExtended:
    def test_load_existing_metadata_file(self, tmp_path):
        store_path = tmp_path / "offline"
        store_path.mkdir(parents=True, exist_ok=True)

        existing_metadata = {
            "datasets": {"alunos": {"versions": []}},
            "created_at": "2026-01-01T00:00:00"
        }
        (store_path / "metadata.json").write_text(json.dumps(existing_metadata))

        store = OfflineStore(store_path)
        assert "alunos" in store._metadata["datasets"]

    def test_get_dataset_errors_and_filters(self, tmp_path, sample_features_df):
        store = OfflineStore(tmp_path / "offline")
        store.ingest(sample_features_df, "alunos", entity_column="aluno_id", version="v1")

        filtered = store.get_dataset(
            "alunos",
            version="v1",
            filters={"aluno_id": [1, 2, 99], "PEDRA": "Ametista", "inexistente": "x"}
        )
        assert filtered["aluno_id"].tolist() == [1]

        with pytest.raises(ValueError, match="Dataset 'dataset_inexistente' não encontrado"):
            store.get_dataset("dataset_inexistente")

        with pytest.raises(ValueError, match="Versão 'v999' não encontrada"):
            store.get_dataset("alunos", version="v999")

    def test_versions_info_and_delete_version_paths(self, tmp_path, sample_features_df):
        store = OfflineStore(tmp_path / "offline")
        path_v1 = store.ingest(sample_features_df, "alunos", version="v1")
        store.ingest(sample_features_df, "alunos", version="v2")

        versions = store.list_versions("alunos")
        assert len(versions) == 2
        assert store.list_versions("nao_existe") == []

        assert store.get_dataset_info("alunos") is not None
        assert store.get_dataset_info("nao_existe") is None

        assert store.delete_version("nao_existe", "v1") is False
        assert store.delete_version("alunos", "v999") is False
        assert store.delete_version("alunos", "v1") is True
        assert not Path(path_v1).exists()


class TestOnlineStoreExtended:
    @pytest.fixture
    def online_store(self, tmp_path):
        return OnlineStore(tmp_path / "online" / "features.db")

    def test_get_features_errors_vector_empty_and_table_info(self, online_store, sample_features_df):
        online_store.materialize(sample_features_df, "alunos", "aluno_id")

        with pytest.raises(ValueError, match="Tabela 'nao_existe' não encontrada"):
            online_store.get_features("nao_existe", [1])

        assert online_store.get_feature_vector("alunos", 999) == {}
        assert online_store.get_table_info("nao_existe") is None

        info = online_store.get_table_info("alunos")
        assert info is not None
        assert info["table_name"] == "alunos"
        assert info["num_rows"] == len(sample_features_df)

    def test_get_access_stats_with_and_without_table_filter(self, online_store, sample_features_df):
        online_store.materialize(sample_features_df, "alunos", "aluno_id")
        online_store.get_features("alunos", [1, 2])

        all_stats = online_store.get_access_stats(days=30)
        assert all_stats["period_days"] == 30
        assert len(all_stats["tables"]) == 1
        assert all_stats["tables"][0]["table_name"] == "alunos"
        assert all_stats["tables"][0]["access_count"] >= 2

        filtered_stats = online_store.get_access_stats(table_name="alunos", days=30)
        assert filtered_stats["tables"][0]["table_name"] == "alunos"

    def test_delete_table_returns_false_on_exception(self, online_store, monkeypatch):
        class BrokenCursor:
            def execute(self, *_args, **_kwargs):
                raise RuntimeError("db error")

        class BrokenConnection:
            def cursor(self):
                return BrokenCursor()

            def commit(self):
                return None

            def close(self):
                return None

        monkeypatch.setattr(online_store, "_get_connection", lambda: BrokenConnection())
        assert online_store.delete_table("alunos") is False
