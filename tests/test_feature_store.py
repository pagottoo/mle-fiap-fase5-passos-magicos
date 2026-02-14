"""
Testes para o Feature Store
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.feature_store import FeatureStore, FeatureRegistry, OfflineStore, OnlineStore
from src.feature_store.registry import FeatureDefinition, FeatureGroup


class TestFeatureRegistry:
    """Testes para o Feature Registry."""
    
    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Cria um registry temporário."""
        registry_path = tmp_path / "registry.json"
        return FeatureRegistry(registry_path)
    
    def test_init_creates_default_features(self, temp_registry):
        """Testa se features padrão são criadas."""
        features = temp_registry.list_features()
        assert len(features) > 0
        assert "inde" in features
        assert "ipv" in features
    
    def test_init_creates_default_groups(self, temp_registry):
        """Testa se grupos padrão são criados."""
        groups = temp_registry.list_groups()
        assert "indices_avaliacao" in groups
        assert "ponto_virada_features" in groups
    
    def test_register_feature(self, temp_registry):
        """Testa registro de feature."""
        feat = FeatureDefinition(
            name="test_feature",
            dtype="numeric",
            description="Feature de teste",
            source="TEST_COL"
        )
        temp_registry.register_feature(feat)
        
        retrieved = temp_registry.get_feature("test_feature")
        assert retrieved is not None
        assert retrieved.name == "test_feature"
        assert retrieved.source == "TEST_COL"
    
    def test_register_group(self, temp_registry):
        """Testa registro de grupo."""
        group = FeatureGroup(
            name="test_group",
            description="Grupo de teste",
            features=["inde", "iaa"],
            entity="aluno"
        )
        temp_registry.register_group(group)
        
        retrieved = temp_registry.get_group("test_group")
        assert retrieved is not None
        assert retrieved.name == "test_group"
        assert "inde" in retrieved.features
    
    def test_get_features_by_group(self, temp_registry):
        """Testa obtenção de features por grupo."""
        features = temp_registry.get_features_by_group("indices_avaliacao")
        assert len(features) > 0
        assert all(isinstance(f, FeatureDefinition) for f in features)
    
    def test_get_features_by_tag(self, temp_registry):
        """Testa obtenção de features por tag."""
        features = temp_registry.get_features_by_tag("indice")
        assert len(features) > 0
    
    def test_get_source_columns(self, temp_registry):
        """Testa mapeamento para colunas de origem."""
        mapping = temp_registry.get_source_columns(["inde", "ipv"])
        assert mapping["inde"] == "INDE"
        assert mapping["ipv"] == "IPV"


class TestOfflineStore:
    """Testes para o Offline Store."""
    
    @pytest.fixture
    def temp_store(self, tmp_path):
        """Cria um store temporário."""
        return OfflineStore(tmp_path / "offline")
    
    @pytest.fixture
    def sample_df(self):
        """DataFrame de exemplo."""
        return pd.DataFrame({
            "aluno_id": [1, 2, 3, 4, 5],
            "INDE": [7.5, 6.0, 8.0, 5.5, 7.0],
            "IPV": [8.0, 6.5, 9.0, 5.0, 7.5],
            "PEDRA": ["Ametista", "Quartzo", "Topázio", "Quartzo", "Ágata"]
        })
    
    def test_ingest(self, temp_store, sample_df):
        """Testa ingestão de dados."""
        path = temp_store.ingest(sample_df, "test_dataset", "aluno_id")
        assert Path(path).exists()
    
    def test_get_dataset(self, temp_store, sample_df):
        """Testa recuperação de dataset."""
        temp_store.ingest(sample_df, "test_dataset", "aluno_id")
        
        df = temp_store.get_dataset("test_dataset")
        assert len(df) == 5
        assert "INDE" in df.columns
    
    def test_get_dataset_with_columns(self, temp_store, sample_df):
        """Testa recuperação com seleção de colunas."""
        temp_store.ingest(sample_df, "test_dataset", "aluno_id")
        
        df = temp_store.get_dataset("test_dataset", columns=["aluno_id", "INDE"])
        assert len(df.columns) == 2
    
    def test_list_datasets(self, temp_store, sample_df):
        """Testa listagem de datasets."""
        temp_store.ingest(sample_df, "dataset1", "aluno_id")
        temp_store.ingest(sample_df, "dataset2", "aluno_id")
        
        datasets = temp_store.list_datasets()
        assert "dataset1" in datasets
        assert "dataset2" in datasets
    
    def test_compute_statistics(self, temp_store, sample_df):
        """Testa cálculo de estatísticas."""
        temp_store.ingest(sample_df, "test_dataset", "aluno_id")
        
        stats = temp_store.compute_statistics("test_dataset")
        assert stats["num_rows"] == 5
        assert "INDE" in stats["columns"]


class TestOnlineStore:
    """Testes para o Online Store."""
    
    @pytest.fixture
    def temp_store(self, tmp_path):
        """Cria um store temporário."""
        store_dir = tmp_path / "online"
        store_dir.mkdir(parents=True, exist_ok=True)
        return OnlineStore(store_dir / "features.db")
    
    @pytest.fixture
    def sample_df(self):
        """DataFrame de exemplo."""
        return pd.DataFrame({
            "aluno_id": [1, 2, 3, 4, 5],
            "INDE": [7.5, 6.0, 8.0, 5.5, 7.0],
            "IPV": [8.0, 6.5, 9.0, 5.0, 7.5],
            "PEDRA": ["Ametista", "Quartzo", "Topázio", "Quartzo", "Ágata"]
        })
    
    def test_materialize(self, temp_store, sample_df):
        """Testa materialização de features."""
        count = temp_store.materialize(sample_df, "alunos", "aluno_id")
        assert count == 5
    
    def test_get_features(self, temp_store, sample_df):
        """Testa recuperação de features."""
        temp_store.materialize(sample_df, "alunos", "aluno_id")
        
        df = temp_store.get_features("alunos", [1, 2, 3])
        assert len(df) == 3
    
    def test_get_feature_vector(self, temp_store, sample_df):
        """Testa recuperação de vetor de features."""
        temp_store.materialize(sample_df, "alunos", "aluno_id")
        
        vector = temp_store.get_feature_vector("alunos", 1)
        assert vector["INDE"] == 7.5
        assert vector["IPV"] == 8.0
    
    def test_list_tables(self, temp_store, sample_df):
        """Testa listagem de tabelas."""
        temp_store.materialize(sample_df, "alunos", "aluno_id")
        
        tables = temp_store.list_tables()
        assert len(tables) == 1
        assert tables[0]["table_name"] == "alunos"
    
    def test_delete_table(self, temp_store, sample_df):
        """Testa remoção de tabela."""
        temp_store.materialize(sample_df, "alunos", "aluno_id")
        
        result = temp_store.delete_table("alunos")
        assert result is True
        
        tables = temp_store.list_tables()
        assert len(tables) == 0


class TestFeatureStore:
    """Testes para o Feature Store integrado."""
    
    @pytest.fixture
    def temp_feature_store(self, tmp_path):
        """Cria um feature store temporário."""
        return FeatureStore(tmp_path / "feature_store")
    
    @pytest.fixture
    def sample_df(self):
        """DataFrame de exemplo."""
        return pd.DataFrame({
            "aluno_id": [1, 2, 3, 4, 5],
            "INDE": [7.5, 6.0, 8.0, 5.5, 7.0],
            "IAA": [7.0, 5.5, 8.5, 5.0, 6.5],
            "IPV": [8.0, 6.5, 9.0, 5.0, 7.5],
            "PEDRA": ["Ametista", "Quartzo", "Topázio", "Quartzo", "Ágata"],
            "ponto_virada_pred": [1, 0, 1, 0, 1]
        })
    
    def test_init(self, temp_feature_store):
        """Testa inicialização."""
        assert temp_feature_store.registry is not None
        assert temp_feature_store.offline_store is not None
        assert temp_feature_store.online_store is not None
    
    def test_list_features(self, temp_feature_store):
        """Testa listagem de features."""
        features = temp_feature_store.list_features()
        assert len(features) > 0
    
    def test_ingest_training_data(self, temp_feature_store, sample_df):
        """Testa ingestão de dados de treinamento."""
        path = temp_feature_store.ingest_training_data(sample_df, "training_v1", "aluno_id")
        assert Path(path).exists()
    
    def test_get_training_data(self, temp_feature_store, sample_df):
        """Testa recuperação de dados de treinamento."""
        temp_feature_store.ingest_training_data(sample_df, "training_v1", "aluno_id")
        
        df = temp_feature_store.get_training_data("training_v1")
        assert len(df) == 5
    
    def test_materialize_and_serve(self, temp_feature_store, sample_df):
        """Testa materialização e serving."""
        temp_feature_store.materialize_for_serving(sample_df, "alunos", "aluno_id")
        
        df = temp_feature_store.get_serving_features("alunos", [1, 2])
        assert len(df) == 2
    
    def test_get_feature_vector(self, temp_feature_store, sample_df):
        """Testa obtenção de vetor de features."""
        temp_feature_store.materialize_for_serving(sample_df, "alunos", "aluno_id")
        
        vector = temp_feature_store.get_feature_vector("alunos", 1)
        assert "INDE" in vector
    
    def test_sync_offline_to_online(self, temp_feature_store, sample_df):
        """Testa sincronização offline -> online."""
        temp_feature_store.ingest_training_data(sample_df, "training_v1", "aluno_id")
        
        count = temp_feature_store.sync_offline_to_online("training_v1", "alunos_synced", "aluno_id")
        assert count == 5
    
    def test_get_status(self, temp_feature_store):
        """Testa obtenção de status."""
        status = temp_feature_store.get_status()
        assert "registry" in status
        assert "offline_store" in status
        assert "online_store" in status
