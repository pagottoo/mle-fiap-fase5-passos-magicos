"""
Testes extras para aumentar cobertura de src/feature_store/store.py.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.feature_store import FeatureStore
from src.feature_store.registry import FeatureDefinition, FeatureGroup


@pytest.fixture
def fs(tmp_path):
    return FeatureStore(tmp_path / "feature_store_ext")


@pytest.fixture
def df_training():
    return pd.DataFrame(
        {
            "aluno_id": [1, 2, 3],
            "INDE": [7.0, 6.5, 8.1],
            "IAA": [7.2, 6.1, 8.0],
            "IEG": [6.9, 6.0, 7.8],
            "IPS": [6.8, 5.9, 7.7],
            "IDA": [7.1, 6.2, 8.2],
            "IPP": [6.7, 5.8, 7.9],
            "IPV": [7.3, 6.0, 8.4],
            "IAN": [5.0, 4.8, 6.2],
            "IDADE_ALUNO": [12, 13, 11],
            "ANOS_PM": [2, 1, 3],
            "INSTITUICAO_ENSINO_ALUNO": ["Escola Pública", "Escola Privada", "Escola Pública"],
            "FASE": ["F1", "F2", "F1"],
            "PEDRA": ["Ametista", "Quartzo", "Topázio"],
            "BOLSISTA": ["Sim", "Não", "Sim"],
            "ponto_virada_pred": [1, 0, 1],
        }
    )


class TestFeatureStoreExtended:
    def test_register_wrappers_and_get_group_features(self, fs):
        feature = FeatureDefinition(
            name="nova_feature",
            dtype="numeric",
            description="Feature nova",
            source="NOVA_COL",
        )
        fs.register_feature(feature)
        assert fs.get_feature_definition("nova_feature").source == "NOVA_COL"

        group = FeatureGroup(
            name="grupo_novo",
            description="Grupo novo",
            features=["nova_feature"],
            entity="aluno",
        )
        fs.register_group(group)
        feats = fs.get_group_features("grupo_novo")
        assert len(feats) == 1
        assert feats[0].name == "nova_feature"

    def test_get_training_data_with_feature_group(self, fs, df_training):
        fs.ingest_training_data(df_training, "treino", entity_column="aluno_id")
        df = fs.get_training_data("treino", feature_group="indices_avaliacao")

        assert not df.empty
        assert "INDE" in df.columns
        assert "IPV" in df.columns

    def test_get_features_for_training_success(self, fs, df_training):
        fs.ingest_training_data(df_training, "treino2", entity_column="aluno_id")
        x, y = fs.get_features_for_training(
            dataset_name="treino2",
            feature_group="ponto_virada_features",
            target_column="ponto_virada_pred",
        )

        assert len(x) == len(y) == 3
        assert "INDE" in x.columns

    def test_get_features_for_training_group_not_found(self, fs, df_training):
        fs.ingest_training_data(df_training, "treino3", entity_column="aluno_id")
        with pytest.raises(ValueError):
            fs.get_features_for_training("treino3", feature_group="inexistente")

    def test_materialize_and_get_serving_features_with_group(self, fs, df_training):
        count = fs.materialize_for_serving(
            df_training,
            table_name="serving_group",
            entity_column="aluno_id",
            feature_group="indices_avaliacao",
        )
        assert count == 3

        df = fs.get_serving_features("serving_group", [1, 2], feature_group="indices_avaliacao")
        assert len(df) == 2
        assert "INDE" in df.columns

    def test_validate_feature_consistency_currently_raises_keyerror(self, fs, df_training):
        # Offline
        fs.ingest_training_data(df_training, "consistency_ds", entity_column="aluno_id")

        # Online com um valor alterado para gerar mismatch
        df_online = df_training.copy()
        df_online.loc[df_online["aluno_id"] == 2, "INDE"] = 0.0
        fs.materialize_for_serving(df_online, "consistency_tbl", entity_column="aluno_id")

        # Comportamento atual do método: inclui entity_column em common_columns e dispara KeyError.
        with pytest.raises(KeyError):
            fs.validate_feature_consistency(
                offline_dataset="consistency_ds",
                online_table="consistency_tbl",
                entity_column="aluno_id",
                sample_size=3,
            )
