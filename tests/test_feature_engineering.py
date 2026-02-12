"""
Testes unitários para o módulo de feature engineering
"""
import pytest
import pandas as pd
import numpy as np

from src.data.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Testes para a classe FeatureEngineer."""
    
    def test_init(self):
        """Testa inicialização do feature engineer."""
        fe = FeatureEngineer()
        
        assert fe._fitted is False
        assert fe._feature_names == []
        assert fe.label_encoders == {}
    
    def test_create_target_variable_with_ponto_virada(self, sample_data):
        """Testa criação de target usando coluna PONTO_VIRADA."""
        # sample_data já tem PONTO_VIRADA
        fe = FeatureEngineer()
        df = fe.create_target_variable(sample_data)
        
        assert "ponto_virada_pred" in df.columns
        
        # Contar quantos "Sim" existem no sample_data original
        expected_positives = (sample_data["PONTO_VIRADA"] == "Sim").sum()
        assert df["ponto_virada_pred"].sum() == expected_positives
    
    def test_create_target_variable_with_ipv(self, sample_data):
        """Testa criação de target usando IPV."""
        # Remover coluna PONTO_VIRADA se existir
        df_test = sample_data.copy()
        if "PONTO_VIRADA" in df_test.columns:
            df_test = df_test.drop(columns=["PONTO_VIRADA"])
        
        # Gerar IPV aleatório para todos os registros
        np.random.seed(42)
        df_test["IPV"] = np.random.uniform(4.0, 9.0, len(df_test))
        
        fe = FeatureEngineer()
        df = fe.create_target_variable(df_test)
        
        assert "ponto_virada_pred" in df.columns
        
        # IPV >= 7.5 deve ser ponto_virada = 1
        high_ipv_mask = df_test["IPV"] >= 7.5
        if high_ipv_mask.any():
            assert df.loc[high_ipv_mask, "ponto_virada_pred"].all() == 1
    
    def test_create_target_variable_with_pedra(self, sample_data):
        """Testa criação de target usando PEDRA."""
        if "PONTO_VIRADA" in sample_data.columns:
            sample_data = sample_data.drop(columns=["PONTO_VIRADA"])
        sample_data["IPV"] = 5.0  # IPV baixo para não afetar
        sample_data["INDE"] = 5.0  # INDE baixo para não afetar
        
        fe = FeatureEngineer()
        df = fe.create_target_variable(sample_data)
        
        # Topázio e Ágata devem ser ponto_virada = 1
        topazio_mask = sample_data["PEDRA"].isin(["Topázio", "Ágata"])
        if topazio_mask.any():
            assert df.loc[topazio_mask, "ponto_virada_pred"].all() == 1
    
    def test_select_features(self, sample_data):
        """Testa seleção de features."""
        fe = FeatureEngineer()
        numeric, categorical = fe.select_features(sample_data)
        
        assert isinstance(numeric, list)
        assert isinstance(categorical, list)
        assert "INDE" in numeric
        assert "PEDRA" in categorical
    
    def test_encode_categorical(self, sample_data):
        """Testa encoding de variáveis categóricas."""
        fe = FeatureEngineer()
        categorical = ["PEDRA", "INSTITUICAO_ENSINO_ALUNO"]
        
        df = fe.encode_categorical(sample_data, categorical, fit=True)
        
        assert "PEDRA_encoded" in df.columns
        assert "INSTITUICAO_ENSINO_ALUNO_encoded" in df.columns
        assert "PEDRA" in fe.label_encoders
    
    def test_encode_categorical_transform(self, sample_data):
        """Testa transform de encoding após fit."""
        fe = FeatureEngineer()
        categorical = ["PEDRA"]
        
        # Fit
        df_fit = fe.encode_categorical(sample_data, categorical, fit=True)
        
        # Transform com novos dados
        new_data = sample_data.copy()
        df_transform = fe.encode_categorical(new_data, categorical, fit=False)
        
        assert "PEDRA_encoded" in df_transform.columns
    
    def test_encode_categorical_unknown_value(self, sample_data):
        """Testa encoding com valor desconhecido."""
        fe = FeatureEngineer()
        categorical = ["PEDRA"]
        
        # Fit com dados originais
        fe.encode_categorical(sample_data, categorical, fit=True)
        
        # Transform com valor desconhecido
        new_data = pd.DataFrame({"PEDRA": ["Diamante"]})  # Valor não visto no treino
        df = fe.encode_categorical(new_data, categorical, fit=False)
        
        # Deve mapear para -1
        assert df["PEDRA_encoded"].iloc[0] == -1
    
    def test_scale_numeric(self, sample_data):
        """Testa scaling de variáveis numéricas."""
        fe = FeatureEngineer()
        numeric = ["INDE", "IAA", "IEG"]
        
        df = fe.scale_numeric(sample_data, numeric, fit=True)
        
        assert "INDE_scaled" in df.columns
        assert "IAA_scaled" in df.columns
        
        # Valores escalados devem ter média ~0
        assert abs(df["INDE_scaled"].mean()) < 1e-10
    
    def test_scale_numeric_with_nan(self, sample_data):
        """Testa scaling com valores NaN."""
        sample_data.loc[0, "INDE"] = np.nan
        
        fe = FeatureEngineer()
        df = fe.scale_numeric(sample_data, ["INDE"], fit=True)
        
        # NaN deve ser preenchido com 0
        assert not df["INDE_scaled"].isnull().any()
    
    def test_fit(self, sample_data):
        """Testa fit do feature engineer."""
        fe = FeatureEngineer()
        fe.fit(sample_data)
        
        assert fe._fitted is True
        assert len(fe._numeric_features) > 0
        assert len(fe._feature_names) > 0
    
    def test_transform_without_fit(self, sample_data):
        """Testa erro ao transformar sem fit."""
        fe = FeatureEngineer()
        
        with pytest.raises(ValueError, match="precisa ser ajustado"):
            fe.transform(sample_data)
    
    def test_fit_transform(self, sample_data):
        """Testa fit_transform."""
        fe = FeatureEngineer()
        df = fe.fit_transform(sample_data)
        
        assert fe._fitted is True
        assert any("_scaled" in col for col in df.columns)
        assert any("_encoded" in col for col in df.columns)
    
    def test_get_feature_matrix(self, sample_data):
        """Testa extração de matriz de features."""
        fe = FeatureEngineer()
        df = fe.create_target_variable(sample_data)
        df = fe.fit_transform(df)
        
        X, y = fe.get_feature_matrix(df)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == len(sample_data)
        assert len(y) == len(sample_data)
    
    def test_feature_names_property(self, sample_data):
        """Testa property de feature_names."""
        fe = FeatureEngineer()
        fe.fit_transform(sample_data)
        
        names = fe.feature_names
        
        assert isinstance(names, list)
        assert len(names) > 0
