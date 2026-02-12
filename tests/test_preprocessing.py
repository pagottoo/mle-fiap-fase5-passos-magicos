"""
Testes unitários para o módulo de pré-processamento
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Testes para a classe DataPreprocessor."""
    
    def test_init(self):
        """Testa inicialização do preprocessador."""
        preprocessor = DataPreprocessor()
        
        assert preprocessor.columns_to_drop == ["NOME"]
        assert preprocessor._fitted is False
        assert preprocessor._numeric_columns == []
        assert preprocessor._categorical_columns == []
    
    def test_load_data_csv(self, sample_csv):
        """Testa carregamento de arquivo CSV."""
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(sample_csv)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 5  # Mínimo esperado
        assert "INDE" in df.columns
    
    def test_load_data_invalid_format(self, tmp_path):
        """Testa erro ao carregar formato inválido."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("invalid")
        
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError, match="Formato de arquivo não suportado"):
            preprocessor.load_data(invalid_file)
    
    def test_filter_columns_by_year(self, sample_data):
        """Testa filtragem de colunas por ano."""
        # Adicionar colunas com sufixo de ano
        sample_data["INDE_2022"] = sample_data["INDE"]
        sample_data["INDE_2021"] = sample_data["INDE"] - 0.5
        
        preprocessor = DataPreprocessor()
        df_filtered = preprocessor.filter_columns_by_year(sample_data, "2022")
        
        assert "INDE" in df_filtered.columns  # Renomeado de INDE_2022
        assert "INDE_2021" not in df_filtered.columns
    
    def test_clean_data(self, sample_data):
        """Testa limpeza de dados."""
        # Adicionar linhas com muitos nulos
        sample_data.loc[len(sample_data)] = [None] * len(sample_data.columns)
        
        preprocessor = DataPreprocessor()
        df_cleaned = preprocessor.clean_data(sample_data)
        
        # Verifica que coluna NOME foi removida
        assert "NOME" not in df_cleaned.columns
        
        # Verifica que linhas com muitos nulos foram removidas
        assert len(df_cleaned) < len(sample_data)
    
    def test_handle_missing_values(self, sample_data):
        """Testa tratamento de valores ausentes."""
        # Adicionar valores nulos
        sample_data.loc[0, "INDE"] = None
        sample_data.loc[1, "INSTITUICAO_ENSINO_ALUNO"] = None
        
        preprocessor = DataPreprocessor()
        df_handled = preprocessor.handle_missing_values(sample_data)
        
        # Verifica que não há mais nulos
        assert df_handled["INDE"].isnull().sum() == 0
        assert df_handled["INSTITUICAO_ENSINO_ALUNO"].isnull().sum() == 0
        
        # Verifica que categóricas foram preenchidas com "Desconhecido"
        assert "Desconhecido" in df_handled["INSTITUICAO_ENSINO_ALUNO"].values
    
    def test_fit(self, sample_data):
        """Testa ajuste do preprocessador."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_data)
        
        assert preprocessor._fitted is True
        assert len(preprocessor._numeric_columns) > 0
        assert len(preprocessor._categorical_columns) > 0
    
    def test_transform_without_fit(self, sample_data):
        """Testa erro ao transformar sem ajustar."""
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError, match="precisa ser ajustado"):
            preprocessor.transform(sample_data)
    
    def test_fit_transform(self, sample_data):
        """Testa fit_transform."""
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.fit_transform(sample_data)
        
        assert preprocessor._fitted is True
        assert isinstance(df_processed, pd.DataFrame)
        assert "NOME" not in df_processed.columns


class TestDataPreprocessorIntegration:
    """Testes de integração para DataPreprocessor."""
    
    def test_prepare_dataset(self, sample_csv):
        """Testa pipeline completo de preparação."""
        preprocessor = DataPreprocessor()
        df = preprocessor.prepare_dataset(sample_csv, year="2022")
        
        assert isinstance(df, pd.DataFrame)
        assert preprocessor._fitted is True
        assert len(df) > 0
