"""
Testes para o módulo de predição
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

from src.models.predictor import ModelPredictor
from src.config import MODELS_DIR, MODEL_CONFIG


class TestModelPredictor:
    """Testes para ModelPredictor."""
    
    @pytest.fixture
    def model_path(self):
        """Retorna o caminho do modelo real ou None."""
        path = MODELS_DIR / f"{MODEL_CONFIG['model_name']}_latest.joblib"
        return path if path.exists() else None
    
    @pytest.fixture
    def predictor(self, model_path, monkeypatch):
        """Cria instância do predictor (real ou mockado)."""
        if model_path:
            return ModelPredictor(model_path=model_path)
        
        # CI environment or no model: Mock it
        monkeypatch.setattr(ModelPredictor, "_load_model", lambda x: None)
        predictor = ModelPredictor()
        predictor.model = MagicMock()
        predictor.model.predict.return_value = np.array([1])
        predictor.model.predict_proba.return_value = np.array([[0.1, 0.9]])
        predictor.loaded_from = "local"
        predictor.artifacts = {
            "model_type": "random_forest", 
            "version": "1.0.0", 
            "trained_at": "2026-03-03",
            "metrics": {"f1_score": 0.92}
        }
        
        # Mock feature engineer
        predictor.feature_engineer = MagicMock()
        predictor.feature_engineer.feature_names = ["F1", "F2"]
        predictor.feature_engineer.get_feature_matrix.return_value = (np.array([[0.1, 0.2]]), None)
        predictor.feature_engineer.transform.side_effect = lambda df: df
        
        # Mock preprocessor
        predictor.preprocessor = MagicMock()
        predictor.preprocessor.handle_missing_values.side_effect = lambda df: df
        
        predictor.explainer = MagicMock()
        predictor.explainer.shap_values.return_value = [np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])]
        predictor.explainer.expected_value = [0.5, 0.5]
        return predictor
    
    def test_load_model_from_file(self, model_path):
        """Testa carregamento de modelo de arquivo local."""
        if not model_path:
            pytest.skip("No real model for load test")
        predictor = ModelPredictor(model_path=model_path)
        
        assert predictor.loaded_from == "local"
        assert predictor.model is not None
        assert predictor.artifacts is not None
        assert predictor.artifacts["model_type"] == "random_forest"
    
    def test_load_model_file_not_found(self, tmp_path, monkeypatch):
        """Testa erro quando arquivo de modelo não existe."""
        # Forçar falha real no construtor
        fake_path = tmp_path / "nonexistent_model.joblib"
        # Garantir que _load_model tente carregar
        with pytest.raises(FileNotFoundError):
            ModelPredictor(model_path=fake_path)
    
    def test_preprocess_input(self, predictor):
        """Testa pré-processamento de entrada."""
        input_data = {
            "INDE": 7.5, "IAA": 8.0, "IEG": 7.0, "IPS": 6.5,
            "IDA": 7.8, "IPP": 6.0, "IPV": 7.2, "IAN": 7.0,
            "FASE": "Fase 7", "PEDRA": "Ametista", "BOLSISTA": "Sim"
        }
        
        result = predictor.preprocess_input(input_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_predict(self, predictor):
        """Testa predição básica."""
        input_data = {
            "INDE": 7.5, "IAA": 8.0, "IEG": 7.0, "IPS": 6.5,
            "IDA": 7.8, "IPP": 6.0, "IPV": 7.2, "IAN": 7.0,
            "FASE": "Fase 7", "PEDRA": "Ametista", "BOLSISTA": "Sim"
        }
        
        result = predictor.predict(input_data)
        
        assert "prediction" in result
        assert "label" in result
        assert "probability_no_turning_point" in result
        assert "probability_turning_point" in result
        assert "confidence" in result
        assert "model_version" in result
        
        assert result["prediction"] in [0, 1]
        assert 0 <= result["probability_no_turning_point"] <= 1
        assert 0 <= result["probability_turning_point"] <= 1
    
    def test_predict_returns_valid_probabilities(self, predictor):
        """Testa que probabilidades somam ~1."""
        input_data = {
            "INDE": 7.5, "IAA": 8.0, "IEG": 7.0, "IPS": 6.5,
            "IDA": 7.8, "IPP": 6.0, "IPV": 7.2, "IAN": 7.0,
            "FASE": "Fase 7", "PEDRA": "Ametista", "BOLSISTA": "Sim"
        }
        
        result = predictor.predict(input_data)
        
        prob_sum = result["probability_no_turning_point"] + result["probability_turning_point"]
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_predict_high_performance_student(self, predictor):
        """Testa predição para aluno de alta performance."""
        input_data = {
            "INDE": 9.5, "IAA": 9.8, "IEG": 9.5, "IPS": 9.3,
            "IDA": 9.7, "IPP": 9.2, "IPV": 9.6, "IAN": 9.4,
            "FASE": "Fase 8", "PEDRA": "Quartzo", "BOLSISTA": "Sim"
        }
        
        result = predictor.predict(input_data)
        
        assert result["prediction"] in [0, 1]
    
    def test_predict_low_performance_student(self, predictor):
        """Testa predição para aluno de baixa performance."""
        input_data = {
            "INDE": 3.0, "IAA": 3.5, "IEG": 3.2, "IPS": 2.8,
            "IDA": 3.0, "IPP": 2.5, "IPV": 2.8, "IAN": 3.0,
            "FASE": "Fase 1", "PEDRA": "Topázio", "BOLSISTA": "Não"
        }
        
        result = predictor.predict(input_data)
        
        assert result["prediction"] in [0, 1]
    
    def test_get_model_info(self, predictor):
        """Testa obtenção de informações do modelo."""
        info = predictor.get_model_info()
        
        assert "model_type" in info
        assert "version" in info
        assert "trained_at" in info
        assert "loaded_from" in info
        
        assert info["model_type"] == "random_forest"
        assert info["loaded_from"] == "local"
    
    def test_predict_batch(self, predictor):
        """Testa predição em lote."""
        data_list = [
            {
                "INDE": 7.5, "IAA": 8.0, "IEG": 7.0, "IPS": 6.5,
                "IDA": 7.8, "IPP": 6.0, "IPV": 7.2, "IAN": 7.0,
                "FASE": "Fase 7", "PEDRA": "Ametista", "BOLSISTA": "Sim"
            },
            {
                "INDE": 5.0, "IAA": 5.0, "IEG": 5.0, "IPS": 5.0,
                "IDA": 5.0, "IPP": 5.0, "IPV": 5.0, "IAN": 5.0,
                "FASE": "Fase 4", "PEDRA": "Ametista", "BOLSISTA": "Não"
            }
        ]
        
        results = predictor.predict_batch(data_list)
        
        assert len(results) == 2
        
        for result in results:
            assert "prediction" in result
            assert result["prediction"] in [0, 1]
    
    def test_predict_batch_empty_list(self, predictor):
        """Testa predição em lote com lista vazia."""
        results = predictor.predict_batch([])
        
        assert len(results) == 0
    
    def test_labels_correct(self, predictor):
        """Testa que labels estão corretos."""
        input_data = {
            "INDE": 7.5, "IAA": 8.0, "IEG": 7.0, "IPS": 6.5,
            "IDA": 7.8, "IPP": 6.0, "IPV": 7.2, "IAN": 7.0,
            "FASE": "Fase 7", "PEDRA": "Ametista", "BOLSISTA": "Sim"
        }
        
        result = predictor.predict(input_data)
        
        if result["prediction"] == 0:
            assert result["label"] == "Ponto de Virada Improvável"
        else:
            assert result["label"] == "Ponto de Virada Provável"
    
    def test_preprocess_input_without_preprocessor(self, predictor):
        """Testa pré-processamento quando preprocessor não está disponível."""
        # Temporariamente remove preprocessor
        original = predictor.preprocessor
        predictor.preprocessor = None
        
        input_data = {"INDE": 7.5, "IAA": 8.0}
        
        result = predictor.preprocess_input(input_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        
        # Restaura
        predictor.preprocessor = original
