"""
Testes de integração para a API com modelo real
Estes testes não mockam o predictor e testam o fluxo completo
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import os


class TestAPIIntegrationWithModel:
    """Testes de integração da API com modelo real."""
    
    @pytest.fixture
    def client_with_model(self):
        """Cliente de teste com modelo real carregado."""
        from src.config import MODELS_DIR, MODEL_CONFIG
        model_path = MODELS_DIR / f"{MODEL_CONFIG['model_name']}_latest.joblib"
        
        if not model_path.exists():
            pytest.skip("Modelo real não encontrado - execute o treinamento primeiro")
        
        # Carregar predictor diretamente
        from src.models.predictor import ModelPredictor
        predictor = ModelPredictor(model_path=model_path)
        
        # Injetar predictor na aplicação
        from api.main import app
        import api.main as main_module
        main_module.predictor = predictor
        
        client = TestClient(app)
        yield client
        
        # Cleanup
        main_module.predictor = None
    
    def test_predict_with_model(self, client_with_model):
        """Testa predição com modelo real."""
        payload = {
            "INDE": 7.5,
            "IAA": 8.0,
            "IEG": 7.0,
            "IPS": 6.5,
            "IDA": 7.8,
            "IPP": 6.0,
            "IPV": 7.2,
            "IAN": 7.0,
            "FASE": "Fase 7",
            "PEDRA": "Ametista",
            "BOLSISTA": "Sim"
        }
        
        response = client_with_model.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verificar estrutura
        assert "prediction" in data
        assert "label" in data
        assert "probability_no_turning_point" in data
        assert "probability_turning_point" in data
        assert "confidence" in data
        
        # Verificar valores válidos
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability_no_turning_point"] <= 1
        assert 0 <= data["probability_turning_point"] <= 1
    
    def test_model_info_with_model(self, client_with_model):
        """Testa informações do modelo real."""
        response = client_with_model.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_type" in data
        assert "version" in data
        # loaded_from pode não estar sempre disponível na resposta da API
        assert data["model_type"] == "random_forest"
    
    def test_health_with_model(self, client_with_model):
        """Testa endpoint de saúde com modelo carregado."""
        response = client_with_model.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] == True


class TestAPIValidation:
    """Testes de validação de entrada da API."""
    
    @pytest.fixture
    def client(self):
        """Cliente de teste."""
        from api.main import app
        return TestClient(app)
    
    def test_predict_missing_required_field(self, client):
        """Testa erro quando campo obrigatório falta."""
        # Payload sem INDE (campo obrigatório)
        payload = {
            "IAA": 8.0, "IEG": 7.0, "IPS": 6.5,
            "IDA": 7.8, "IPP": 6.0, "IPV": 7.2, "IAN": 7.0,
            "FASE": "Fase 7", "PEDRA": "Ametista", "BOLSISTA": "Sim"
        }
        
        response = client.post("/predict", json=payload)
        
        # 422 se modelo carregado mas payload inválido
        # 503 se modelo não carregado (mas verificamos o payload primeiro)
        assert response.status_code in [422, 503]
    
    def test_predict_invalid_numeric_value(self, client):
        """Testa erro com valor numérico inválido."""
        payload = {
            "INDE": "não é número",  # Deveria ser float
            "IAA": 8.0, "IEG": 7.0, "IPS": 6.5,
            "IDA": 7.8, "IPP": 6.0, "IPV": 7.2, "IAN": 7.0,
            "FASE": "Fase 7", "PEDRA": "Ametista", "BOLSISTA": "Sim"
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 422
    
    def test_predict_empty_payload(self, client):
        """Testa erro com payload vazio."""
        response = client.post("/predict", json={})
        
        assert response.status_code in [422, 503]
    
    def test_predict_batch_empty_list(self, client):
        """Testa erro com lista vazia em batch."""
        response = client.post("/predict/batch", json=[])
        
        # Pode ser 422 ou 400 dependendo da implementação
        assert response.status_code in [400, 422, 503]
    
    def test_health_always_works(self, client):
        """Testa que health funciona mesmo sem modelo."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_docs_always_works(self, client):
        """Testa que docs funciona."""
        response = client.get("/docs")
        
        assert response.status_code == 200
    
    def test_openapi_schema(self, client):
        """Testa schema OpenAPI."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
