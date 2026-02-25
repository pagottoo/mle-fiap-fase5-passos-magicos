"""
Testes unitários para a API FastAPI
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Mock do predictor antes de importar a API
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAPIEndpoints:
    """Testes para os endpoints da API."""
    
    @pytest.fixture
    def mock_predictor(self):
        """Mock do ModelPredictor."""
        predictor = Mock()
        predictor.predict.return_value = {
            "prediction": 1,
            "label": "Ponto de Virada Provável",
            "probability_no_turning_point": 0.15,
            "probability_turning_point": 0.85,
            "confidence": 0.85,
            "model_version": "1.0.0"
        }
        predictor.get_model_info.return_value = {
            "model_type": "random_forest",
            "version": "1.0.0",
            "trained_at": "2024-01-01_120000",
            "metrics": {"accuracy": 0.9, "f1_score": 0.85}
        }
        predictor.get_feature_names.return_value = ["INDE_scaled", "IAA_scaled"]
        return predictor
    
    @pytest.fixture
    def client(self, mock_predictor):
        """Cliente de teste com predictor mockado."""
        with patch("api.main.predictor", mock_predictor):
            with patch("api.main.ModelPredictor", return_value=mock_predictor):
                from api.main import app
                yield TestClient(app)
    
    def test_root(self, client):
        """Testa endpoint raiz."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
    
    def test_health_check(self, client):
        """Testa health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
    
    def test_model_info(self, client, mock_predictor):
        """Testa endpoint de informações do modelo."""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "random_forest"
        assert data["version"] == "1.0.0"
    
    def test_predict(self, client, mock_predictor, sample_input):
        """Testa endpoint de predição."""
        response = client.post("/predict", json=sample_input)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "label" in data
        assert "probability_no_turning_point" in data
        assert "probability_turning_point" in data
        assert "confidence" in data
        assert "model_version" in data
        assert "timestamp" in data
    
    def test_predict_with_minimal_data(self, client, mock_predictor, sample_input):
        """Testa predição com dados mínimos obrigatórios."""
        # Dados mínimos obrigatórios para predição
        minimal_input = {
            "INDE": 7.5,
            "IAA": 8.0,
            "IEG": 7.0,
            "IPS": 6.5,
            "IDA": 7.2,
            "IPP": 6.8,
            "IPV": 7.5,
            "IAN": 5.0,
            "FASE": "Fase 7",
            "PEDRA": "Ametista",
            "BOLSISTA": "Não"
        }
        
        response = client.post("/predict", json=minimal_input)
        
        assert response.status_code == 200
    
    def test_predict_batch(self, client, mock_predictor, sample_input):
        """Testa predição em lote."""
        batch_request = {
            "students": [sample_input, sample_input]
        }
        
        response = client.post("/predict/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 2
    
    def test_get_features(self, client, mock_predictor):
        """Testa endpoint de features."""
        response = client.get("/features")
        
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert isinstance(data["features"], list)
    
    def test_get_metrics(self, client):
        """Testa endpoint de métricas."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "total_predictions" in data

    def test_get_prometheus_metrics(self, client):
        """Testa endpoint de métricas no formato Prometheus."""
        response = client.get("/metrics/prometheus")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "passos_magicos_api_requests_total" in response.text


class TestAPIWithoutModel:
    """Testes para API sem modelo carregado."""
    
    @pytest.fixture
    def client_no_model(self):
        """Cliente de teste sem modelo."""
        with patch("api.main.predictor", None):
            from api.main import app
            yield TestClient(app)
    
    def test_health_degraded(self, client_no_model):
        """Testa que health retorna degraded sem modelo."""
        response = client_no_model.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
    
    def test_predict_without_model(self, client_no_model, sample_input):
        """Testa erro ao predizer sem modelo."""
        response = client_no_model.post("/predict", json=sample_input)
        
        assert response.status_code == 503
        assert "Modelo não carregado" in response.json()["detail"]
    
    def test_model_info_without_model(self, client_no_model):
        """Testa erro ao obter info sem modelo."""
        response = client_no_model.get("/model/info")
        
        assert response.status_code == 503


class TestAPIValidation:
    """Testes de validação da API."""
    
    @pytest.fixture
    def mock_predictor(self):
        """Mock do ModelPredictor."""
        predictor = Mock()
        predictor.predict.return_value = {
            "prediction": 1,
            "label": "Ponto de Virada Provável",
            "probability_no_turning_point": 0.15,
            "probability_turning_point": 0.85,
            "confidence": 0.85,
            "model_version": "1.0.0"
        }
        return predictor
    
    @pytest.fixture
    def client(self, mock_predictor):
        """Cliente de teste."""
        with patch("api.main.predictor", mock_predictor):
            from api.main import app
            yield TestClient(app)
    
    def test_predict_with_invalid_types(self, client):
        """Testa validação de tipos."""
        invalid_input = {
            "INDE": "not_a_number"  # Deveria ser float
        }
        
        response = client.post("/predict", json=invalid_input)
        
        # FastAPI/Pydantic deve retornar erro de validação
        assert response.status_code == 422
    
    def test_predict_empty_body(self, client):
        """Testa predição com corpo vazio - deve retornar erro de validação."""
        response = client.post("/predict", json={})
        
        # Campos obrigatórios faltando - deve retornar 422
        assert response.status_code == 422
