"""
Testes unitários para o módulo de monitoramento
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import tempfile

from src.monitoring.metrics import MetricsCollector
from src.monitoring.drift import DriftDetector


class TestMetricsCollector:
    """Testes para a classe MetricsCollector."""
    
    def test_init(self):
        """Testa inicialização do coletor."""
        collector = MetricsCollector()
        
        assert collector.total_requests == 0
        assert collector.total_predictions == 0
        assert isinstance(collector.start_time, datetime)
    
    def test_record_request(self):
        """Testa registro de requisição."""
        collector = MetricsCollector()
        
        collector.record_request(
            endpoint="/predict",
            method="POST",
            status_code=200,
            duration=0.1
        )
        
        assert collector.total_requests == 1
        assert "POST /predict" in collector.requests_by_endpoint
        assert 200 in collector.requests_by_status
        assert len(collector.request_durations) == 1
    
    def test_record_multiple_requests(self):
        """Testa múltiplos registros."""
        collector = MetricsCollector()
        
        for i in range(10):
            collector.record_request(
                endpoint="/predict",
                method="POST",
                status_code=200,
                duration=0.05 + i * 0.01
            )
        
        assert collector.total_requests == 10
        assert len(collector.request_durations) == 10
    
    def test_record_prediction(self):
        """Testa registro de predição."""
        collector = MetricsCollector()
        
        collector.record_prediction(0)
        collector.record_prediction(1)
        collector.record_prediction(0)
        
        assert collector.total_predictions == 3
        assert collector.predictions_by_class[0] == 2
        assert collector.predictions_by_class[1] == 1
    
    def test_get_metrics(self):
        """Testa obtenção de métricas."""
        collector = MetricsCollector()
        
        # Registrar alguns dados
        for _ in range(5):
            collector.record_request("/predict", "POST", 200, 0.1)
            collector.record_prediction(0)
        
        for _ in range(3):
            collector.record_prediction(1)
        
        metrics = collector.get_metrics()
        
        assert "uptime_seconds" in metrics
        assert "total_requests" in metrics
        assert "total_predictions" in metrics
        assert "predictions_by_class" in metrics
        assert "latency" in metrics
        assert metrics["total_requests"] == 5
        assert metrics["total_predictions"] == 8
    
    def test_latency_percentiles(self):
        """Testa cálculo de percentis de latência."""
        collector = MetricsCollector()
        
        # Adicionar durações variadas
        for i in range(100):
            collector.record_request("/test", "GET", 200, i / 100)
        
        metrics = collector.get_metrics()
        latency = metrics["latency"]
        
        assert latency["avg_ms"] > 0
        assert latency["p95_ms"] > latency["avg_ms"]
        assert latency["p99_ms"] >= latency["p95_ms"]
    
    def test_reset(self):
        """Testa reset das métricas."""
        collector = MetricsCollector()
        
        collector.record_request("/test", "GET", 200, 0.1)
        collector.record_prediction(1)
        
        collector.reset()
        
        assert collector.total_requests == 0
        assert collector.total_predictions == 0
    
    def test_thread_safety(self):
        """Testa thread safety básico."""
        import threading
        
        collector = MetricsCollector()
        
        def record_requests():
            for _ in range(100):
                collector.record_request("/test", "GET", 200, 0.01)
        
        threads = [threading.Thread(target=record_requests) for _ in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert collector.total_requests == 500


class TestDriftDetector:
    """Testes para a classe DriftDetector."""
    
    @pytest.fixture
    def reference_data(self):
        """Dados de referência para testes."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature_1": np.random.normal(10, 2, 100),
            "feature_2": np.random.normal(5, 1, 100),
            "feature_3": np.random.normal(0, 0.5, 100)
        })
    
    def test_init_without_reference(self):
        """Testa inicialização sem dados de referência."""
        detector = DriftDetector()
        
        assert detector.reference_data is None
        assert detector.reference_stats == {}
    
    def test_init_with_reference(self, reference_data):
        """Testa inicialização com dados de referência."""
        detector = DriftDetector(reference_data)
        
        assert detector.reference_data is not None
        assert len(detector.reference_stats) == 3
        assert "feature_1" in detector.reference_stats
    
    def test_compute_reference_stats(self, reference_data):
        """Testa cálculo de estatísticas de referência."""
        detector = DriftDetector(reference_data)
        
        stats = detector.reference_stats["feature_1"]
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
    
    def test_detect_data_drift_no_drift(self, reference_data):
        """Testa detecção quando não há drift."""
        detector = DriftDetector(reference_data)
        
        # Dados similares aos de referência
        np.random.seed(43)
        current_data = pd.DataFrame({
            "feature_1": np.random.normal(10, 2, 50),
            "feature_2": np.random.normal(5, 1, 50),
            "feature_3": np.random.normal(0, 0.5, 50)
        })
        
        result = detector.detect_data_drift(current_data)
        
        assert result["drift_detected"] == False
        assert result["features_analyzed"] == 3
    
    def test_detect_data_drift_with_drift(self, reference_data):
        """Testa detecção quando há drift."""
        detector = DriftDetector(reference_data)
        
        # Dados com distribuição muito diferente
        current_data = pd.DataFrame({
            "feature_1": np.random.normal(20, 2, 50),  # Média muito diferente
            "feature_2": np.random.normal(15, 1, 50),  # Média muito diferente
            "feature_3": np.random.normal(5, 0.5, 50)  # Média muito diferente
        })
        
        result = detector.detect_data_drift(current_data)
        
        assert result["features_with_drift"] > 0
    
    def test_detect_data_drift_no_reference(self):
        """Testa erro quando não há referência."""
        detector = DriftDetector()
        
        current_data = pd.DataFrame({"feature_1": [1, 2, 3]})
        result = detector.detect_data_drift(current_data)
        
        assert "error" in result
    
    def test_detect_prediction_drift_no_drift(self):
        """Testa detecção de drift em predições sem drift."""
        detector = DriftDetector()
        
        # Distribuição balanceada
        predictions = [0] * 50 + [1] * 50
        reference = {0: 0.5, 1: 0.5}
        
        result = detector.detect_prediction_drift(predictions, reference)
        
        assert result["drift_detected"] == False
        assert result["drift_score"] == 0
    
    def test_detect_prediction_drift_with_drift(self):
        """Testa detecção de drift em predições com drift."""
        detector = DriftDetector()
        
        # Distribuição muito desbalanceada
        predictions = [0] * 10 + [1] * 90
        reference = {0: 0.5, 1: 0.5}
        
        result = detector.detect_prediction_drift(predictions, reference)
        
        assert result["drift_score"] > 0.1
    
    def test_detect_prediction_drift_empty(self):
        """Testa com lista vazia."""
        detector = DriftDetector()
        
        result = detector.detect_prediction_drift([])
        
        assert "error" in result
    
    def test_get_drift_report(self, reference_data):
        """Testa geração de relatório de drift."""
        detector = DriftDetector(reference_data)
        
        report = detector.get_drift_report()
        
        assert "generated_at" in report
        assert "reference_features" in report
        assert "drift_threshold" in report
        assert "drift_history_count" in report
    
    def test_drift_history(self, reference_data):
        """Testa histórico de drift."""
        detector = DriftDetector(reference_data)
        
        # Fazer múltiplas detecções
        for _ in range(3):
            current = pd.DataFrame({
                "feature_1": np.random.normal(10, 2, 20),
                "feature_2": np.random.normal(5, 1, 20),
                "feature_3": np.random.normal(0, 0.5, 20)
            })
            detector.detect_data_drift(current)
        
        assert len(detector.drift_history) == 3


class TestDriftDetectorIntegration:
    """Testes de integração para DriftDetector."""
    
    def test_save_and_load_reference(self, tmp_path):
        """Testa salvar e carregar dados de referência."""
        # Criar detector com dados
        reference = pd.DataFrame({
            "feature_1": np.random.normal(10, 2, 100),
            "feature_2": np.random.normal(5, 1, 100)
        })
        
        detector1 = DriftDetector(reference)
        
        # Salvar
        filepath = tmp_path / "reference.parquet"
        detector1.save_reference_data(str(filepath))
        
        assert filepath.exists()
        
        # Carregar em novo detector
        detector2 = DriftDetector()
        detector2.load_reference_data(str(filepath))
        
        assert detector2.reference_data is not None
        assert len(detector2.reference_stats) == 2
