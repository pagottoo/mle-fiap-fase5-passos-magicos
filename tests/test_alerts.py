"""
Testes adicionais para aumentar cobertura do módulo de alertas
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.monitoring.alerts import (
    AlertType,
    AlertSeverity,
    Alert,
    ConsoleChannel,
    AlertManager,
    get_alert_manager,
    send_alert
)


class TestAlertType:
    """Testes para AlertType enum."""
    
    def test_alert_types_exist(self):
        """Verifica que tipos de alerta existem."""
        assert AlertType.DATA_DRIFT
        assert AlertType.MODEL_PERFORMANCE
        assert AlertType.SYSTEM_ERROR
        assert AlertType.PREDICTION_DRIFT


class TestAlertSeverity:
    """Testes para AlertSeverity enum."""
    
    def test_alert_severities_exist(self):
        """Verifica que níveis de severidade existem."""
        assert AlertSeverity.INFO
        assert AlertSeverity.WARNING
        assert AlertSeverity.ERROR
        assert AlertSeverity.CRITICAL


class TestAlert:
    """Testes para classe Alert."""
    
    def test_create_alert(self):
        """Testa criação de alerta."""
        alert = Alert(
            alert_type=AlertType.DATA_DRIFT,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test"
        )
        
        assert alert.alert_type == AlertType.DATA_DRIFT
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test"
        assert alert.timestamp is not None
    
    def test_alert_with_metadata(self):
        """Testa alerta com metadados."""
        metadata = {"drift_score": 0.85, "feature": "INDE"}
        alert = Alert(
            alert_type=AlertType.DATA_DRIFT,
            severity=AlertSeverity.ERROR,
            title="Drift Alert",
            message="Drift detected",
            metadata=metadata
        )
        
        assert alert.metadata == metadata
        assert alert.metadata["drift_score"] == 0.85
    
    def test_alert_to_dict(self):
        """Testa conversão para dicionário."""
        alert = Alert(
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.CRITICAL,
            title="System Error",
            message="Critical error occurred"
        )
        
        d = alert.to_dict()
        
        assert "alert_type" in d
        assert "severity" in d
        assert "title" in d
        assert "message" in d
        assert "timestamp" in d
        assert d["alert_type"] == "system_error"
        assert d["severity"] == "critical"


class TestConsoleChannel:
    """Testes para ConsoleChannel."""
    
    def test_console_channel_configured(self):
        """Testa que ConsoleChannel está sempre configurado."""
        channel = ConsoleChannel()
        assert channel.is_configured() == True
    
    def test_console_channel_send(self):
        """Testa envio de alerta via ConsoleChannel."""
        channel = ConsoleChannel()
        alert = Alert(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test message"
        )
        
        result = channel.send(alert)
        assert result == True


class TestAlertManager:
    """Testes para AlertManager."""
    
    @pytest.fixture
    def manager(self):
        """Cria AlertManager para testes."""
        return AlertManager()
    
    def test_manager_has_console_channel(self, manager):
        """Testa que manager tem canal console por padrão."""
        assert len(manager.channels) >= 1
        assert any(isinstance(c, ConsoleChannel) for c in manager.channels)
    
    def test_send_alert_success(self, manager):
        """Testa envio de alerta com sucesso."""
        results = manager.send_alert(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test Alert",
            message="Test message"
        )
        
        assert isinstance(results, dict)
        assert "ConsoleChannel" in results
        assert results["ConsoleChannel"] == True
    
    def test_get_status(self, manager):
        """Testa obtenção de status."""
        status = manager.get_status()
        
        assert "channels" in status
        assert "total_channels" in status
        assert "configured_channels" in status
        assert "alerts_in_history" in status
        assert status["total_channels"] >= 1
    
    def test_get_history(self, manager):
        """Testa histórico de alertas."""
        # Enviar alguns alertas
        manager.send_alert(
            AlertType.CUSTOM,
            AlertSeverity.INFO,
            "Test 1",
            "Message 1"
        )
        manager.send_alert(
            AlertType.CUSTOM,
            AlertSeverity.WARNING,
            "Test 2",
            "Message 2"
        )
        
        history = manager.get_history(limit=10)
        
        assert len(history) >= 2
    
    def test_get_history_with_limit(self, manager):
        """Testa histórico com limite."""
        for i in range(5):
            manager.send_alert(
                AlertType.CUSTOM,
                AlertSeverity.INFO,
                f"Test {i}",
                f"Message {i}"
            )
        
        history = manager.get_history(limit=3)
        assert len(history) <= 3


class TestAlertManagerSingleton:
    """Testes para singleton do AlertManager."""
    
    def test_get_alert_manager_returns_same_instance(self):
        """Testa que get_alert_manager retorna mesma instância."""
        manager1 = get_alert_manager()
        manager2 = get_alert_manager()
        
        assert manager1 is manager2
    
    def test_send_alert_function(self):
        """Testa função de conveniência send_alert."""
        results = send_alert(
            AlertType.CUSTOM,
            AlertSeverity.INFO,
            "Convenience Test",
            "Using send_alert function"
        )
        
        assert isinstance(results, dict)
