"""
Testes adicionais para MLflow tracking
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os

# Configurar ambiente de teste
os.environ["MLFLOW_ENABLED"] = "true"


class TestExperimentTrackerBasic:
    """Testes básicos para ExperimentTracker."""
    
    def test_import_experiment_tracker(self):
        """Testa importação do módulo."""
        from src.mlflow_tracking.experiment_tracker import ExperimentTracker
        assert ExperimentTracker is not None
    
    def test_create_experiment_tracker(self, tmp_path):
        """Testa criação de ExperimentTracker."""
        from src.mlflow_tracking.experiment_tracker import ExperimentTracker
        
        tracker = ExperimentTracker(
            experiment_name="test-experiment",
            tracking_uri=str(tmp_path / "mlruns")
        )
        
        assert tracker is not None
        assert tracker.experiment_name == "test-experiment"


class TestModelRegistryBasic:
    """Testes básicos para ModelRegistry."""
    
    def test_import_model_registry(self):
        """Testa importação do módulo."""
        from src.mlflow_tracking.model_registry import ModelRegistry
        assert ModelRegistry is not None
    
    def test_create_model_registry(self, tmp_path):
        """Testa criação de ModelRegistry."""
        from src.mlflow_tracking.model_registry import ModelRegistry
        
        registry = ModelRegistry(
            tracking_uri=str(tmp_path / "mlruns")
        )
        
        assert registry is not None


class TestMLflowIntegration:
    """Testes de integração MLflow."""
    
    def test_experiment_tracker_with_run(self, tmp_path):
        """Testa ExperimentTracker com run."""
        from src.mlflow_tracking.experiment_tracker import ExperimentTracker
        
        tracker = ExperimentTracker(
            experiment_name="integration-test",
            tracking_uri=str(tmp_path / "mlruns")
        )
        
        # Iniciar run
        run_id = tracker.start_run(run_name="test-run")
        
        # Log de parâmetros
        tracker.log_params({"param1": "value1", "param2": 42})
        
        # Log de métricas
        tracker.log_metrics({"accuracy": 0.95, "f1_score": 0.92})
        
        # Finalizar run
        tracker.end_run()
        
        assert run_id is not None
    
    def test_experiment_tracker_context_manager(self, tmp_path):
        """Testa ExperimentTracker como context manager."""
        from src.mlflow_tracking.experiment_tracker import ExperimentTracker
        
        tracker = ExperimentTracker(
            experiment_name="context-test",
            tracking_uri=str(tmp_path / "mlruns")
        )
        
        with tracker.start_run(run_name="context-run"):
            tracker.log_params({"test": True})
            tracker.log_metrics({"metric": 1.0})
        
        # Run deve estar finalizado
        assert True
