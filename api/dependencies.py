import os
from pathlib import Path
import structlog
from src.models import ModelPredictor
from src.monitoring.metrics import MetricsCollector
from src.feature_store import FeatureStore

logger = structlog.get_logger(component="api_dependencies")

# Singletons
metrics_collector = MetricsCollector()
predictor = None
feature_store = None

def _create_predictor_from_runtime_config() -> ModelPredictor:
    """Build the predictor using runtime strategy."""
    model_source = os.getenv("MODEL_SOURCE", "local").strip().lower()
    model_path_env = os.getenv("MODEL_PATH", "").strip()
    model_path = Path(model_path_env) if model_path_env else None

    mlflow_model_name = os.getenv("MLFLOW_MODEL_NAME", "passos-magicos-ponto-virada").strip()
    mlflow_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production").strip() or "Production"
    fallback_local = os.getenv("MODEL_FALLBACK_LOCAL", "true").strip().lower() == "true"

    if model_source == "mlflow":
        try:
            logger.info("loading_predictor_mlflow", model_name=mlflow_model_name, stage=mlflow_stage)
            return ModelPredictor.from_mlflow(model_name=mlflow_model_name, stage=mlflow_stage)
        except Exception as e:
            logger.warning("mlflow_predictor_load_failed", error=str(e), fallback_local=fallback_local)
            if not fallback_local:
                raise

    if model_path is not None:
        return ModelPredictor(model_path=model_path)
    return ModelPredictor()

def init_app_state():
    """Initialize global state for the API."""
    global predictor, feature_store
    
    # Load model
    try:
        predictor = _create_predictor_from_runtime_config()
        logger.info("model_loaded", status="success")
    except Exception as e:
        logger.error("model_load_error", error=str(e))
        predictor = None

    metrics_collector.set_model_loaded(predictor is not None)
    
    # Initialize Feature Store
    try:
        feature_store = FeatureStore()
        logger.info("feature_store_loaded", status="success")
    except Exception as e:
        logger.error("feature_store_error", error=str(e))
        feature_store = None

    metrics_collector.set_feature_store_loaded(feature_store is not None)
    
    return predictor, feature_store, metrics_collector

def get_predictor():
    return predictor

def get_feature_store():
    return feature_store

def get_metrics_collector():
    return metrics_collector
