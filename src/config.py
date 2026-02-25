"""
Centralized project configuration.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dados"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create runtime directories if missing
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "model_name": "ponto_virada_model",
    "model_version": "1.0.0",
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5,
}

# Model target
TARGET_COLUMN = "ponto_virada_pred"

# Main numeric features (evaluation indexes)
NUMERIC_FEATURES = [
    "INDE",  # Educational Development Index
    "IAA",   # Self-Assessment Index
    "IEG",   # Engagement Index
    "IPS",   # Psychosocial Index
    "IDA",   # Academic Performance Index
    "IPP",   # Participation Index
    "IPV",   # Turning Point Propensity Index
    "IAN",   # Level Adequacy Index
]

# Categorical columns
CATEGORICAL_FEATURES = [
    "INSTITUICAO_ENSINO_ALUNO",
    "FASE",
    "PEDRA",
    "PONTO_VIRADA",
]

# API configuration
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "debug": os.getenv("API_DEBUG", "false").lower() == "true",
}

# Logging configuration
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "json",
}

# Monitoring configuration
MONITORING_CONFIG = {
    "drift_threshold": 0.1,
    "performance_window": 1000,  # Number of predictions used for metrics
}
