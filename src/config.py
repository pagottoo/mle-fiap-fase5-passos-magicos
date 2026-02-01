"""
Configurações centralizadas do projeto
"""
import os
from pathlib import Path

# Diretórios base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dados"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Criar diretórios se não existirem
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Configurações do modelo
MODEL_CONFIG = {
    "model_name": "ponto_virada_model",
    "model_version": "1.0.0",
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5,
}

# Features para o modelo
TARGET_COLUMN = "ponto_virada_pred"

# Colunas numéricas importantes (índices de avaliação)
NUMERIC_FEATURES = [
    "INDE",  # Índice de Desenvolvimento Educacional
    "IAA",   # Índice de Autoavaliação
    "IEG",   # Índice de Engajamento
    "IPS",   # Índice Psicossocial
    "IDA",   # Índice de Desempenho Acadêmico
    "IPP",   # Índice de Ponto de Virada
    "IPV",   # Índice de Propensão à Virada
    "IAN",   # Índice de Adequação ao Nível
]

# Colunas categóricas
CATEGORICAL_FEATURES = [
    "INSTITUICAO_ENSINO_ALUNO",
    "FASE",
    "PEDRA",
    "PONTO_VIRADA",
]

# Configurações da API
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "debug": os.getenv("API_DEBUG", "false").lower() == "true",
}

# Configurações de logging
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "json",
}

# Configurações de monitoramento
MONITORING_CONFIG = {
    "drift_threshold": 0.1,
    "performance_window": 1000,  # Número de predições para calcular métricas
}
