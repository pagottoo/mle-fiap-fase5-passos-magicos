"""
Configuração de logging estruturado
"""
import sys
import structlog
from typing import Dict, Any
from datetime import datetime
import json
from pathlib import Path

from ..config import LOG_CONFIG, LOGS_DIR


def setup_logging():
    """
    Configura o logging estruturado para a aplicação.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def log_prediction(input_data: Dict[str, Any], prediction: Dict[str, Any]) -> None:
    """
    Loga uma predição para monitoramento e análise posterior.
    
    Args:
        input_data: Dados de entrada da predição
        prediction: Resultado da predição
    """
    logger = structlog.get_logger()
    
    log_entry = {
        "type": "prediction",
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "output": prediction
    }
    
    # Log estruturado
    logger.info("prediction_logged", **log_entry)
    
    # Salvar em arquivo para análise de drift
    predictions_log = LOGS_DIR / "predictions.jsonl"
    with open(predictions_log, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def get_recent_predictions(n: int = 1000) -> list:
    """
    Recupera as últimas n predições do log.
    
    Args:
        n: Número de predições a recuperar
        
    Returns:
        Lista de predições
    """
    predictions_log = LOGS_DIR / "predictions.jsonl"
    
    if not predictions_log.exists():
        return []
    
    predictions = []
    with open(predictions_log, "r") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    
    return predictions[-n:]
