"""
Structured logging configuration.
"""
import logging
import os
import sys
import structlog
from typing import Any, Dict
from datetime import datetime
import json

from ..config import LOG_CONFIG, LOGS_DIR


def _add_schema_defaults(service_name: str, environment: str):
    """
    Ensure a minimum schema for Loki/Grafana filtering.
    """

    def _processor(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        event_name = str(event_dict.get("event", "") or "").strip() or "log"
        event_dict["event"] = event_name
        event_dict.setdefault("message", event_name)

        event_dict.setdefault("service", service_name)
        event_dict.setdefault("environment", environment)

        component = event_dict.get("component")
        if not component:
            component = event_dict.get("logger") or event_dict.get("module") or "application"
            event_dict["component"] = component

        event_dict.setdefault("trace_id", None)
        event_dict.setdefault("run_id", None)
        event_dict.setdefault("model_version", None)
        return event_dict

    return _processor


def setup_logging() -> None:
    """
    Configure structured logging for the application.
    """
    raw_level = os.getenv("LOG_LEVEL", LOG_CONFIG.get("level", "INFO")).upper()
    log_level = getattr(logging, raw_level, logging.INFO)
    service_name = os.getenv("LOG_SERVICE_NAME", "passos-magicos").strip() or "passos-magicos"
    environment = os.getenv("ENVIRONMENT", "local").strip() or "local"

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            _add_schema_defaults(service_name=service_name, environment=environment),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(component: str | None = None, **context: Any):
    """
    Return a structured logger with optional bound context.
    """
    logger = structlog.get_logger(component) if component else structlog.get_logger()
    return logger.bind(**context) if context else logger


def log_prediction(input_data: Dict[str, Any], prediction: Dict[str, Any]) -> None:
    """
    Log one prediction for monitoring and later analysis.

    Args:
        input_data: Prediction input payload.
        prediction: Prediction output payload.
    """
    logger = get_logger(component="prediction")
    
    log_entry = {
        "type": "prediction",
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "output": prediction,
        "model_version": prediction.get("model_version"),
    }
    
    # Structured log entry
    logger.info("prediction_logged", **log_entry)

    # Persist prediction log for drift analysis
    predictions_log = LOGS_DIR / "predictions.jsonl"
    with open(predictions_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def get_recent_predictions(n: int = 1000) -> list:
    """
    Return the latest N predictions from the log file.

    Args:
        n: Number of predictions to retrieve.

    Returns:
        List of prediction records.
    """
    predictions_log = LOGS_DIR / "predictions.jsonl"
    
    if not predictions_log.exists():
        return []
    
    predictions = []
    with open(predictions_log, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    
    return predictions[-n:]
