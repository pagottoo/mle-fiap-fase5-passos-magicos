"""
Structured logging configuration.
"""
import logging
import os
import sys
import structlog
from typing import Any, Dict, Optional
from datetime import datetime
import json

# OpenTelemetry Logging
from opentelemetry import _logs
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk.resources import Resource

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


def setup_logging(component: Optional[str] = None) -> None:
    """
    Configure structured logging for the application with optional OpenTelemetry support.
    """
    raw_level = os.getenv("LOG_LEVEL", LOG_CONFIG.get("level", "INFO")).upper()
    log_level = getattr(logging, raw_level, logging.INFO)
    service_name = os.getenv("LOG_SERVICE_NAME", "passos-magicos").strip() or "passos-magicos"
    environment = os.getenv("ENVIRONMENT", "local").strip() or "local"

    # 1. Configure Root Logger (Standard Output)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    # Use a simpler format for console if not using OTel
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level)

    # 2. Setup OpenTelemetry Logging SDK (Optional)
    otel_enabled = os.getenv("ENABLE_OTEL", "false").lower() == "true"
    if otel_enabled:
        try:
            resource = Resource.create({
                "service.name": service_name,
                "service.component": component or "api",
                "environment": environment
            })
            
            logger_provider = LoggerProvider(resource=resource)
            _logs.set_logger_provider(logger_provider)
            
            otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://passos-magicos-otel-collector.mle-system.svc.cluster.local:4317")
            exporter = OTLPLogExporter(endpoint=otel_endpoint, insecure=True)
            logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
            
            otel_handler = LoggingHandler(level=log_level, logger_provider=logger_provider)
            root_logger.addHandler(otel_handler)
        except Exception:
            # Silent fail for OTel in case of network/config issues
            pass

    # 3. Configure Structlog
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
            # Important: JSONRenderer must be last to provide a string to the standard log handlers
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


def log_feedback(feedback_data: Dict[str, Any]) -> None:
    """
    Log ground truth feedback for a prediction.

    Args:
        feedback_data: Feedback record containing prediction_id and actual_outcome.
    """
    logger = get_logger(component="feedback")
    
    log_entry = {
        "type": "feedback",
        "timestamp": datetime.now().isoformat(),
        **feedback_data
    }
    
    # Structured log entry
    logger.info("feedback_logged", **log_entry)

    # Persist feedback log
    feedback_log = LOGS_DIR / "feedback.jsonl"
    with open(feedback_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def get_recent_feedback(n: int = 1000) -> list:
    """
    Return the latest N feedback records from the log file.

    Args:
        n: Number of feedback records to retrieve.

    Returns:
        List of feedback records.
    """
    feedback_log = LOGS_DIR / "feedback.jsonl"
    
    if not feedback_log.exists():
        return []
    
    feedback = []
    with open(feedback_log, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                feedback.append(json.loads(line))
    
    return feedback[-n:]


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
