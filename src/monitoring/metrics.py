"""
API metrics collector
"""
from collections import defaultdict
from datetime import datetime
import threading
from typing import Any, Dict, List, Optional

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, REGISTRY, generate_latest


def _get_or_create_metric(metric_cls, name: str, documentation: str, *args, **kwargs):
    """
    Reuse existing collectors when tests/imports instantiate the module multiple times, to avoid "Collector already registered" errors.
    """
    existing = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
    if existing is not None:
        return existing
    try:
        return metric_cls(name, documentation, *args, **kwargs)
    except ValueError:
        return REGISTRY._names_to_collectors[name]


class MetricsCollector:
    """
    API metrics collector with two outputs:
    - JSON metrics (legacy endpoint `/metrics`)
    - Prometheus exposition format (`/metrics/prometheus`)
    """

    REQUESTS_TOTAL = _get_or_create_metric(
        Counter,
        "passos_magicos_api_requests_total",
        "Total API requests.",
        labelnames=("method", "endpoint", "status_code"),
    )
    REQUEST_DURATION_SECONDS = _get_or_create_metric(
        Histogram,
        "passos_magicos_api_request_duration_seconds",
        "API request latency in seconds.",
        labelnames=("method", "endpoint", "status_code"),
    )
    REQUESTS_IN_FLIGHT = _get_or_create_metric(
        Gauge,
        "passos_magicos_api_requests_in_flight",
        "Requests currently in flight.",
    )
    PREDICTIONS_TOTAL = _get_or_create_metric(
        Counter,
        "passos_magicos_api_predictions_total",
        "Total predictions by class.",
        labelnames=("prediction_class",),
    )
    PREDICTION_CONFIDENCE = _get_or_create_metric(
        Histogram,
        "passos_magicos_api_prediction_confidence",
        "Prediction confidence distribution.",
        buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0),
    )
    MODEL_LOADED = _get_or_create_metric(
        Gauge,
        "passos_magicos_api_model_loaded",
        "Model loaded status (1 loaded, 0 not loaded).",
    )
    FEATURE_STORE_LOADED = _get_or_create_metric(
        Gauge,
        "passos_magicos_api_feature_store_loaded",
        "Feature Store loaded status (1 loaded, 0 not loaded).",
    )

    def __init__(self):
        self._lock = threading.Lock()
        self._reset_metrics()

    def _reset_metrics(self):
        """Reset in-memory JSON metrics."""
        self.total_requests = 0
        self.total_predictions = 0
        self.predictions_by_class = defaultdict(int)
        self.request_durations: List[float] = []
        self.requests_by_endpoint = defaultdict(int)
        self.requests_by_status = defaultdict(int)
        self.model_loaded = False
        self.feature_store_loaded = False
        self.start_time = datetime.now()

        # Gauges can be reset safely; counters/histograms are monotonic by design.
        self.REQUESTS_IN_FLIGHT.set(0)
        self.MODEL_LOADED.set(0)
        self.FEATURE_STORE_LOADED.set(0)

    def request_started(self) -> None:
        """Increment in-flight request gauge."""
        self.REQUESTS_IN_FLIGHT.inc()

    def request_finished(self) -> None:
        """Decrement in-flight request gauge."""
        try:
            self.REQUESTS_IN_FLIGHT.dec()
        except ValueError:
            # Safety fallback when gauge is already zero.
            self.REQUESTS_IN_FLIGHT.set(0)

    def set_model_loaded(self, loaded: bool) -> None:
        """Expose model loaded status as gauge."""
        with self._lock:
            self.model_loaded = loaded
            self.MODEL_LOADED.set(1 if loaded else 0)

    def set_feature_store_loaded(self, loaded: bool) -> None:
        """Expose feature store loaded status as gauge."""
        with self._lock:
            self.feature_store_loaded = loaded
            self.FEATURE_STORE_LOADED.set(1 if loaded else 0)

    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float
    ) -> None:
        """
        Record one HTTP request.
        """
        method_label = (method or "UNKNOWN").upper()
        endpoint_label = endpoint or "unknown"
        status_code_label = str(status_code)
        safe_duration = max(duration, 0.0)

        with self._lock:
            self.total_requests += 1
            self.requests_by_endpoint[f"{method_label} {endpoint_label}"] += 1
            self.requests_by_status[status_code] += 1
            self.request_durations.append(safe_duration)

            # Keep only the latest 10000 samples in memory for JSON endpoint.
            if len(self.request_durations) > 10000:
                self.request_durations = self.request_durations[-10000:]

        self.REQUESTS_TOTAL.labels(
            method=method_label,
            endpoint=endpoint_label,
            status_code=status_code_label,
        ).inc()
        self.REQUEST_DURATION_SECONDS.labels(
            method=method_label,
            endpoint=endpoint_label,
            status_code=status_code_label,
        ).observe(safe_duration)

    def record_prediction(self, prediction: int, confidence: Optional[float] = None) -> None:
        """
        Record one prediction event.
        """
        with self._lock:
            self.total_predictions += 1
            self.predictions_by_class[prediction] += 1

        self.PREDICTIONS_TOTAL.labels(prediction_class=str(prediction)).inc()
        if confidence is not None:
            self.PREDICTION_CONFIDENCE.observe(max(0.0, min(1.0, float(confidence))))

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return JSON metrics for backward compatibility.
        """
        with self._lock:
            uptime = (datetime.now() - self.start_time).total_seconds()

            durations = self.request_durations
            avg_duration = sum(durations) / len(durations) if durations else 0
            max_duration = max(durations) if durations else 0
            min_duration = min(durations) if durations else 0

            sorted_durations = sorted(durations)
            p95_idx = int(len(sorted_durations) * 0.95)
            p99_idx = int(len(sorted_durations) * 0.99)
            p95_duration = sorted_durations[p95_idx] if sorted_durations else 0
            p99_duration = sorted_durations[p99_idx] if sorted_durations else 0

            total_preds = self.total_predictions
            pred_distribution = {}
            if total_preds > 0:
                pred_distribution = {
                    "class_0_pct": round(self.predictions_by_class[0] / total_preds * 100, 2),
                    "class_1_pct": round(self.predictions_by_class[1] / total_preds * 100, 2),
                }

            return {
                "uptime_seconds": round(uptime, 2),
                "total_requests": self.total_requests,
                "total_predictions": self.total_predictions,
                "requests_per_second": round(self.total_requests / uptime, 4) if uptime > 0 else 0,
                "predictions_by_class": dict(self.predictions_by_class),
                "prediction_distribution": pred_distribution,
                "latency": {
                    "avg_ms": round(avg_duration * 1000, 2),
                    "min_ms": round(min_duration * 1000, 2),
                    "max_ms": round(max_duration * 1000, 2),
                    "p95_ms": round(p95_duration * 1000, 2),
                    "p99_ms": round(p99_duration * 1000, 2),
                },
                "requests_by_endpoint": dict(self.requests_by_endpoint),
                "requests_by_status": dict(self.requests_by_status),
                "model_loaded": self.model_loaded,
                "feature_store_loaded": self.feature_store_loaded,
                "start_time": self.start_time.isoformat(),
            }

    def get_prometheus_metrics(self) -> bytes:
        """Return Prometheus exposition payload."""
        return generate_latest()

    @property
    def prometheus_content_type(self) -> str:
        """Prometheus content type header value."""
        return CONTENT_TYPE_LATEST

    def reset(self) -> None:
        """Reset in-memory JSON metrics and gauges."""
        with self._lock:
            self._reset_metrics()
