from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from api.schemas import HealthResponse, FeedbackRequest
from api.dependencies import get_metrics_collector
from src.monitoring.logger import get_recent_predictions, log_feedback, get_logger

router = APIRouter(tags=["Monitoring"])
logger = get_logger(component="api_monitoring")

class AlertTestRequest(BaseModel):
    message: str = "Test alert from API"

class ManualAlertRequest(BaseModel):
    type: str
    severity: str
    title: str
    message: str
    metadata: Optional[Dict[str, Any]] = None

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health state."""
    import api.main as main
    predictor = main.predictor
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        timestamp=datetime.now().isoformat()
    )

@router.post("/predict/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit ground truth for a past prediction."""
    metrics_collector = get_metrics_collector()
    try:
        recent_preds = get_recent_predictions(n=500)
        original_pred = next(
            (p for p in recent_preds if p["output"].get("prediction_id") == feedback.prediction_id), 
            None
        )
        
        is_correct = None
        if original_pred:
            predicted_class = int(original_pred["output"]["prediction"])
            is_correct = (predicted_class == feedback.actual_outcome)
            metrics_collector.record_feedback(is_correct=is_correct)
            
        log_feedback({
            "prediction_id": feedback.prediction_id,
            "actual_outcome": feedback.actual_outcome,
            "comment": feedback.comment,
            "was_correct": is_correct
        })
        
        return {"status": "success", "message": "Feedback registrado", "was_correct": is_correct}
    except Exception as e:
        logger.error("feedback_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics():
    """Return API usage metrics in JSON format."""
    return get_metrics_collector().get_metrics()

@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Return metrics in Prometheus exposition format."""
    metrics_collector = get_metrics_collector()
    return Response(
        content=metrics_collector.get_prometheus_metrics(),
        media_type=metrics_collector.prometheus_content_type,
    )

@router.get("/alerts/status")
async def get_alert_status():
    """Return alerting system status."""
    try:
        from src.monitoring.alerts import get_alert_manager
        alert_manager = get_alert_manager()
        status = alert_manager.get_status()
        return {"status": "configured" if status["configured_channels"] > 0 else "not_configured", **status}
    except Exception as e:
        logger.error("alert_status_error", error=str(e))
        return {"status": "error", "error": str(e), "configured_channels": 0}

@router.get("/alerts/history")
async def get_alert_history(limit: int = 50):
    """Return alert history."""
    try:
        from src.monitoring.alerts import get_alert_manager
        alert_manager = get_alert_manager()
        history = alert_manager.get_history(limit=limit)
        return {"alerts": history, "count": len(history), "limit": limit}
    except Exception as e:
        logger.error("alert_history_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/test")
async def test_alerts(request: AlertTestRequest):
    """Send a test alert to all configured channels."""
    try:
        from src.monitoring.alerts import send_alert
        results = send_alert(
            alert_type="api_test",
            severity="INFO",
            title="API Test Alert",
            message=request.message,
            metadata={"source": "api_test_endpoint"}
        )
        return {"status": "success", "success": results}
    except Exception as e:
        logger.error("alert_test_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/send")
async def send_manual_alert(request: ManualAlertRequest):
    """Manually send an alert."""
    try:
        from src.monitoring.alerts import send_alert
        results = send_alert(
            alert_type=request.type,
            severity=request.severity,
            title=request.title,
            message=request.message,
            metadata=request.metadata
        )
        return {"status": "success", "success": results}
    except Exception as e:
        logger.error("manual_alert_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
