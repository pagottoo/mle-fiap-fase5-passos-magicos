"""
FastAPI service for the Turning Point prediction model.
"""
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
from structlog.contextvars import bind_contextvars, clear_contextvars

from src.models import ModelPredictor
from src.monitoring.logger import setup_logging, log_prediction, get_logger
from src.monitoring.metrics import MetricsCollector
from src.feature_store import FeatureStore

# Configure logging
setup_logging()
logger = get_logger(component="api")

# Initialize metrics
metrics_collector = MetricsCollector()

# Create FastAPI application
app = FastAPI(
    title="Passos Magicos - Turning Point Prediction API",
    description="""
    API for predicting the student Turning Point outcome.

    The **Turning Point** represents a key transformation moment in a student's journey.

    ## Endpoints

    - **POST /predict**: Predict for one student
    - **POST /predict/batch**: Predict for multiple students
    - **GET /predict/aluno/{aluno_id}**: Predict using Feature Store
    - **GET /health**: Check API health
    - **GET /model/info**: Get model information
    - **GET /metrics**: Get JSON API metrics
    - **GET /metrics/prometheus**: Get Prometheus metrics
    - **GET /features/status**: Get Feature Store status
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and Feature Store on startup
predictor = None
feature_store = None


def _create_predictor_from_runtime_config() -> ModelPredictor:
    """
    Build the predictor using runtime strategy.

    Strategies:
    - local (default): load from local file (.joblib)
    - mlflow: load from MLflow Model Registry
    """
    model_source = os.getenv("MODEL_SOURCE", "local").strip().lower()
    model_path_env = os.getenv("MODEL_PATH", "").strip()
    model_path = Path(model_path_env) if model_path_env else None

    mlflow_model_name = os.getenv("MLFLOW_MODEL_NAME", "passos-magicos-ponto-virada").strip()
    mlflow_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production").strip() or "Production"
    fallback_local = os.getenv("MODEL_FALLBACK_LOCAL", "true").strip().lower() == "true"

    if model_source == "mlflow":
        try:
            logger.info(
                "loading_predictor_mlflow",
                model_name=mlflow_model_name,
                stage=mlflow_stage
            )
            return ModelPredictor.from_mlflow(
                model_name=mlflow_model_name,
                stage=mlflow_stage
            )
        except Exception as e:
            logger.warning(
                "mlflow_predictor_load_failed",
                error=str(e),
                fallback_local=fallback_local
            )
            if not fallback_local:
                raise

    # local (default) or fallback
    if model_path is not None:
        return ModelPredictor(model_path=model_path)
    return ModelPredictor()


@app.on_event("startup")
async def startup_event():
    """Load model and Feature Store during API startup."""
    global predictor, feature_store

    # Load model
    try:
        predictor = _create_predictor_from_runtime_config()
        logger.info(
            "model_loaded",
            status="success",
            loaded_from=getattr(predictor, "loaded_from", "unknown")
        )
    except FileNotFoundError as e:
        logger.error("model_not_found", error=str(e))
        predictor = None
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
    
    logger.info("api_started", model_loaded=predictor is not None, feature_store_loaded=feature_store is not None)


# Pydantic schemas
class StudentData(BaseModel):
    """Schema for student input payload.

    Required fields for prediction:
    INDE, IAA, IEG, IPS, IDA, IPP, IPV, IAN, FASE, PEDRA, BOLSISTA.
    """
    # Required features for inference
    INDE: float = Field(..., description="Educational Development Index")
    IAA: float = Field(..., description="Self-Assessment Index")
    IEG: float = Field(..., description="Engagement Index")
    IPS: float = Field(..., description="Psychosocial Index")
    IDA: float = Field(..., description="Academic Performance Index")
    IPP: float = Field(..., description="Participation Index")
    IPV: float = Field(..., description="Turning Point Propensity Index")
    IAN: float = Field(..., description="Level Adequacy Index")
    FASE: str = Field(..., description="Student stage/class")
    PEDRA: str = Field(..., description="Student cluster (Quartzo, Ametista, etc.)")
    BOLSISTA: str = Field(..., description="Scholarship status (Yes/No)")

    # Optional fields (not used by the current model)
    IDADE_ALUNO: Optional[int] = Field(None, description="Student age")
    ANOS_PM: Optional[int] = Field(None, description="Years in Passos Magicos")
    INSTITUICAO_ENSINO_ALUNO: Optional[str] = Field(None, description="School institution type")
    PONTO_VIRADA: Optional[str] = Field(None, description="Observed turning point label (not used as feature)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "INDE": 7.5,
                "IAA": 8.0,
                "IEG": 7.0,
                "IPS": 6.5,
                "IDA": 7.2,
                "IPP": 6.8,
                "IPV": 7.5,
                "IAN": 5.0,
                "IDADE_ALUNO": 12,
                "ANOS_PM": 2,
                "INSTITUICAO_ENSINO_ALUNO": "Escola Pública",
                "FASE": "3",
                "PEDRA": "Ametista",
                "PONTO_VIRADA": "Não",
                "BOLSISTA": "Não"
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    prediction: int
    label: str
    probability_no_turning_point: float
    probability_turning_point: float
    confidence: float
    model_version: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""
    students: List[StudentData]


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Schema for model metadata."""
    model_type: str
    version: str
    trained_at: str
    metrics: Dict[str, Any]


# Middleware for logging and metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log requests and collect metrics."""
    start_time = time.time()
    trace_id = request.headers.get("x-request-id", "").strip() or uuid.uuid4().hex
    request.state.trace_id = trace_id
    bind_contextvars(trace_id=trace_id)

    metrics_collector.request_started()
    try:
        response = await call_next(request)
        duration = time.time() - start_time

        route = request.scope.get("route")
        route_path = request.url.path
        if route is not None and hasattr(route, "path"):
            route_path = route.path

        metrics_collector.record_request(
            endpoint=route_path,
            method=request.method,
            status_code=response.status_code,
            duration=duration,
        )

        response.headers["x-request-id"] = trace_id
        logger.info(
            "request_processed",
            method=request.method,
            path=route_path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )
        return response
    finally:
        metrics_collector.request_finished()
        clear_contextvars()


# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Passos Magicos - Turning Point Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and model health state."""
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return metadata for the loaded model."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o treinamento primeiro."
        )
    
    info = predictor.get_model_info()
    return ModelInfoResponse(
        model_type=info["model_type"],
        version=info["version"],
        trained_at=info["trained_at"],
        metrics=info["metrics"]
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(student: StudentData):
    """
    Predict Turning Point for a single student.

    Returns:
    - prediction: 0 (unlikely) or 1 (likely)
    - label: class label from the model
    - probability_*: class probabilities
    - confidence: model confidence
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o treinamento primeiro."
        )
    
    try:
        # Convert payload to dictionary
        input_data = student.model_dump(exclude_none=True)

        # Run prediction
        result = predictor.predict(input_data)

        # Log prediction for monitoring
        log_prediction(input_data, result)

        # Record metrics
        metrics_collector.record_prediction(
            prediction=result["prediction"],
            confidence=result.get("confidence"),
        )
        
        return PredictionResponse(
            prediction=result["prediction"],
            label=result["label"],
            probability_no_turning_point=result["probability_no_turning_point"],
            probability_turning_point=result["probability_turning_point"],
            confidence=result["confidence"],
            model_version=result["model_version"],
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error("prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict for multiple students.

    Returns one prediction result per student.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o treinamento primeiro."
        )
    
    try:
        results = []
        for student in request.students:
            input_data = student.model_dump(exclude_none=True)
            result = predictor.predict(input_data)
            
            # Log prediction
            log_prediction(input_data, result)
            metrics_collector.record_prediction(
                prediction=result["prediction"],
                confidence=result.get("confidence"),
            )
            
            results.append({
                **result,
                "timestamp": datetime.now().isoformat()
            })
        
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        logger.error("batch_prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Return API usage metrics in JSON format."""
    return metrics_collector.get_metrics()


@app.get("/metrics/prometheus", tags=["Monitoring"])
async def get_prometheus_metrics():
    """Return metrics in Prometheus exposition format."""
    return Response(
        content=metrics_collector.get_prometheus_metrics(),
        media_type=metrics_collector.prometheus_content_type,
    )


# ==================== Alert Endpoints ====================

@app.get("/alerts/status", tags=["Monitoring"])
async def get_alert_status():
    """
    Return alerting system status.

    Includes:
    - configured channels (Slack/Webhook/Console)
    - alert count by severity
    - latest alerts
    """
    try:
        from src.monitoring.alerts import get_alert_manager
        
        alert_manager = get_alert_manager()
        status = alert_manager.get_status()
        
        return {
            "status": "configured" if status["configured_channels"] > 0 else "not_configured",
            **status
        }
    except Exception as e:
        logger.error("alert_status_error", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "configured_channels": 0
        }


@app.get("/alerts/history", tags=["Monitoring"])
async def get_alert_history(limit: int = 50):
    """
    Return alert history.

    Args:
        limit: Maximum number of alerts to return (default: 50)

    Returns alerts sorted from newest to oldest.
    """
    try:
        from src.monitoring.alerts import get_alert_manager
        
        alert_manager = get_alert_manager()
        history = alert_manager.get_history(limit=limit)
        
        return {
            "alerts": history,
            "count": len(history),
            "limit": limit
        }
    except Exception as e:
        logger.error("alert_history_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class AlertTestRequest(BaseModel):
    """Schema for alert test request."""
    channel: Optional[str] = Field(None, description="Specific channel to test (console, slack, webhook)")
    message: Optional[str] = Field("API test alert", description="Test message")


@app.post("/alerts/test", tags=["Monitoring"])
async def test_alert(request: AlertTestRequest):
    """
    Send a test alert.

    Useful to:
    - verify channel configuration
    - validate Slack/Webhook connectivity
    - validate message formatting
    """
    try:
        from src.monitoring.alerts import get_alert_manager, AlertType, AlertSeverity
        
        alert_manager = get_alert_manager()
        
        # Send test alert
        success = alert_manager.send_alert(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Teste de Alertas",
            message=request.message or "Alerta de teste via API",
            metadata={
                "source": "api_test",
                "channel_requested": request.channel,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": success,
            "message": "Alerta de teste enviado" if success else "Falha ao enviar alerta",
            "channels_notified": alert_manager.get_status()["channels"]
        }
    except Exception as e:
        logger.error("alert_test_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class ManualAlertRequest(BaseModel):
    """Schema for manual alert payload."""
    type: str = Field(..., description="Alert type: data_drift, prediction_drift, model_performance, api_error, system")
    severity: str = Field("WARNING", description="Severity: INFO, WARNING, ERROR, CRITICAL")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


@app.post("/admin/reload-model", tags=["Admin"])
async def reload_model(request: Request):
    """
    Reload model at runtime without restarting the API process.

    Security:
    - If `ADMIN_RELOAD_TOKEN` is set, requires `X-Admin-Token` header.
    """
    global predictor

    reload_token = os.getenv("ADMIN_RELOAD_TOKEN", "").strip()
    provided_token = request.headers.get("X-Admin-Token", "").strip()

    if reload_token and provided_token != reload_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        predictor = _create_predictor_from_runtime_config()
        info = predictor.get_model_info()
        metrics_collector.set_model_loaded(True)
        return {
            "status": "reloaded",
            "model_loaded": predictor is not None,
            "loaded_from": info.get("loaded_from"),
            "model_version": info.get("version"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        metrics_collector.set_model_loaded(False)
        logger.error("model_reload_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Falha ao recarregar modelo: {e}")


@app.post("/alerts/send", tags=["Monitoring"])
async def send_manual_alert(request: ManualAlertRequest):
    """
    Send a manual alert.

    Useful for:
    - integrations with external systems
    - batch process notifications
    - custom operational events
    """
    try:
        from src.monitoring.alerts import get_alert_manager, AlertType, AlertSeverity
        
        alert_manager = get_alert_manager()
        
        # Map alert type
        type_map = {
            "data_drift": AlertType.DATA_DRIFT,
            "prediction_drift": AlertType.PREDICTION_DRIFT,
            "model_performance": AlertType.MODEL_PERFORMANCE,
            "api_error": AlertType.API_ERROR,
            "system": AlertType.SYSTEM_ERROR,
            "system_error": AlertType.SYSTEM_ERROR,
            "custom": AlertType.CUSTOM
        }
        
        # Map severity
        severity_map = {
            "INFO": AlertSeverity.INFO,
            "WARNING": AlertSeverity.WARNING,
            "ERROR": AlertSeverity.ERROR,
            "CRITICAL": AlertSeverity.CRITICAL
        }
        
        alert_type = type_map.get(request.type.lower(), AlertType.CUSTOM)
        severity = severity_map.get(request.severity.upper(), AlertSeverity.WARNING)
        
        success = alert_manager.send_alert(
            alert_type=alert_type,
            severity=severity,
            title=request.title,
            message=request.message,
            metadata=request.details or {}
        )
        
        return {
            "success": success,
            "alert_type": request.type,
            "severity": request.severity,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("manual_alert_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features", tags=["Model"])
async def get_features():
    """Return the list of features used by the model."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o treinamento primeiro."
        )
    
    return {"features": predictor.get_feature_names()}


# ==================== Feature Store Endpoints ====================

@app.get("/features/status", tags=["Feature Store"])
async def get_feature_store_status():
    """
    Return Feature Store status.

    Includes:
    - registered features
    - feature groups
    - offline store datasets
    - online store tables
    """
    if feature_store is None:
        raise HTTPException(
            status_code=503,
            detail="Feature Store não inicializado."
        )
    
    return feature_store.get_status()


@app.get("/features/registry", tags=["Feature Store"])
async def get_feature_registry():
    """
    List all registered features in Feature Store.

    Includes:
    - name, type and description
    - source column
    - transformation metadata
    - tags
    """
    if feature_store is None:
        raise HTTPException(
            status_code=503,
            detail="Feature Store não inicializado."
        )
    
    features = []
    for name in feature_store.list_features():
        feat = feature_store.get_feature_definition(name)
        if feat:
            features.append({
                "name": feat.name,
                "dtype": feat.dtype,
                "description": feat.description,
                "source": feat.source,
                "transformation": feat.transformation,
                "tags": feat.tags
            })
    
    return {"features": features, "count": len(features)}


@app.get("/features/groups", tags=["Feature Store"])
async def get_feature_groups():
    """List available feature groups."""
    if feature_store is None:
        raise HTTPException(
            status_code=503,
            detail="Feature Store não inicializado."
        )
    
    groups = []
    for name in feature_store.list_groups():
        group = feature_store.registry.get_group(name)
        if group:
            groups.append({
                "name": group.name,
                "description": group.description,
                "features": group.features,
                "entity": group.entity
            })
    
    return {"groups": groups, "count": len(groups)}


@app.get("/predict/aluno/{aluno_id}", response_model=PredictionResponse, tags=["Feature Store"])
async def predict_by_aluno_id(aluno_id: int):
    """
    Predict for one student using Feature Store features.

    This endpoint demonstrates Feature Store + Model integration:
    1. fetches features from Online Store (low latency)
    2. runs model inference
    3. returns prediction result
    """
    start_time = time.time()
    
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o treinamento primeiro."
        )
    
    if feature_store is None:
        raise HTTPException(
            status_code=503,
            detail="Feature Store não inicializado."
        )
    
    try:
        # 1. Fetch features from Online Store
        feature_vector = feature_store.get_feature_vector("alunos_features", aluno_id)
        
        if not feature_vector:
            raise HTTPException(
                status_code=404,
                detail=f"Aluno {aluno_id} não encontrado no Feature Store."
            )
        
        # 2. Run prediction
        result = predictor.predict(feature_vector)

        # 3. Log and record metrics (request metric already handled by middleware)
        metrics_collector.record_prediction(
            prediction=result["prediction"],
            confidence=result.get("confidence"),
        )
        
        log_prediction({"aluno_id": aluno_id, "source": "feature_store"}, result)
        
        return PredictionResponse(
            prediction=result["prediction"],
            label=result["label"],
            probability_no_turning_point=result["probability_no_turning_point"],
            probability_turning_point=result["probability_turning_point"],
            confidence=result["confidence"],
            model_version=result["model_version"],
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("feature_store_prediction_error", aluno_id=aluno_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/alunos", tags=["Feature Store"])
async def predict_batch_by_ids(aluno_ids: str):
    """
    Run batch prediction for multiple students using Feature Store.

    Parameters:
    - `aluno_ids`: comma-separated IDs (e.g. `"1,2,3,4,5"`)

    Ideal Feature Store usage:
    - fetches many students with one query
    - avoids multiple request round-trips
    """
    if predictor is None or feature_store is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo ou Feature Store não carregado."
        )
    
    try:
        # Parse IDs
        ids = [int(id.strip()) for id in aluno_ids.split(",")]

        # Fetch features in batch
        df = feature_store.get_serving_features("alunos_features", ids)
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="Nenhum aluno encontrado no Feature Store."
            )
        
        # Run predictions
        results = []
        for _, row in df.iterrows():
            feature_vector = row.to_dict()
            result = predictor.predict(feature_vector)
            results.append({
                "aluno_id": feature_vector.get("aluno_id"),
                **result,
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "predictions": results,
            "count": len(results),
            "source": "feature_store"
        }
    
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="IDs inválidos. Use formato: 1,2,3,4"
        )
    except Exception as e:
        logger.error("batch_feature_store_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    trace_id = getattr(request.state, "trace_id", None)
    logger.error("unhandled_exception", error=str(exc), path=request.url.path, trace_id=trace_id)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc), "trace_id": trace_id}
    )


if __name__ == "__main__":
    import uvicorn
    from src.config import API_CONFIG
    
    uvicorn.run(
        "api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["debug"]
    )
