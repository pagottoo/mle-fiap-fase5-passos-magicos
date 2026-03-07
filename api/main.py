"""
FastAPI service for the Turning Point prediction model.
"""
import uuid
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from structlog.contextvars import bind_contextvars, clear_contextvars

# Expose classes and functions for tests to patch (backward compatibility)
from src.models import ModelPredictor
from src.feature_store import FeatureStore
from api.dependencies import init_app_state, get_metrics_collector, _create_predictor_from_runtime_config
from api.routes import predictions, monitoring, feature_store as fs_routes, admin
from src.monitoring.logger import setup_logging, get_logger

# Configure logging
setup_logging()
logger = get_logger(component="api")

# Expose globals for backward compatibility with tests
predictor = None
feature_store = None
metrics_collector = get_metrics_collector()

# Create FastAPI application
app = FastAPI(
    title="Passos Magicos - Turning Point Prediction API",
    description="API for predicting the student Turning Point outcome.",
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

@app.on_event("startup")
async def startup_event():
    """Initialize state on startup."""
    global predictor, feature_store
    predictor, feature_store, _ = init_app_state()
    logger.info("api_started")

# Middleware for logging and metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log requests and collect metrics."""
    start_time = time.time()
    trace_id = request.headers.get("x-request-id", "").strip() or uuid.uuid4().hex
    request.state.trace_id = trace_id
    bind_contextvars(trace_id=trace_id)

    metrics_collector = get_metrics_collector()
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
        
        noisy_paths = ["/health", "/metrics", "/metrics/prometheus"]
        if route_path not in noisy_paths:
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

# Include Routers
app.include_router(predictions.router)
app.include_router(monitoring.router)
app.include_router(fs_routes.router)
app.include_router(admin.router)

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Passos Magicos - Turning Point Prediction API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return metadata for the loaded model (Compat)."""
    from fastapi import HTTPException
    # Use the global that tests patch
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    info = predictor.get_model_info()
    return info

@app.get("/features", tags=["Model"])
async def get_features_compat():
    """Return features (Compat)."""
    from fastapi import HTTPException
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return {"features": predictor.get_feature_names()}

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
        host=str(API_CONFIG["host"]),
        port=int(API_CONFIG["port"]),
        reload=bool(API_CONFIG["debug"])
    )
