import os
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from api.dependencies import init_app_state, get_metrics_collector
from src.monitoring.logger import get_logger

router = APIRouter(prefix="/admin", tags=["Admin"])
logger = get_logger(component="api_admin")

@router.post("/reload-model")
async def reload_model(request: Request):
    """Reload model at runtime without restarting the API process."""
    reload_token = os.getenv("ADMIN_RELOAD_TOKEN", "").strip()
    provided_token = request.headers.get("X-Admin-Token", "").strip()

    if reload_token and provided_token != reload_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        predictor, _, metrics_collector = init_app_state()
        info = predictor.get_model_info()
        return {
            "status": "reloaded",
            "model_loaded": predictor is not None,
            "loaded_from": info.get("loaded_from"),
            "model_version": info.get("version"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("model_reload_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Falha ao recarregar modelo: {e}")
