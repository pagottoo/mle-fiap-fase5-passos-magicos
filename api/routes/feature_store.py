import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from api.schemas import PredictionResponse
from api.dependencies import get_metrics_collector
from src.monitoring.logger import log_prediction, get_logger

router = APIRouter(prefix="/features", tags=["Feature Store"])
logger = get_logger(component="api_feature_store")

@router.get("/status")
async def get_feature_store_status():
    """Return Feature Store status."""
    import api.main as main
    feature_store = main.feature_store
    if feature_store is None:
        raise HTTPException(status_code=503, detail="Feature Store não inicializado.")
    return feature_store.get_status()

@router.get("/registry")
async def get_feature_registry():
    """List all registered features in Feature Store."""
    import api.main as main
    feature_store = main.feature_store
    if feature_store is None:
        raise HTTPException(status_code=503, detail="Feature Store não inicializado.")
    features = []
    for name in feature_store.list_features():
        feat = feature_store.get_feature_definition(name)
        if feat:
            features.append({
                "name": feat.name, "dtype": feat.dtype, "description": feat.description,
                "source": feat.source, "transformation": feat.transformation, "tags": feat.tags
            })
    return {"features": features, "count": len(features)}

@router.get("/groups")
async def get_feature_groups():
    """List available feature groups."""
    import api.main as main
    feature_store = main.feature_store
    if feature_store is None:
        raise HTTPException(status_code=503, detail="Feature Store não inicializado.")
    groups = []
    for name in feature_store.list_groups():
        group = feature_store.registry.get_group(name)
        if group:
            groups.append({
                "name": group.name, "description": group.description,
                "features": group.features, "entity": group.entity
            })
    return {"groups": groups, "count": len(groups)}

@router.get("/predict/aluno/{aluno_id}", response_model=PredictionResponse)
async def predict_by_aluno_id(aluno_id: int):
    """Predict for one student using Feature Store features."""
    import api.main as main
    predictor = main.predictor
    feature_store = main.feature_store
    metrics_collector = get_metrics_collector()
    if predictor is None or feature_store is None:
        raise HTTPException(status_code=503, detail="Modelo ou Feature Store não carregado.")
    
    try:
        feature_vector = feature_store.get_feature_vector("alunos_features", aluno_id)
        if not feature_vector:
            raise HTTPException(status_code=404, detail=f"Aluno {aluno_id} não encontrado.")
        
        prediction_id = str(uuid.uuid4())
        result = predictor.predict(feature_vector)
        result["prediction_id"] = prediction_id
        metrics_collector.record_prediction(prediction=result["prediction"], confidence=result.get("confidence"))
        log_prediction({"aluno_id": aluno_id, "source": "feature_store"}, result)
        
        return PredictionResponse(
            prediction_id=prediction_id, prediction=result["prediction"], label=result["label"],
            probability_no_turning_point=result["probability_no_turning_point"],
            probability_turning_point=result["probability_turning_point"],
            confidence=result["confidence"], model_version=result["model_version"],
            timestamp=datetime.now().isoformat()
        )
    except HTTPException: raise
    except Exception as e:
        logger.error("feature_store_prediction_error", aluno_id=aluno_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
