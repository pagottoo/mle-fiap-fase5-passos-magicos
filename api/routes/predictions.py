import uuid
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, Query
from api.schemas import (
    StudentData, PredictionResponse, ExplanationResponse, 
    FeatureContribution, BatchPredictionRequest
)
from api.dependencies import get_metrics_collector
from src.monitoring.logger import log_prediction, get_logger

router = APIRouter(tags=["Prediction"])
logger = get_logger(component="api_predictions")

@router.post("/predict", response_model=PredictionResponse)
async def predict(student: StudentData):
    """Predict Turning Point for a single student."""
    import api.main as main
    predictor = main.predictor
    metrics_collector = get_metrics_collector()
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    
    try:
        prediction_id = str(uuid.uuid4())
        input_data = student.model_dump(exclude_none=True)
        result = predictor.predict(input_data)
        result["prediction_id"] = prediction_id
        log_prediction(input_data, result)
        metrics_collector.record_prediction(
            prediction=result["prediction"],
            confidence=result.get("confidence"),
        )
        
        return PredictionResponse(
            prediction_id=prediction_id,
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

@router.post("/predict/explain", response_model=ExplanationResponse)
async def explain_prediction(student: StudentData, top_n: int = 5):
    """Predict and explain the outcome for a single student (XAI)."""
    import api.main as main
    predictor = main.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    
    try:
        prediction_id = str(uuid.uuid4())
        input_data = student.model_dump(exclude_none=True)
        result = predictor.predict(input_data)
        result["prediction_id"] = prediction_id
        explanation = predictor.explain(input_data, top_n=top_n)
        
        if "error" in explanation:
            raise HTTPException(status_code=500, detail=explanation["error"])
            
        log_prediction(input_data, result)
            
        return ExplanationResponse(
            prediction_id=prediction_id,
            prediction=result["prediction"],
            label=result["label"],
            probability=result["probability_turning_point"],
            base_value=explanation["base_value"],
            top_contributions=[
                FeatureContribution(**c) for c in explanation["top_contributions"]
            ],
            model_version=result["model_version"],
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("explanation_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict for multiple students."""
    import api.main as main
    predictor = main.predictor
    metrics_collector = get_metrics_collector()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    
    try:
        results = []
        for student in request.students:
            input_data = student.model_dump(exclude_none=True)
            result = predictor.predict(input_data)
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

@router.get("/predict/alunos")
async def predict_batch_by_ids(aluno_ids: str = Query(..., description="Comma-separated student IDs")):
    """Predict for multiple student IDs from Feature Store."""
    import api.main as main
    predictor = main.predictor
    feature_store = main.feature_store
    metrics_collector = get_metrics_collector()
    if predictor is None or feature_store is None:
        raise HTTPException(status_code=503, detail="Modelo ou Feature Store não carregado.")
    
    try:
        try:
            ids = [int(i.strip()) for i in aluno_ids.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="aluno_ids deve ser uma lista de inteiros separados por vírgula.")
            
        features_df = feature_store.get_serving_features("alunos_features", ids)
        if features_df.empty:
            raise HTTPException(status_code=404, detail="Nenhum aluno encontrado para os IDs fornecidos.")
            
        results = []
        for _, row in features_df.iterrows():
            input_data = row.to_dict()
            result = predictor.predict(input_data)
            metrics_collector.record_prediction(prediction=result["prediction"], confidence=result.get("confidence"))
            results.append({**result, "timestamp": datetime.now().isoformat()})
            
        return {"predictions": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("batch_id_prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
