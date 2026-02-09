"""
API FastAPI para o modelo de Ponto de Virada
"""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import structlog

from src.models import ModelPredictor
from src.monitoring.logger import setup_logging, log_prediction
from src.monitoring.metrics import MetricsCollector
from src.feature_store import FeatureStore

# Configurar logging
setup_logging()
logger = structlog.get_logger()

# Inicializar métricas
metrics_collector = MetricsCollector()

# Criar aplicação FastAPI
app = FastAPI(
    title="Passos Mágicos - API de Predição de Ponto de Virada",
    description="""
    API para predição de Ponto de Virada dos alunos da Passos Mágicos.
    
    O **Ponto de Virada** representa o momento transformador na vida do aluno,
    quando ele "vira a chave" e começa uma mudança real em sua trajetória educacional.
    
    ## Endpoints
    
    - **POST /predict**: Realiza predição para um único aluno
    - **POST /predict/batch**: Realiza predição para múltiplos alunos
    - **GET /predict/aluno/{aluno_id}**: Predição usando Feature Store
    - **GET /health**: Verifica status da API
    - **GET /model/info**: Retorna informações do modelo
    - **GET /metrics**: Retorna métricas de uso da API
    - **GET /features/status**: Status do Feature Store
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar modelo e Feature Store na inicialização
predictor = None
feature_store = None


def _create_predictor_from_runtime_config() -> ModelPredictor:
    """
    Cria o predictor com base na estratégia de runtime.

    Estratégias:
    - local (padrão): carrega de arquivo local (.joblib)
    - mlflow: carrega do MLflow Model Registry
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

    # local (padrão) ou fallback
    if model_path is not None:
        return ModelPredictor(model_path=model_path)
    return ModelPredictor()


@app.on_event("startup")
async def startup_event():
    """Carrega o modelo e Feature Store na inicialização da API."""
    global predictor, feature_store
    
    # Carregar modelo
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
    
    # Inicializar Feature Store
    try:
        feature_store = FeatureStore()
        logger.info("feature_store_loaded", status="success")
    except Exception as e:
        logger.error("feature_store_error", error=str(e))
        feature_store = None
    
    logger.info("api_started", model_loaded=predictor is not None, feature_store_loaded=feature_store is not None)


# Schemas Pydantic
class StudentData(BaseModel):
    """Schema para dados de entrada do aluno.
    
    Campos obrigatórios para predição: INDE, IAA, IEG, IPS, IDA, IPP, IPV, IAN, FASE, PEDRA, BOLSISTA
    """
    # Campos obrigatórios para predição
    INDE: float = Field(..., description="Índice de Desenvolvimento Educacional")
    IAA: float = Field(..., description="Índice de Autoavaliação")
    IEG: float = Field(..., description="Índice de Engajamento")
    IPS: float = Field(..., description="Índice Psicossocial")
    IDA: float = Field(..., description="Índice de Desempenho Acadêmico")
    IPP: float = Field(..., description="Índice de Participação")
    IPV: float = Field(..., description="Índice de Propensão à Virada")
    IAN: float = Field(..., description="Índice de Adequação ao Nível")
    FASE: str = Field(..., description="Fase/turma do aluno")
    PEDRA: str = Field(..., description="Classificação do aluno (Quartzo, Ametista, etc)")
    BOLSISTA: str = Field(..., description="Se é bolsista (Sim/Não)")
    
    # Campos opcionais (não usados no modelo atual)
    IDADE_ALUNO: Optional[int] = Field(None, description="Idade do aluno")
    ANOS_PM: Optional[int] = Field(None, description="Anos na Passos Mágicos")
    INSTITUICAO_ENSINO_ALUNO: Optional[str] = Field(None, description="Tipo de instituição de ensino")
    PONTO_VIRADA: Optional[str] = Field(None, description="Se atingiu ponto de virada (não usar como feature)")
    
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
    """Schema para resposta de predição."""
    prediction: int
    label: str
    probability_no_turning_point: float
    probability_turning_point: float
    confidence: float
    model_version: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Schema para requisição de predição em lote."""
    students: List[StudentData]


class HealthResponse(BaseModel):
    """Schema para resposta de health check."""
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Schema para informações do modelo."""
    model_type: str
    version: str
    trained_at: str
    metrics: Dict[str, Any]


# Middleware para logging e métricas
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para logar requisições e coletar métricas."""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Coletar métricas
    metrics_collector.record_request(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        duration=duration
    )
    
    logger.info(
        "request_processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2)
    )
    
    return response


# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz."""
    return {
        "message": "Passos Mágicos - API de Predição de Ponto de Virada",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Verifica o status da API e do modelo."""
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Retorna informações sobre o modelo carregado."""
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
    Realiza predição de Ponto de Virada para um aluno.
    
    Retorna:
    - prediction: 0 (improvável) ou 1 (provável)
    - label: "Ponto de Virada Provável" ou "Ponto de Virada Improvável"
    - probability_*: Probabilidades de cada classe
    - confidence: Confiança da predição
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o treinamento primeiro."
        )
    
    try:
        # Converter para dicionário
        input_data = student.model_dump(exclude_none=True)
        
        # Fazer predição
        result = predictor.predict(input_data)
        
        # Logar predição para monitoramento
        log_prediction(input_data, result)
        
        # Registrar métricas
        metrics_collector.record_prediction(result["prediction"])
        
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
    Realiza predição para múltiplos alunos.
    
    Retorna lista de predições, uma para cada aluno.
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
            
            # Logar predição
            log_prediction(input_data, result)
            metrics_collector.record_prediction(result["prediction"])
            
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
    """Retorna métricas de uso da API."""
    return metrics_collector.get_metrics()


# ==================== Alert Endpoints ====================

@app.get("/alerts/status", tags=["Monitoring"])
async def get_alert_status():
    """
    Retorna status do sistema de alertas.
    
    Mostra:
    - Canais configurados (Slack, Email, Webhook)
    - Contagem de alertas por severidade
    - Último alerta enviado
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
    Retorna histórico de alertas enviados.
    
    Args:
        limit: Número máximo de alertas a retornar (default: 50)
    
    Retorna lista de alertas ordenados do mais recente para o mais antigo.
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
    """Schema para requisição de teste de alerta."""
    channel: Optional[str] = Field(None, description="Canal específico para testar (console, slack, email, webhook)")
    message: Optional[str] = Field("Alerta de teste via API", description="Mensagem de teste")


@app.post("/alerts/test", tags=["Monitoring"])
async def test_alert(request: AlertTestRequest):
    """
    Envia um alerta de teste.
    
    Útil para:
    - Verificar se canais estão configurados corretamente
    - Testar conectividade com Slack, Email, etc.
    - Validar formatação de mensagens
    """
    try:
        from src.monitoring.alerts import get_alert_manager, AlertType, AlertSeverity
        
        alert_manager = get_alert_manager()
        
        # Enviar alerta de teste
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
    """Schema para envio manual de alerta."""
    type: str = Field(..., description="Tipo do alerta: data_drift, prediction_drift, model_performance, api_error, system")
    severity: str = Field("WARNING", description="Severidade: INFO, WARNING, ERROR, CRITICAL")
    title: str = Field(..., description="Título do alerta")
    message: str = Field(..., description="Mensagem detalhada")
    details: Optional[Dict[str, Any]] = Field(None, description="Detalhes adicionais")


@app.post("/admin/reload-model", tags=["Admin"])
async def reload_model(request: Request):
    """
    Recarrega o modelo em runtime sem reiniciar o processo da API.

    Segurança:
    - Se ADMIN_RELOAD_TOKEN estiver definido, exige header X-Admin-Token.
    """
    global predictor

    reload_token = os.getenv("ADMIN_RELOAD_TOKEN", "").strip()
    provided_token = request.headers.get("X-Admin-Token", "").strip()

    if reload_token and provided_token != reload_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        predictor = _create_predictor_from_runtime_config()
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


@app.post("/alerts/send", tags=["Monitoring"])
async def send_manual_alert(request: ManualAlertRequest):
    """
    Envia um alerta manualmente.
    
    Permite enviar alertas customizados via API, útil para:
    - Integração com sistemas externos
    - Alertas de processos de batch
    - Notificações de eventos importantes
    """
    try:
        from src.monitoring.alerts import get_alert_manager, AlertType, AlertSeverity
        
        alert_manager = get_alert_manager()
        
        # Mapear tipo
        type_map = {
            "data_drift": AlertType.DATA_DRIFT,
            "prediction_drift": AlertType.PREDICTION_DRIFT,
            "model_performance": AlertType.MODEL_PERFORMANCE,
            "api_error": AlertType.API_ERROR,
            "system": AlertType.SYSTEM_ERROR,
            "system_error": AlertType.SYSTEM_ERROR,
            "custom": AlertType.CUSTOM
        }
        
        # Mapear severidade
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
    """Retorna lista de features utilizadas pelo modelo."""
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
    Retorna status do Feature Store.
    
    Mostra:
    - Features registradas
    - Grupos de features
    - Datasets no offline store
    - Tabelas no online store
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
    Lista todas as features registradas no Feature Store.
    
    Retorna definições de features incluindo:
    - Nome, tipo, descrição
    - Coluna de origem
    - Transformação aplicada
    - Tags para filtragem
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
    """Lista grupos de features disponíveis."""
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
    Faz predição para um aluno usando features do Feature Store.
    
    Este endpoint demonstra a integração Feature Store + Modelo:
    1. Busca features do aluno no Online Store (baixa latência)
    2. Aplica o modelo de predição
    3. Retorna resultado
    
    **Vantagem**: Não precisa enviar todas as features - elas já estão armazenadas!
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
        # 1. Buscar features do Online Store
        feature_vector = feature_store.get_feature_vector("alunos_features", aluno_id)
        
        if not feature_vector:
            raise HTTPException(
                status_code=404,
                detail=f"Aluno {aluno_id} não encontrado no Feature Store."
            )
        
        # 2. Fazer predição
        result = predictor.predict(feature_vector)
        
        # 3. Logar e registrar métricas (apenas predição - requisição já contabilizada no middleware)
        metrics_collector.record_prediction(result["prediction"])
        
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
    Faz predição em lote para múltiplos alunos usando Feature Store.
    
    **Parâmetros:**
    - aluno_ids: IDs separados por vírgula (ex: "1,2,3,4,5")
    
    Este é o caso de uso ideal do Feature Store:
    - Busca features de múltiplos alunos com uma única query
    - Evita latência de múltiplas requisições
    """
    if predictor is None or feature_store is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo ou Feature Store não carregado."
        )
    
    try:
        # Parse IDs
        ids = [int(id.strip()) for id in aluno_ids.split(",")]
        
        # Buscar features em batch
        df = feature_store.get_serving_features("alunos_features", ids)
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="Nenhum aluno encontrado no Feature Store."
            )
        
        # Fazer predições
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


# Handler de exceções
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handler global de exceções."""
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
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
