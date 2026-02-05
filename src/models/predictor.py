"""
Módulo de predição do modelo
"""
import os
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import structlog

from ..config import MODELS_DIR, MODEL_CONFIG

logger = structlog.get_logger()

# Flag para habilitar MLflow
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"

try:
    if MLFLOW_ENABLED:
        import mlflow
        from ..mlflow_tracking import ModelRegistry
        MLFLOW_AVAILABLE = True
    else:
        MLFLOW_AVAILABLE = False
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow_not_available_predictor", message="MLflow não instalado")


class ModelPredictor:
    """
    Classe responsável por fazer predições com o modelo treinado.
    
    Suporta carregamento de:
    - Arquivo local (.joblib)
    - MLflow Model Registry (models:/name/stage)
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        mlflow_model_name: Optional[str] = None,
        mlflow_stage: Optional[str] = "Production"
    ):
        """
        Inicializa o preditor.
        
        Args:
            model_path: Caminho para o arquivo do modelo. Se None, tenta MLflow ou carrega o mais recente.
            mlflow_model_name: Nome do modelo no MLflow Registry (prioridade sobre model_path)
            mlflow_stage: Stage do modelo no MLflow (Production, Staging, etc.)
        """
        self.model_path = model_path
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_stage = mlflow_stage
        self.artifacts = None
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.loaded_from = None  # "local" ou "mlflow"
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Carrega o modelo do MLflow ou arquivo local."""
        
        # Tentar carregar do MLflow primeiro
        if self.mlflow_model_name and MLFLOW_AVAILABLE:
            try:
                self._load_from_mlflow()
                return
            except Exception as e:
                logger.warning(
                    "mlflow_load_failed",
                    model_name=self.mlflow_model_name,
                    error=str(e),
                    message="Tentando carregar do arquivo local"
                )
        
        # Fallback para arquivo local
        self._load_from_file()
    
    def _load_from_mlflow(self) -> None:
        """Carrega modelo do MLflow Model Registry."""
        model_uri = f"models:/{self.mlflow_model_name}/{self.mlflow_stage}"
        logger.info("loading_model_from_mlflow", model_uri=model_uri)
        
        # Carregar modelo sklearn do MLflow
        self.model = mlflow.sklearn.load_model(model_uri)
        
        # Para MLflow, precisamos carregar preprocessor e feature_engineer separadamente
        # Eles são salvos como artefatos no run
        registry = ModelRegistry()
        versions = registry.get_latest_versions(self.mlflow_model_name, stages=[self.mlflow_stage])
        
        if versions:
            version = versions[0]
            run_id = version.run_id
            
            # Carregar artefatos adicionais se existirem no run
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            
            self.artifacts = {
                "model": self.model,
                "model_type": run.data.params.get("model_type", "unknown"),
                "version": f"mlflow-v{version.version}",
                "trained_at": "mlflow",
                "metrics": run.data.metrics,
                "preprocessor": None,  # Será carregado separadamente se necessário
                "feature_engineer": None
            }
            
            logger.info(
                "model_loaded_from_mlflow",
                model_name=self.mlflow_model_name,
                version=version.version,
                stage=self.mlflow_stage
            )
            
            self.loaded_from = "mlflow"
        else:
            raise ValueError(f"Nenhuma versão encontrada para {self.mlflow_model_name} em {self.mlflow_stage}")
    
    def _load_from_file(self) -> None:
        """Carrega o modelo e artefatos do arquivo local."""
        if self.model_path is None:
            self.model_path = MODELS_DIR / f"{MODEL_CONFIG['model_name']}_latest.joblib"
        
        logger.info("loading_model_from_file", path=str(self.model_path))
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        
        self.artifacts = joblib.load(self.model_path)
        self.model = self.artifacts["model"]
        self.preprocessor = self.artifacts["preprocessor"]
        self.feature_engineer = self.artifacts["feature_engineer"]
        self.loaded_from = "local"
        
        logger.info(
            "model_loaded_from_file",
            model_type=self.artifacts["model_type"],
            version=self.artifacts["version"],
            trained_at=self.artifacts["trained_at"]
        )
    
    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Pré-processa dados de entrada para predição.
        
        Args:
            data: Dicionário com dados do aluno
            
        Returns:
            DataFrame processado
        """
        # Converter para DataFrame
        df = pd.DataFrame([data])
        
        # Aplicar transformações (se disponíveis)
        if self.preprocessor and self.feature_engineer:
            df = self.preprocessor.handle_missing_values(df)
            df = self.feature_engineer.transform(df)
        
        return df
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Faz predição para um único registro.
        
        Args:
            data: Dicionário com dados do aluno
            
        Returns:
            Dicionário com predição e probabilidades
        """
        logger.info("making_prediction", input_data=data)
        
        # Se carregado do MLflow, pode receber features já processadas
        if self.loaded_from == "mlflow" or (not self.preprocessor or not self.feature_engineer):
            # Converter para array numpy diretamente
            df = pd.DataFrame([data])
            X = df.values
        else:
            df = self.preprocess_input(data)
            X, _ = self.feature_engineer.get_feature_matrix(df)
        
        prediction = int(self.model.predict(X)[0])
        probabilities = self.model.predict_proba(X)[0].tolist()
        
        result = {
            "prediction": prediction,
            "label": "Ponto de Virada Provável" if prediction == 1 else "Ponto de Virada Improvável",
            "probability_no_turning_point": round(probabilities[0], 4),
            "probability_turning_point": round(probabilities[1], 4),
            "confidence": round(max(probabilities), 4),
            "model_version": self.artifacts["version"],
            "loaded_from": self.loaded_from
        }
        
        logger.info("prediction_made", result=result)
        
        return result
    
    def predict_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Faz predição para múltiplos registros.
        
        Args:
            data_list: Lista de dicionários com dados dos alunos
            
        Returns:
            Lista de dicionários com predições
        """
        logger.info("making_batch_prediction", batch_size=len(data_list))
        
        results = []
        for data in data_list:
            result = self.predict(data)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo carregado.
        
        Returns:
            Dicionário com informações do modelo
        """
        return {
            "model_type": self.artifacts["model_type"],
            "version": self.artifacts["version"],
            "trained_at": self.artifacts["trained_at"],
            "metrics": self.artifacts["metrics"],
            "feature_importance": self.artifacts.get("feature_importance", {}),
            "loaded_from": self.loaded_from,
            "mlflow_model_name": self.mlflow_model_name,
            "mlflow_stage": self.mlflow_stage if self.loaded_from == "mlflow" else None
        }
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna nomes das features utilizadas pelo modelo.
        
        Returns:
            Lista com nomes das features
        """
        if self.feature_engineer:
            return self.feature_engineer.feature_names
        return []
    
    @classmethod
    def from_mlflow(
        cls,
        model_name: str = "passos-magicos-ponto-virada",
        stage: str = "Production"
    ) -> "ModelPredictor":
        """
        Factory method para criar preditor do MLflow.
        
        Args:
            model_name: Nome do modelo no registry
            stage: Stage (Production, Staging)
            
        Returns:
            ModelPredictor configurado
        """
        return cls(mlflow_model_name=model_name, mlflow_stage=stage)
