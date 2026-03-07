"""
Model prediction module.
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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
    logger.warning("shap_not_available_predictor", message="SHAP is not installed; explain endpoint will be limited")

# Feature flag for MLflow support
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
    logger.warning("mlflow_not_available_predictor", message="MLflow is not installed")


class ModelPredictor:
    """
    Prediction interface for trained models.

    Supports loading from:
    - Local file (.joblib)
    - MLflow Model Registry (models:/name/stage)
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        mlflow_model_name: Optional[str] = None,
        mlflow_stage: Optional[str] = "Production"
    ):
        """
        Initialize predictor.

        Args:
            model_path: Local model path. If None, fallback to MLflow/latest local.
            mlflow_model_name: Model name in MLflow Registry (priority over model_path).
            mlflow_stage: MLflow stage/alias (Production, Staging, etc.).
        """
        self.model_path = model_path
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_stage = mlflow_stage
        self.artifacts = None
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.explainer = None
        self.loaded_from = None  # "local" or "mlflow"
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model from MLflow or local file."""

        # Try MLflow first
        if self.mlflow_model_name and MLFLOW_AVAILABLE:
            try:
                self._load_from_mlflow()
                return
            except Exception as e:
                logger.warning(
                    "mlflow_load_failed",
                    model_name=self.mlflow_model_name,
                    error=str(e),
                    message="Trying local file fallback"
                )

        # Local file fallback
        self._load_from_file()
        
        # Initialize explainer if model is loaded
        if self.model is not None and SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("shap_explainer_initialized")
            except Exception as e:
                logger.warning("shap_explainer_init_failed", error=str(e))
    
    def _load_from_mlflow(self) -> None:
        """Load model from MLflow Model Registry."""
        stage_ref = (self.mlflow_stage or "").strip()
        if stage_ref.lower() == "production":
            model_uri = f"models:/{self.mlflow_model_name}@production"
        elif stage_ref.lower() == "staging":
            model_uri = f"models:/{self.mlflow_model_name}@staging"
        else:
            model_uri = f"models:/{self.mlflow_model_name}/{self.mlflow_stage}"
        logger.info("loading_model_from_mlflow", model_uri=model_uri)
        
        # Load sklearn model artifact from MLflow
        self.model = mlflow.sklearn.load_model(model_uri)

        # For MLflow, preprocessor/feature_engineer may need separate loading
        registry = ModelRegistry()
        versions = registry.get_latest_versions(self.mlflow_model_name, stages=[self.mlflow_stage])
        
        if versions:
            version = versions[0]
            run_id = version.run_id
            
            # Load additional run artifacts when available
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            
            self.artifacts = {
                "model": self.model,
                "model_type": run.data.params.get("model_type", "unknown"),
                "version": f"mlflow-v{version.version}",
                "trained_at": "mlflow",
                "metrics": run.data.metrics,
                "preprocessor": None,  # Can be loaded separately if needed
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
        """Load model and artifacts from local file."""
        if self.model_path is None:
            self.model_path = MODELS_DIR / f"{MODEL_CONFIG['model_name']}_latest.joblib"
        
        logger.info("loading_model_from_file", path=str(self.model_path))
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
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
        Preprocess input payload before prediction.

        Args:
            data: Student input dictionary.

        Returns:
            Processed DataFrame.
        """
        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Add default values for new multi-year features if they are missing
        # This ensures compatibility with single-year or manual inputs (fixes tests)
        evolution_features = [
            "YEAR_REF", "INDE_delta", "INDE_velocity", 
            "IPV_delta", "IPV_velocity", "IAA_delta", "IAA_velocity",
            "IEG_delta", "IEG_velocity", "IPS_delta", "IDA_delta", 
            "IPP_delta", "IAN_delta", "IDADE_ALUNO", "ANOS_PM"
        ]
        
        for feat in evolution_features:
            if feat not in df.columns:
                if feat == "YEAR_REF":
                    df[feat] = 2022  # Default reference year
                else:
                    df[feat] = 0.0  # Default for deltas and missing numeric

        # Apply fitted transforms when available
        if self.preprocessor:
            df = self.preprocessor.handle_missing_values(df)
            
        if self.feature_engineer:
            df = self.feature_engineer.transform(df)
        
        return df
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict for a single record.

        Args:
            data: Student input dictionary.

        Returns:
            Dictionary with prediction and probabilities.
        """
        logger.info("making_prediction", input_data=data)
        
        # Preprocess input (adds missing features and applies transforms)
        df = self.preprocess_input(data)
        
        # MLflow-loaded models may already receive processed features
        if self.loaded_from == "mlflow" or (not self.feature_engineer):
            X = df.values
        else:
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
        Predict for multiple records.

        Args:
            data_list: List of student dictionaries.

        Returns:
            List of prediction dictionaries.
        """
        logger.info("making_batch_prediction", batch_size=len(data_list))
        
        results = []
        for data in data_list:
            result = self.predict(data)
            results.append(result)
        
        return results
    
    def explain(self, data: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
        """
        Explain prediction using SHAP.

        Args:
            data: Input data.
            top_n: Number of top features to return.

        Returns:
            Dictionary with top feature contributions.
        """
        if self.explainer is None:
            return {"error": "SHAP explainer not available for this model"}

        # Preprocess input
        df = self.preprocess_input(data)
        if self.loaded_from == "mlflow" or (not self.feature_engineer):
            X = df.values
            feature_names = df.columns.tolist()
        else:
            X, _ = self.feature_engineer.get_feature_matrix(df)
            feature_names = self.feature_engineer.feature_names

        # Calculate SHAP values
        # For RandomForest, shap_values is a list [class_0, class_1]
        shap_values = self.explainer.shap_values(X)
        
        # We focus on class 1 (Ponto de Virada)
        if isinstance(shap_values, list):
            # SHAP version < 0.45 or specific formats
            cls_1_shap = shap_values[1][0] if len(shap_values) > 1 else shap_values[0]
        else:
            # SHAP version >= 0.45 often returns (n_samples, n_features, n_classes)
            if len(shap_values.shape) == 3:
                cls_1_shap = shap_values[0, :, 1]
            else:
                cls_1_shap = shap_values[0]

        # Map to feature names and sort
        contributions = []
        for i, val in enumerate(cls_1_shap):
            contributions.append({
                "feature": feature_names[i],
                "contribution": round(float(val), 4)
            })
            
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        
        # Get base value
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_val = float(self.explainer.expected_value[1])
        else:
            base_val = float(self.explainer.expected_value)
            
        return {
            "base_value": round(base_val, 4),
            "top_contributions": contributions[:top_n],
            "all_contributions": contributions
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return metadata about the loaded model.

        Returns:
            Model metadata dictionary.
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
        Return model feature names.

        Returns:
            Feature name list.
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
        Factory method to create predictor from MLflow.

        Args:
            model_name: Registered model name.
            stage: Stage/alias (Production, Staging).

        Returns:
            Configured `ModelPredictor` instance.
        """
        return cls(mlflow_model_name=model_name, mlflow_stage=stage)
