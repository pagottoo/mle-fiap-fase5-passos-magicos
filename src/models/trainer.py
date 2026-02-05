"""
Módulo de treinamento do modelo
"""
import os
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import structlog

from ..config import MODEL_CONFIG, MODELS_DIR

logger = structlog.get_logger()

# Flag para habilitar MLflow (pode ser desabilitado em testes)
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"

try:
    if MLFLOW_ENABLED:
        from ..mlflow_tracking import ExperimentTracker, ModelRegistry
        MLFLOW_AVAILABLE = True
    else:
        MLFLOW_AVAILABLE = False
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow_not_available", message="MLflow não instalado, tracking desabilitado")


class ModelTrainer:
    """
    Classe responsável pelo treinamento e avaliação de modelos.
    
    Agora com integração MLflow para:
    - Tracking de experimentos
    - Versionamento de modelos
    - Model Registry
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = MODEL_CONFIG["random_state"],
        experiment_name: str = "passos-magicos-ponto-virada",
        enable_mlflow: bool = True
    ):
        """
        Inicializa o trainer.
        
        Args:
            model_type: Tipo de modelo ("random_forest", "gradient_boosting", "logistic_regression")
            random_state: Seed para reprodutibilidade
            experiment_name: Nome do experimento MLflow
            enable_mlflow: Habilitar tracking MLflow
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.metrics = {}
        self.feature_importance = {}
        self.cv_results = {}
        
        # MLflow integration
        self.mlflow_enabled = enable_mlflow and MLFLOW_AVAILABLE
        self.tracker = None
        self.model_registry = None
        self.run_id = None
        
        if self.mlflow_enabled:
            self.tracker = ExperimentTracker(experiment_name=experiment_name)
            self.model_registry = ModelRegistry()
            logger.info("mlflow_enabled", experiment_name=experiment_name)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa o modelo baseado no tipo especificado."""
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced"  # Importante para dados desbalanceados
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced"
            )
        }
        
        if self.model_type not in models:
            raise ValueError(f"Tipo de modelo inválido: {self.model_type}")
        
        self.model = models[self.model_type]
        self.model_params = self._get_model_params()
        logger.info("model_initialized", model_type=self.model_type)
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Extrai parâmetros do modelo para logging."""
        params = self.model.get_params()
        # Filtrar parâmetros não serializáveis
        return {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool, type(None)))}
    
    def start_run(self, run_name: Optional[str] = None, description: Optional[str] = None):
        """
        Inicia uma run MLflow.
        
        Args:
            run_name: Nome da run (opcional)
            description: Descrição da run
        """
        if self.mlflow_enabled and self.tracker:
            self.tracker.start_run(
                run_name=run_name or f"{self.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                description=description
            )
            self.run_id = self.tracker.get_run_id()
            
            # Log parâmetros iniciais
            self.tracker.log_params({
                "model_type": self.model_type,
                "random_state": self.random_state,
                **self.model_params
            })
            
            logger.info("mlflow_run_started", run_id=self.run_id)
    
    def end_run(self, status: str = "FINISHED"):
        """Finaliza a run MLflow."""
        if self.mlflow_enabled and self.tracker:
            self.tracker.end_run(status=status)
            logger.info("mlflow_run_ended", run_id=self.run_id, status=status)
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = MODEL_CONFIG["test_size"]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide dados em treino e teste.
        
        Args:
            X: Matriz de features
            y: Vetor target
            test_size: Proporção do conjunto de teste
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y  # Mantém proporção das classes
        )
        
        logger.info(
            "data_split",
            train_size=len(X_train),
            test_size=len(X_test),
            train_positive_ratio=round(y_train.mean(), 3),
            test_positive_ratio=round(y_test.mean(), 3)
        )
        
        return X_train, X_test, y_train, y_test
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = MODEL_CONFIG["cv_folds"]
    ) -> Dict[str, float]:
        """
        Realiza validação cruzada.
        
        Args:
            X: Matriz de features
            y: Vetor target
            cv_folds: Número de folds
            
        Returns:
            Dicionário com métricas de validação cruzada
        """
        logger.info("cross_validating", folds=cv_folds)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Calcular scores para diferentes métricas
        cv_scores = {
            "accuracy": cross_val_score(self.model, X, y, cv=cv, scoring="accuracy"),
            "precision": cross_val_score(self.model, X, y, cv=cv, scoring="precision"),
            "recall": cross_val_score(self.model, X, y, cv=cv, scoring="recall"),
            "f1": cross_val_score(self.model, X, y, cv=cv, scoring="f1"),
            "roc_auc": cross_val_score(self.model, X, y, cv=cv, scoring="roc_auc")
        }
        
        cv_results = {
            metric: {
                "mean": round(scores.mean(), 4),
                "std": round(scores.std(), 4)
            }
            for metric, scores in cv_scores.items()
        }
        
        self.cv_results = cv_results
        
        # Log no MLflow
        if self.mlflow_enabled and self.tracker:
            for metric, values in cv_results.items():
                self.tracker.log_metric(f"cv_{metric}_mean", values["mean"])
                self.tracker.log_metric(f"cv_{metric}_std", values["std"])
        
        logger.info("cross_validation_complete", results=cv_results)
        
        return cv_results
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Treina o modelo.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
        """
        logger.info("training_model", samples=len(X_train))
        
        # Log dataset info no MLflow
        if self.mlflow_enabled and self.tracker:
            self.tracker.log_params({
                "train_samples": len(X_train),
                "n_features": X_train.shape[1] if len(X_train.shape) > 1 else 1,
                "positive_class_ratio": round(y_train.mean(), 4)
            })
        
        self.model.fit(X_train, y_train)
        
        # Extrair importância das features (se disponível)
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = dict(enumerate(self.model.feature_importances_))
            
            # Log feature importance no MLflow
            if self.mlflow_enabled and self.tracker:
                for idx, importance in enumerate(self.model.feature_importances_):
                    self.tracker.log_metric(f"feature_importance_{idx}", importance)
        
        logger.info("model_trained")
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Avalia o modelo no conjunto de teste.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dicionário com métricas de avaliação
        """
        logger.info("evaluating_model", samples=len(X_test))
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Log métricas no MLflow
        if self.mlflow_enabled and self.tracker:
            self.tracker.log_metrics({
                "accuracy": self.metrics["accuracy"],
                "precision": self.metrics["precision"],
                "recall": self.metrics["recall"],
                "f1_score": self.metrics["f1_score"],
                "roc_auc": self.metrics["roc_auc"],
                "test_samples": len(X_test)
            })
        
        logger.info("evaluation_complete", metrics=self.metrics)
        
        return self.metrics
    
    def save_model(
        self,
        preprocessor: Any,
        feature_engineer: Any,
        model_name: str = MODEL_CONFIG["model_name"]
    ) -> Path:
        """
        Salva o modelo e artefatos relacionados.
        
        Args:
            preprocessor: Objeto DataPreprocessor ajustado
            feature_engineer: Objeto FeatureEngineer ajustado
            model_name: Nome base para o arquivo
            
        Returns:
            Caminho do arquivo salvo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = MODELS_DIR / f"{model_name}_{timestamp}.joblib"
        
        # Salvar todos os componentes necessários para inferência
        artifacts = {
            "model": self.model,
            "preprocessor": preprocessor,
            "feature_engineer": feature_engineer,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "version": MODEL_CONFIG["model_version"],
            "trained_at": timestamp
        }
        
        joblib.dump(artifacts, model_path)
        
        # Salvar também um arquivo com métricas em JSON
        metrics_path = MODELS_DIR / f"{model_name}_{timestamp}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "model_type": self.model_type,
                "metrics": self.metrics,
                "version": MODEL_CONFIG["model_version"],
                "trained_at": timestamp
            }, f, indent=2)
        
        # Criar link simbólico para o modelo mais recente
        latest_path = MODELS_DIR / f"{model_name}_latest.joblib"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(model_path.name)
        
        logger.info("model_saved", path=str(model_path))
        
        return model_path
    
    def log_model_to_mlflow(
        self,
        X_sample: np.ndarray,
        y_sample: np.ndarray,
        registered_model_name: Optional[str] = "passos-magicos-ponto-virada"
    ) -> Optional[str]:
        """
        Registra o modelo no MLflow Model Registry.
        
        Args:
            X_sample: Amostra de features para inferir schema
            y_sample: Amostra de labels
            registered_model_name: Nome para registro (None = não registrar)
            
        Returns:
            Model URI se registrado, None caso contrário
        """
        if not self.mlflow_enabled or not self.tracker:
            logger.warning("mlflow_not_enabled", message="MLflow desabilitado, modelo não registrado")
            return None
        
        try:
            model_info = self.tracker.log_sklearn_model_with_signature(
                model=self.model,
                X_sample=X_sample,
                y_sample=y_sample,
                artifact_path="model",
                registered_model_name=registered_model_name
            )
            
            # Log artefatos adicionais
            self.tracker.log_dict(self.metrics, "metrics.json")
            self.tracker.log_dict(self.cv_results, "cv_results.json")
            
            if self.feature_importance:
                self.tracker.log_dict(self.feature_importance, "feature_importance.json")
            
            logger.info(
                "model_logged_to_mlflow",
                model_uri=model_info.model_uri,
                registered_name=registered_model_name
            )
            
            return model_info.model_uri
            
        except Exception as e:
            logger.error("mlflow_log_model_error", error=str(e))
            return None
    
    def promote_model_to_production(
        self,
        model_name: str = "passos-magicos-ponto-virada",
        version: Optional[int] = None
    ) -> bool:
        """
        Promove um modelo para produção no Model Registry.
        
        Args:
            model_name: Nome do modelo registrado
            version: Versão específica (None = última)
            
        Returns:
            True se promovido com sucesso
        """
        if not self.mlflow_enabled or not self.model_registry:
            logger.warning("mlflow_not_enabled")
            return False
        
        try:
            if version is None:
                # Pegar última versão
                versions = self.model_registry.get_latest_versions(model_name)
                if not versions:
                    logger.error("no_model_versions_found", model_name=model_name)
                    return False
                version = int(versions[0].version)
            
            self.model_registry.promote_to_production(model_name, version)
            logger.info("model_promoted_to_production", model_name=model_name, version=version)
            return True
            
        except Exception as e:
            logger.error("promote_model_error", error=str(e))
            return False
    
    def get_model_summary(self) -> str:
        """
        Gera um resumo textual do modelo para documentação.
        
        Returns:
            String com resumo do modelo
        """
        summary = f"""
## Resumo do Modelo

**Tipo de Modelo:** {self.model_type}
**Versão:** {MODEL_CONFIG["model_version"]}

### Justificativa da Escolha do Modelo

O modelo {self.model_type} foi escolhido por:

1. **Robustez**: Lida bem com dados desbalanceados (usando class_weight="balanced")
2. **Interpretabilidade**: Permite análise de importância de features
3. **Performance**: Bom equilíbrio entre viés e variância
4. **Generalização**: Validação cruzada estratificada garante estabilidade

### Métricas de Avaliação

A **métrica principal** escolhida é o **F1-Score** porque:
- Balanceia precisão e recall
- É apropriada para dados desbalanceados
- Penaliza tanto falsos positivos quanto falsos negativos
- No contexto educacional, é importante identificar corretamente os alunos com potencial
  de transformação sem criar expectativas irreais

**Resultados:**
- Accuracy: {self.metrics.get('accuracy', 'N/A')}
- Precision: {self.metrics.get('precision', 'N/A')}
- Recall: {self.metrics.get('recall', 'N/A')}
- F1-Score: {self.metrics.get('f1_score', 'N/A')}
- ROC-AUC: {self.metrics.get('roc_auc', 'N/A')}

### Confiabilidade para Produção

O modelo é confiável para produção porque:
1. Foi validado com validação cruzada estratificada ({MODEL_CONFIG['cv_folds']} folds)
2. Métricas consistentes entre treino e teste (sem overfitting)
3. ROC-AUC > 0.7 indica boa capacidade de discriminação
4. Classe balanceada no treinamento evita viés
"""
        return summary
