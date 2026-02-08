"""
Experiment Tracker - MLflow Integration

Responsável por rastrear experimentos de ML:
- Parâmetros de treinamento
- Métricas de avaliação
- Artefatos (modelo, gráficos, datasets)
- Metadados do ambiente
"""
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
import structlog

logger = structlog.get_logger()


class ExperimentTracker:
    """
    Gerencia tracking de experimentos com MLflow.
    
    Exemplo de uso:
        tracker = ExperimentTracker(experiment_name="passos-magicos")
        
        with tracker.start_run(run_name="rf-v1"):
            tracker.log_params({"n_estimators": 100})
            # ... treinar modelo ...
            tracker.log_metrics({"f1_score": 0.85})
            tracker.log_model(model, "random_forest")
    """
    
    def __init__(
        self,
        experiment_name: str = "passos-magicos-ponto-virada",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Inicializa o tracker.
        
        Args:
            experiment_name: Nome do experimento MLflow
            tracking_uri: URI do MLflow server (None = local ./mlruns)
            artifact_location: Onde armazenar artefatos
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        self.artifact_location = artifact_location
        
        # Configurar MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Criar ou obter experimento
        self.experiment = self._get_or_create_experiment()
        mlflow.set_experiment(self.experiment_name)
        
        self.client = MlflowClient()
        self.active_run = None
        
        logger.info(
            "experiment_tracker_initialized",
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri,
            experiment_id=self.experiment.experiment_id
        )
    
    def _get_or_create_experiment(self) -> mlflow.entities.Experiment:
        """Obtém experimento existente ou cria novo."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=self.artifact_location
            )
            experiment = mlflow.get_experiment(experiment_id)
            logger.info("experiment_created", experiment_id=experiment_id)
        
        return experiment
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ):
        """
        Inicia uma nova run de experimento.
        
        Pode ser usado como context manager:
            with tracker.start_run("minha-run"):
                ...
        """
        default_tags = {
            "project": "passos-magicos",
            "objective": "ponto_de_virada",
            "team": "mle-fiap"
        }
        
        if tags:
            default_tags.update(tags)
        
        if description:
            default_tags["mlflow.note.content"] = description
        
        self.active_run = mlflow.start_run(
            run_name=run_name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            experiment_id=self.experiment.experiment_id,
            tags=default_tags
        )
        
        logger.info(
            "run_started",
            run_id=self.active_run.info.run_id,
            run_name=run_name
        )
        
        return self
    
    def __enter__(self):
        """Suporte a context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finaliza run ao sair do context."""
        self.end_run()
        return False
    
    def end_run(self, status: str = "FINISHED"):
        """Finaliza a run atual."""
        if self.active_run:
            mlflow.end_run(status=status)
            logger.info("run_ended", run_id=self.active_run.info.run_id, status=status)
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Registra parâmetros de treinamento.
        
        Args:
            params: Dicionário de parâmetros (ex: {"n_estimators": 100})
        """
        # MLflow aceita apenas strings, números e booleanos
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, (str, int, float, bool)):
                clean_params[k] = v
            else:
                clean_params[k] = str(v)
        
        mlflow.log_params(clean_params)
        logger.debug("params_logged", count=len(clean_params))
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Registra métricas de avaliação.
        
        Args:
            metrics: Dicionário de métricas (ex: {"f1_score": 0.85})
            step: Step/época opcional para métricas temporais
        """
        mlflow.log_metrics(metrics, step=step)
        logger.debug("metrics_logged", count=len(metrics), step=step)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Registra uma única métrica."""
        mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Registra um arquivo como artefato.
        
        Args:
            local_path: Caminho local do arquivo
            artifact_path: Subdiretório no artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug("artifact_logged", path=local_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Registra todos os arquivos de um diretório."""
        mlflow.log_artifacts(local_dir, artifact_path)
        logger.debug("artifacts_logged", dir=local_dir)
    
    def log_dict(self, dictionary: Dict, artifact_file: str) -> None:
        """
        Salva um dicionário como JSON no artifact store.
        
        Args:
            dictionary: Dicionário a salvar
            artifact_file: Nome do arquivo (ex: "config.json")
        """
        mlflow.log_dict(dictionary, artifact_file)
    
    def log_figure(self, figure, artifact_file: str) -> None:
        """
        Salva uma figura matplotlib/plotly.
        
        Args:
            figure: Objeto figure
            artifact_file: Nome do arquivo (ex: "confusion_matrix.png")
        """
        mlflow.log_figure(figure, artifact_file)
    
    def log_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        signature=None,
        input_example=None,
        pip_requirements: Optional[List[str]] = None
    ) -> mlflow.models.model.ModelInfo:
        """
        Registra modelo sklearn no MLflow.
        
        Args:
            model: Modelo treinado (sklearn)
            artifact_path: Nome do artefato (ex: "model")
            registered_model_name: Se fornecido, registra no Model Registry
            signature: Assinatura do modelo (input/output schema)
            input_example: Exemplo de input para inferência
            pip_requirements: Dependências Python
            
        Returns:
            ModelInfo com URI e metadados
        """
        # Inferir requirements se não fornecido
        if pip_requirements is None:
            pip_requirements = [
                "scikit-learn>=1.3.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0"
            ]
        
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements
        )
        
        logger.info(
            "model_logged",
            model_uri=model_info.model_uri,
            registered_name=registered_model_name
        )
        
        return model_info
    
    def log_sklearn_model_with_signature(
        self,
        model,
        X_sample,
        y_sample,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ) -> mlflow.models.model.ModelInfo:
        """
        Registra modelo sklearn com assinatura inferida automaticamente.
        
        Args:
            model: Modelo sklearn treinado
            X_sample: Amostra de features (para inferir schema)
            y_sample: Amostra de labels (para inferir schema)
            artifact_path: Nome do artefato
            registered_model_name: Nome para Model Registry
        """
        from mlflow.models.signature import infer_signature
        
        # Fazer predição para inferir output schema
        predictions = model.predict(X_sample)
        
        # Inferir assinatura
        signature = infer_signature(X_sample, predictions)
        
        return self.log_model(
            model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=X_sample[:1]  # Primeiro exemplo
        )
    
    def log_training_dataset(
        self,
        df,
        name: str = "training_data",
        targets: Optional[str] = None
    ) -> None:
        """
        Registra dataset de treinamento.
        
        Args:
            df: DataFrame com dados de treino
            name: Nome do dataset
            targets: Nome da coluna target
        """
        dataset = mlflow.data.from_pandas(df, name=name, targets=targets)
        mlflow.log_input(dataset, context="training")
        logger.debug("dataset_logged", name=name, rows=len(df))
    
    def set_tag(self, key: str, value: str) -> None:
        """Define uma tag na run atual."""
        mlflow.set_tag(key, value)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Define múltiplas tags."""
        mlflow.set_tags(tags)
    
    def get_run_id(self) -> Optional[str]:
        """Retorna ID da run atual."""
        if self.active_run:
            return self.active_run.info.run_id
        return None
    
    def get_artifact_uri(self) -> Optional[str]:
        """Retorna URI dos artefatos da run atual."""
        if self.active_run:
            return self.active_run.info.artifact_uri
        return None
    
    def list_runs(
        self,
        max_results: int = 100,
        order_by: Optional[List[str]] = None
    ) -> List[mlflow.entities.Run]:
        """
        Lista runs do experimento.
        
        Args:
            max_results: Número máximo de resultados
            order_by: Ordenação (ex: ["metrics.f1_score DESC"])
        """
        if order_by is None:
            order_by = ["start_time DESC"]
        
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            max_results=max_results,
            order_by=order_by
        )
        
        return runs
    
    def get_best_run(self, metric: str = "f1_score", ascending: bool = False) -> Optional[mlflow.entities.Run]:
        """
        Retorna a melhor run baseada em uma métrica.
        
        Args:
            metric: Nome da métrica
            ascending: True se menor é melhor
        """
        order = "ASC" if ascending else "DESC"
        runs = self.list_runs(
            max_results=1,
            order_by=[f"metrics.{metric} {order}"]
        )
        
        return runs[0] if runs else None
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Dict]:
        """
        Compara múltiplas runs.
        
        Returns:
            Dicionário com métricas de cada run
        """
        comparison = {}
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
                "status": run.info.status,
                "start_time": run.info.start_time
            }
        
        return comparison
