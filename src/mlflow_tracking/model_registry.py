"""
Model Registry - MLflow Model Management

Responsável por:
- Registrar modelos no MLflow Model Registry
- Gerenciar versões de modelos
- Transicionar modelos via aliases (staging, production)
- Carregar modelos para inferência
"""
import os
from typing import Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import structlog

logger = structlog.get_logger()


# Stages disponíveis no MLflow
class ModelStage:
    """Constantes para stages de modelo."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelAlias:
    """Aliases usados no Model Registry."""
    STAGING = "staging"
    PRODUCTION = "production"


class ModelRegistry:
    """
    Gerencia o Model Registry do MLflow.
    
    Permite:
    - Registrar novos modelos
    - Listar versões
    - Promover/rebaixar modelos entre stages
    - Carregar modelos para produção
    
    Exemplo de uso:
        registry = ModelRegistry()
        
        # Registrar modelo
        registry.register_model(
            model_uri="runs:/abc123/model",
            name="passos-magicos-classifier"
        )
        
        # Promover para produção
        registry.transition_model_stage(
            name="passos-magicos-classifier",
            version=1,
            stage="Production"
        )
        
        # Carregar modelo de produção
        model = registry.load_production_model("passos-magicos-classifier")
    """
    
    DEFAULT_MODEL_NAME = "passos-magicos-ponto-virada"
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Inicializa o Model Registry.
        
        Args:
            tracking_uri: URI do MLflow server
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()
        
        logger.info("model_registry_initialized", tracking_uri=self.tracking_uri)

    def _normalize_stage(self, stage: str) -> str:
        """Normaliza o nome do stage para as constantes internas."""
        if not stage:
            return ModelStage.NONE

        value = stage.strip().lower()
        if value == "staging":
            return ModelStage.STAGING
        if value == "production":
            return ModelStage.PRODUCTION
        if value == "archived":
            return ModelStage.ARCHIVED
        if value == "none":
            return ModelStage.NONE
        return stage

    def _stage_to_alias(self, stage: str) -> Optional[str]:
        """Mapeia stage legado para alias moderno."""
        normalized = self._normalize_stage(stage)
        if normalized == ModelStage.STAGING:
            return ModelAlias.STAGING
        if normalized == ModelStage.PRODUCTION:
            return ModelAlias.PRODUCTION
        return None

    def _get_model_version_by_alias(
        self, name: str, alias: str
    ) -> Optional[mlflow.entities.model_registry.ModelVersion]:
        """Obtém versão por alias, retornando None quando não existe."""
        try:
            return self.client.get_model_version_by_alias(name, alias)
        except MlflowException:
            return None

    def _remove_alias_if_points_to_version(self, name: str, alias: str, version: int) -> None:
        """Remove alias quando ele aponta para uma versão específica."""
        current = self._get_model_version_by_alias(name, alias)
        if current is None or str(current.version) != str(version):
            return

        delete_alias = getattr(self.client, "delete_registered_model_alias", None)
        if delete_alias is None:
            return

        try:
            delete_alias(name, alias)
        except MlflowException:
            logger.warning("model_alias_remove_failed", name=name, alias=alias, version=version)
    
    def register_model(
        self,
        model_uri: str,
        name: str = DEFAULT_MODEL_NAME,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Registra um modelo no Model Registry.
        
        Args:
            model_uri: URI do modelo (ex: "runs:/run_id/model")
            name: Nome do modelo registrado
            tags: Tags para a versão
            description: Descrição da versão
            
        Returns:
            ModelVersion com informações da versão criada
        """
        # Criar modelo registrado se não existe
        try:
            self.client.get_registered_model(name)
        except MlflowException:
            self.client.create_registered_model(
                name=name,
                description=f"Modelo de predição de Ponto de Virada - Passos Mágicos"
            )
            logger.info("registered_model_created", name=name)
        
        # Registrar nova versão
        model_version = mlflow.register_model(model_uri=model_uri, name=name)
        
        # Adicionar tags e descrição
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(name, model_version.version, key, value)
        
        if description:
            self.client.update_model_version(
                name=name,
                version=model_version.version,
                description=description
            )
        
        logger.info(
            "model_registered",
            name=name,
            version=model_version.version,
            source=model_uri
        )
        
        return model_version
    
    def transition_model_stage(
        self,
        name: str,
        version: int,
        stage: str,
        archive_existing_versions: bool = True
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Transiciona modelo para um novo stage.
        
        Args:
            name: Nome do modelo registrado
            version: Versão a transicionar
            stage: Novo stage (Staging, Production, Archived)
            archive_existing_versions: Arquivar outras versões no mesmo stage
            
        Returns:
            ModelVersion atualizada
        """
        normalized_stage = self._normalize_stage(stage)
        if normalized_stage not in [ModelStage.STAGING, ModelStage.PRODUCTION, ModelStage.ARCHIVED, ModelStage.NONE]:
            raise ValueError(f"Stage inválido: {stage}")

        alias = self._stage_to_alias(normalized_stage)
        if alias:
            # API moderna do MLflow: aliases substituem stages legados.
            self.client.set_registered_model_alias(name=name, alias=alias, version=str(version))
            model_version = self.client.get_model_version(name=name, version=str(version))

            logger.info(
                "model_alias_updated",
                name=name,
                version=version,
                alias=alias,
                stage=normalized_stage,
            )
            return model_version

        # Stages legados não mapeados para alias: mantém apenas metadado e
        # remove aliases de serving quando aplicável.
        if normalized_stage in [ModelStage.ARCHIVED, ModelStage.NONE]:
            self._remove_alias_if_points_to_version(name, ModelAlias.PRODUCTION, version)
            self._remove_alias_if_points_to_version(name, ModelAlias.STAGING, version)

            try:
                self.client.set_model_version_tag(name, str(version), "lifecycle_stage", normalized_stage)
            except MlflowException:
                logger.warning(
                    "model_lifecycle_tag_failed",
                    name=name,
                    version=version,
                    stage=normalized_stage,
                )

            model_version = self.client.get_model_version(name=name, version=str(version))
            logger.info(
                "model_lifecycle_updated",
                name=name,
                version=version,
                stage=normalized_stage,
                archive_existing_versions=archive_existing_versions,
            )
            return model_version

        # Guard clause (não deve acontecer, mas deixa fluxo explícito).
        raise ValueError(f"Stage inválido: {stage}")
    
    def promote_to_staging(self, name: str, version: int) -> mlflow.entities.model_registry.ModelVersion:
        """Promove modelo para Staging."""
        return self.transition_model_stage(name, version, ModelStage.STAGING)
    
    def promote_to_production(self, name: str, version: int) -> mlflow.entities.model_registry.ModelVersion:
        """Promove modelo para Production."""
        return self.transition_model_stage(name, version, ModelStage.PRODUCTION)
    
    def archive_model(self, name: str, version: int) -> mlflow.entities.model_registry.ModelVersion:
        """Arquiva uma versão do modelo."""
        return self.transition_model_stage(name, version, ModelStage.ARCHIVED)
    
    def get_latest_versions(
        self,
        name: str = DEFAULT_MODEL_NAME,
        stages: Optional[List[str]] = None
    ) -> List[mlflow.entities.model_registry.ModelVersion]:
        """
        Obtém as versões mais recentes de um modelo por stage.
        
        Args:
            name: Nome do modelo
            stages: Lista de stages (None = todos)
            
        Returns:
            Lista de ModelVersion
        """
        try:
            if stages:
                versions: List[mlflow.entities.model_registry.ModelVersion] = []
                for stage in stages:
                    alias = self._stage_to_alias(stage)
                    if alias:
                        mv = self._get_model_version_by_alias(name, alias)
                        if mv:
                            versions.append(mv)
                        continue

                    # Fallback para filtros não mapeados em alias.
                    normalized = self._normalize_stage(stage)
                    candidates = [
                        v for v in self.search_model_versions(f"name='{name}'")
                        if self._normalize_stage(getattr(v, "current_stage", ModelStage.NONE)) == normalized
                    ]
                    if candidates:
                        versions.append(sorted(candidates, key=lambda x: int(x.version), reverse=True)[0])
                return versions

            versions = self.search_model_versions(f"name='{name}'")
            return sorted(versions, key=lambda x: int(x.version), reverse=True)
        except MlflowException:
            logger.warning("model_not_found", name=name)
            return []
    
    def get_production_version(self, name: str = DEFAULT_MODEL_NAME) -> Optional[mlflow.entities.model_registry.ModelVersion]:
        """Obtém a versão em produção."""
        return self._get_model_version_by_alias(name, ModelAlias.PRODUCTION)
    
    def get_staging_version(self, name: str = DEFAULT_MODEL_NAME) -> Optional[mlflow.entities.model_registry.ModelVersion]:
        """Obtém a versão em staging."""
        return self._get_model_version_by_alias(name, ModelAlias.STAGING)
    
    def load_model(
        self,
        name: str = DEFAULT_MODEL_NAME,
        version: Optional[int] = None,
        stage: Optional[str] = None
    ):
        """
        Carrega modelo do registry.
        
        Args:
            name: Nome do modelo
            version: Versão específica (prioridade sobre stage)
            stage: Stage do modelo (Production, Staging, etc.)
            
        Returns:
            Modelo carregado
        """
        if version:
            model_uri = f"models:/{name}/{version}"
        elif stage:
            alias = self._stage_to_alias(stage)
            if alias:
                model_uri = f"models:/{name}@{alias}"
            else:
                model_uri = f"models:/{name}/{stage}"
        else:
            # Padrão: Production, se não existir, última versão
            prod_version = self.get_production_version(name)
            if prod_version:
                model_uri = f"models:/{name}@{ModelAlias.PRODUCTION}"
            else:
                model_uri = f"models:/{name}/latest"
        
        logger.info("loading_model", model_uri=model_uri)
        model = mlflow.sklearn.load_model(model_uri)
        
        return model
    
    def load_production_model(self, name: str = DEFAULT_MODEL_NAME):
        """Carrega o modelo em produção."""
        return self.load_model(name, stage=ModelStage.PRODUCTION)
    
    def load_staging_model(self, name: str = DEFAULT_MODEL_NAME):
        """Carrega o modelo em staging."""
        return self.load_model(name, stage=ModelStage.STAGING)
    
    def list_registered_models(self) -> List[mlflow.entities.model_registry.RegisteredModel]:
        """Lista todos os modelos registrados."""
        return list(self.client.search_registered_models())
    
    def get_model_version_details(
        self,
        name: str,
        version: int
    ) -> Dict:
        """
        Obtém detalhes de uma versão específica.
        
        Returns:
            Dicionário com detalhes da versão
        """
        mv = self.client.get_model_version(name, str(version))

        detected_stage = self._normalize_stage(getattr(mv, "current_stage", ModelStage.NONE))
        if detected_stage in [ModelStage.NONE, "", None]:
            prod = self.get_production_version(name)
            staging = self.get_staging_version(name)
            if prod and str(prod.version) == str(mv.version):
                detected_stage = ModelStage.PRODUCTION
            elif staging and str(staging.version) == str(mv.version):
                detected_stage = ModelStage.STAGING
            else:
                detected_stage = ModelStage.NONE
        
        return {
            "name": mv.name,
            "version": mv.version,
            "stage": detected_stage,
            "status": mv.status,
            "source": mv.source,
            "run_id": mv.run_id,
            "creation_timestamp": mv.creation_timestamp,
            "last_updated_timestamp": mv.last_updated_timestamp,
            "description": mv.description,
            "tags": dict(mv.tags) if mv.tags else {}
        }
    
    def compare_model_versions(
        self,
        name: str,
        version1: int,
        version2: int
    ) -> Dict:
        """
        Compara duas versões de um modelo.
        
        Returns:
            Dicionário com comparação de métricas
        """
        v1 = self.get_model_version_details(name, version1)
        v2 = self.get_model_version_details(name, version2)
        
        # Buscar runs para obter métricas
        run1 = self.client.get_run(v1["run_id"]) if v1.get("run_id") else None
        run2 = self.client.get_run(v2["run_id"]) if v2.get("run_id") else None
        
        return {
            "version1": {
                "version": version1,
                "stage": v1["stage"],
                "metrics": run1.data.metrics if run1 else {},
                "params": run1.data.params if run1 else {}
            },
            "version2": {
                "version": version2,
                "stage": v2["stage"],
                "metrics": run2.data.metrics if run2 else {},
                "params": run2.data.params if run2 else {}
            }
        }
    
    def delete_model_version(self, name: str, version: int) -> None:
        """Deleta uma versão do modelo."""
        self.client.delete_model_version(name, str(version))
        logger.info("model_version_deleted", name=name, version=version)
    
    def delete_registered_model(self, name: str) -> None:
        """Deleta um modelo registrado (todas as versões)."""
        self.client.delete_registered_model(name)
        logger.info("registered_model_deleted", name=name)
    
    def search_model_versions(
        self,
        filter_string: str = "",
        max_results: int = 100
    ) -> List[mlflow.entities.model_registry.ModelVersion]:
        """
        Busca versões de modelo com filtro.
        
        Args:
            filter_string: Filtro MLflow (ex: "name='model-name'")
            max_results: Máximo de resultados
        """
        return list(self.client.search_model_versions(
            filter_string=filter_string,
            max_results=max_results
        ))
    
    def get_model_uri(
        self,
        name: str = DEFAULT_MODEL_NAME,
        version: Optional[int] = None,
        stage: Optional[str] = None
    ) -> str:
        """
        Gera URI do modelo para carregamento.
        
        Args:
            name: Nome do modelo
            version: Versão específica
            stage: Stage (Production, Staging)
            
        Returns:
            URI no formato models:/name/version_or_stage
        """
        if version:
            return f"models:/{name}/{version}"
        elif stage:
            alias = self._stage_to_alias(stage)
            if alias:
                return f"models:/{name}@{alias}"
            return f"models:/{name}/{stage}"
        else:
            return f"models:/{name}/latest"
    
    def set_model_version_tag(
        self,
        name: str,
        version: int,
        key: str,
        value: str
    ) -> None:
        """Define uma tag em uma versão do modelo."""
        self.client.set_model_version_tag(name, str(version), key, value)
    
    def get_registry_status(self) -> Dict:
        """
        Retorna status do Model Registry.
        
        Returns:
            Dicionário com estatísticas
        """
        models = self.list_registered_models()
        
        status = {
            "total_models": len(models),
            "models": []
        }
        
        for model in models:
            prod = self.get_production_version(model.name)
            staging = self.get_staging_version(model.name)
            
            model_info = {
                "name": model.name,
                "total_versions": len(list(self.search_model_versions(f"name='{model.name}'"))),
                "latest_versions": {}
            }
            
            if prod:
                model_info["latest_versions"][ModelStage.PRODUCTION] = {
                    "version": prod.version,
                    "status": prod.status
                }
            if staging:
                model_info["latest_versions"][ModelStage.STAGING] = {
                    "version": staging.version,
                    "status": staging.status
                }
            
            status["models"].append(model_info)
        
        return status
