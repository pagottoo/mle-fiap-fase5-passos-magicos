"""
Model Registry - MLflow Model Management

Responsible for:
- Registering models in the MLflow Model Registry
- Managing model versions
- Transitioning models via aliases (staging, production)
- Loading models for inference
"""
import os
from typing import Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import structlog

logger = structlog.get_logger()


# Available stages in MLflow
class ModelStage:
    """Model stage constants."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelAlias:
    """Aliases used in the Model Registry."""
    STAGING = "staging"
    PRODUCTION = "production"


class ModelRegistry:
    """
    Manage the MLflow Model Registry.
    
    It supports:
    - Registering new models
    - Listing versions
    - Promoting/demoting models across stages
    - Loading models for production
    
    Example:
        registry = ModelRegistry()
        
        # Register model
        registry.register_model(
            model_uri="runs:/abc123/model",
            name="passos-magicos-classifier"
        )
        
        # Promote to production
        registry.transition_model_stage(
            name="passos-magicos-classifier",
            version=1,
            stage="Production"
        )
        
        # Load production model
        model = registry.load_production_model("passos-magicos-classifier")
    """
    
    DEFAULT_MODEL_NAME = "passos-magicos-ponto-virada"
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize the Model Registry.
        
        Args:
            tracking_uri: MLflow server URI
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()
        
        logger.info("model_registry_initialized", tracking_uri=self.tracking_uri)

    def _normalize_stage(self, stage: str) -> str:
        """Normalize stage names to internal constants."""
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
        """Map legacy stage names to modern aliases."""
        normalized = self._normalize_stage(stage)
        if normalized == ModelStage.STAGING:
            return ModelAlias.STAGING
        if normalized == ModelStage.PRODUCTION:
            return ModelAlias.PRODUCTION
        return None

    def _get_model_version_by_alias(
        self, name: str, alias: str
    ) -> Optional[mlflow.entities.model_registry.ModelVersion]:
        """Get model version by alias; return None when it does not exist."""
        try:
            return self.client.get_model_version_by_alias(name, alias)
        except MlflowException:
            return None

    def _remove_alias_if_points_to_version(self, name: str, alias: str, version: int) -> None:
        """Remove an alias if it currently points to a specific version."""
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
        Register a model in the Model Registry.
        
        Args:
            model_uri: Model URI (example: "runs:/run_id/model")
            name: Registered model name
            tags: Version tags
            description: Version description
            
        Returns:
            ModelVersion with created version metadata
        """
        # Create the registered model if needed.
        try:
            self.client.get_registered_model(name)
        except MlflowException:
            self.client.create_registered_model(
                name=name,
                description="Passos Magicos turning-point prediction model"
            )
            logger.info("registered_model_created", name=name)
        
        # Register new version.
        model_version = mlflow.register_model(model_uri=model_uri, name=name)
        
        # Attach tags and description.
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
        Transition model to a new stage.
        
        Args:
            name: Registered model name
            version: Version to transition
            stage: Target stage (Staging, Production, Archived)
            archive_existing_versions: Archive other versions in the same stage
            
        Returns:
            Updated ModelVersion
        """
        normalized_stage = self._normalize_stage(stage)
        if normalized_stage not in [ModelStage.STAGING, ModelStage.PRODUCTION, ModelStage.ARCHIVED, ModelStage.NONE]:
            raise ValueError(f"Invalid stage: {stage}")

        alias = self._stage_to_alias(normalized_stage)
        if alias:
            # Modern MLflow API: aliases replace legacy stage transitions.
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

        # For legacy stages not mapped to aliases, keep metadata only and
        # remove serving aliases when applicable.
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

        # Guard clause (should not happen, but keeps the flow explicit).
        raise ValueError(f"Invalid stage: {stage}")
    
    def promote_to_staging(self, name: str, version: int) -> mlflow.entities.model_registry.ModelVersion:
        """Promote a model version to Staging."""
        return self.transition_model_stage(name, version, ModelStage.STAGING)
    
    def promote_to_production(self, name: str, version: int) -> mlflow.entities.model_registry.ModelVersion:
        """Promote a model version to Production."""
        return self.transition_model_stage(name, version, ModelStage.PRODUCTION)
    
    def archive_model(self, name: str, version: int) -> mlflow.entities.model_registry.ModelVersion:
        """Archive a model version."""
        return self.transition_model_stage(name, version, ModelStage.ARCHIVED)
    
    def get_latest_versions(
        self,
        name: str = DEFAULT_MODEL_NAME,
        stages: Optional[List[str]] = None
    ) -> List[mlflow.entities.model_registry.ModelVersion]:
        """
        Get the latest versions of a model by stage.
        
        Args:
            name: Model name
            stages: List of stages (None = all)
            
        Returns:
            List of ModelVersion objects
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

                    # Fallback for filters not mapped to aliases.
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
        """Get the production version."""
        return self._get_model_version_by_alias(name, ModelAlias.PRODUCTION)
    
    def get_staging_version(self, name: str = DEFAULT_MODEL_NAME) -> Optional[mlflow.entities.model_registry.ModelVersion]:
        """Get the staging version."""
        return self._get_model_version_by_alias(name, ModelAlias.STAGING)
    
    def load_model(
        self,
        name: str = DEFAULT_MODEL_NAME,
        version: Optional[int] = None,
        stage: Optional[str] = None
    ):
        """
        Load a model from the registry.
        
        Args:
            name: Model name
            version: Specific version (takes precedence over stage)
            stage: Model stage (Production, Staging, etc.)
            
        Returns:
            Loaded model object
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
            # Default: Production, otherwise latest version.
            prod_version = self.get_production_version(name)
            if prod_version:
                model_uri = f"models:/{name}@{ModelAlias.PRODUCTION}"
            else:
                model_uri = f"models:/{name}/latest"
        
        logger.info("loading_model", model_uri=model_uri)
        model = mlflow.sklearn.load_model(model_uri)
        
        return model
    
    def load_production_model(self, name: str = DEFAULT_MODEL_NAME):
        """Load the production model."""
        return self.load_model(name, stage=ModelStage.PRODUCTION)
    
    def load_staging_model(self, name: str = DEFAULT_MODEL_NAME):
        """Load the staging model."""
        return self.load_model(name, stage=ModelStage.STAGING)
    
    def list_registered_models(self) -> List[mlflow.entities.model_registry.RegisteredModel]:
        """List all registered models."""
        return list(self.client.search_registered_models())
    
    def get_model_version_details(
        self,
        name: str,
        version: int
    ) -> Dict:
        """
        Get details for a specific version.
        
        Returns:
            Dictionary with version details
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
        Compare two model versions.
        
        Returns:
            Dictionary with metrics comparison
        """
        v1 = self.get_model_version_details(name, version1)
        v2 = self.get_model_version_details(name, version2)
        
        # Fetch runs to retrieve metrics.
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
        """Delete a model version."""
        self.client.delete_model_version(name, str(version))
        logger.info("model_version_deleted", name=name, version=version)
    
    def delete_registered_model(self, name: str) -> None:
        """Delete a registered model (all versions)."""
        self.client.delete_registered_model(name)
        logger.info("registered_model_deleted", name=name)
    
    def search_model_versions(
        self,
        filter_string: str = "",
        max_results: int = 100
    ) -> List[mlflow.entities.model_registry.ModelVersion]:
        """
        Search model versions with a filter.
        
        Args:
            filter_string: MLflow filter string (example: "name='model-name'")
            max_results: Maximum number of results
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
        Build a model URI for loading.
        
        Args:
            name: Model name
            version: Specific version
            stage: Stage (Production, Staging)
            
        Returns:
            URI in the format models:/name/version_or_stage
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
        """Set a tag on a model version."""
        self.client.set_model_version_tag(name, str(version), key, value)
    
    def get_registry_status(self) -> Dict:
        """
        Return Model Registry status.
        
        Returns:
            Dictionary with aggregated statistics
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
