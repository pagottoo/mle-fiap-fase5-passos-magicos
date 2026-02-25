"""
Feature Store - unified interface for feature management.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import structlog

from .registry import FeatureRegistry, FeatureDefinition, FeatureGroup
from .offline_store import OfflineStore
from .online_store import OnlineStore

logger = structlog.get_logger()


class FeatureStore:
    """
    Unified Feature Store interface.

    Centralizes feature access for training and inference,
    ensuring consistency across both environments.

    Components:
    - Registry: feature metadata and definitions
    - Offline Store: batch storage for training (Parquet)
    - Online Store: low-latency storage for inference (SQLite)
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize Feature Store.

        Args:
            base_path: Base storage directory.
        """
        self.base_path = base_path or Path(__file__).parent.parent.parent / "feature_store"
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.registry = FeatureRegistry(self.base_path / "registry.json")
        self.offline_store = OfflineStore(self.base_path / "offline")
        self.online_store = OnlineStore(self.base_path / "online" / "features.db")
        
        logger.info("feature_store_initialized", path=str(self.base_path))
    
    # ==================== Registry Operations ====================
    
    def register_feature(self, feature: FeatureDefinition) -> None:
        """Register a new feature definition."""
        self.registry.register_feature(feature)
    
    def register_group(self, group: FeatureGroup) -> None:
        """Register a new feature group."""
        self.registry.register_group(group)
    
    def get_feature_definition(self, name: str) -> Optional[FeatureDefinition]:
        """Return feature definition by name."""
        return self.registry.get_feature(name)
    
    def get_group_features(self, group_name: str) -> List[FeatureDefinition]:
        """Return features that belong to a group."""
        return self.registry.get_features_by_group(group_name)
    
    def list_features(self) -> List[str]:
        """List all registered feature names."""
        return self.registry.list_features()
    
    def list_groups(self) -> List[str]:
        """List all registered group names."""
        return self.registry.list_groups()
    
    # ==================== Offline Store Operations ====================
    
    def ingest_training_data(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        entity_column: str = "aluno_id",
        version: Optional[str] = None
    ) -> str:
        """
        Ingest training data into offline store.

        Args:
            df: Feature DataFrame.
            dataset_name: Dataset name.
            entity_column: Entity identifier column.
            version: Dataset version.

        Returns:
            Saved file path.
        """
        return self.offline_store.ingest(df, dataset_name, entity_column, version=version)
    
    def get_training_data(
        self,
        dataset_name: str,
        version: Optional[str] = None,
        feature_group: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve training data from offline store.

        Args:
            dataset_name: Dataset name.
            version: Dataset version.
            feature_group: Optional feature group to select.

        Returns:
            DataFrame with training data.
        """
        columns = None
        if feature_group:
            group = self.registry.get_group(feature_group)
            if group:
                # Map feature names to source columns
                columns = []
                for feat_name in group.features:
                    feat = self.registry.get_feature(feat_name)
                    if feat:
                        columns.append(feat.source)
        
        return self.offline_store.get_dataset(dataset_name, version=version, columns=columns)
    
    def get_features_for_training(
        self,
        dataset_name: str,
        feature_group: str = "ponto_virada_features",
        target_column: str = "ponto_virada_pred",
        version: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Return features and target for training.

        Args:
            dataset_name: Dataset name.
            feature_group: Feature group name.
            target_column: Target column.
            version: Dataset version.

        Returns:
            Tuple `(X, y)` with features and target.
        """
        # Resolve group columns
        group = self.registry.get_group(feature_group)
        if not group:
            raise ValueError(f"Group '{feature_group}' not found")
        
        feature_columns = []
        for feat_name in group.features:
            feat = self.registry.get_feature(feat_name)
            if feat:
                feature_columns.append(feat.source)
        
        return self.offline_store.get_features_for_training(
            dataset_name, feature_columns, target_column, version
        )
    
    # ==================== Online Store Operations ====================
    
    def materialize_for_serving(
        self,
        df: pd.DataFrame,
        table_name: str,
        entity_column: str,
        feature_group: Optional[str] = None
    ) -> int:
        """
        Materialize features in online store for serving.

        Args:
            df: Feature DataFrame.
            table_name: Target table name.
            entity_column: Entity identifier column.
            feature_group: Optional feature group to materialize.

        Returns:
            Number of rows materialized.
        """
        feature_columns = None
        if feature_group:
            group = self.registry.get_group(feature_group)
            if group:
                feature_columns = []
                for feat_name in group.features:
                    feat = self.registry.get_feature(feat_name)
                    if feat and feat.source in df.columns:
                        feature_columns.append(feat.source)
        
        return self.online_store.materialize(df, table_name, entity_column, feature_columns)
    
    def get_serving_features(
        self,
        table_name: str,
        entity_ids: List[Any],
        feature_group: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve serving features from online store.

        Args:
            table_name: Source table name.
            entity_ids: Entity IDs.
            feature_group: Optional feature group.

        Returns:
            Feature DataFrame.
        """
        feature_columns = None
        if feature_group:
            group = self.registry.get_group(feature_group)
            if group:
                feature_columns = []
                for feat_name in group.features:
                    feat = self.registry.get_feature(feat_name)
                    if feat:
                        feature_columns.append(feat.source)
        
        return self.online_store.get_features(table_name, entity_ids, feature_columns)
    
    def get_feature_vector(
        self,
        table_name: str,
        entity_id: Any
    ) -> Dict[str, Any]:
        """
        Return feature vector for a single entity.

        Args:
            table_name: Source table name.
            entity_id: Entity ID.

        Returns:
            Feature dictionary.
        """
        return self.online_store.get_feature_vector(table_name, entity_id)
    
    # ==================== Sync Operations ====================
    
    def sync_offline_to_online(
        self,
        dataset_name: str,
        table_name: str,
        entity_column: str,
        version: Optional[str] = None
    ) -> int:
        """
        Sync data from offline store to online store.

        Args:
            dataset_name: Dataset name in offline store.
            table_name: Target table in online store.
            entity_column: Entity identifier column.
            version: Dataset version.

        Returns:
            Number of synced rows.
        """
        df = self.offline_store.get_dataset(dataset_name, version=version)
        return self.online_store.materialize(df, table_name, entity_column)
    
    # ==================== Utility Operations ====================
    
    def get_status(self) -> Dict[str, Any]:
        """
        Return high-level Feature Store status.

        Returns:
            Status dictionary.
        """
        return {
            "registry": {
                "features": len(self.registry.list_features()),
                "groups": len(self.registry.list_groups())
            },
            "offline_store": {
                "datasets": self.offline_store.list_datasets()
            },
            "online_store": {
                "tables": [t["table_name"] for t in self.online_store.list_tables()]
            },
            "base_path": str(self.base_path)
        }
    
    def validate_feature_consistency(
        self,
        offline_dataset: str,
        online_table: str,
        entity_column: str,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Validate consistency between offline and online stores.

        Args:
            offline_dataset: Offline dataset name.
            online_table: Online table name.
            entity_column: Entity identifier column.
            sample_size: Validation sample size.

        Returns:
            Validation report dictionary.
        """
        # Load sample from offline store
        df_offline = self.offline_store.get_dataset(offline_dataset)
        
        if len(df_offline) > sample_size:
            df_offline = df_offline.sample(sample_size, random_state=42)
        
        entity_ids = df_offline[entity_column].tolist()
        
        # Load from online store
        df_online = self.online_store.get_features(online_table, entity_ids)
        
        # Compare rows
        common_columns = list(set(df_offline.columns) & set(df_online.columns))
        common_columns = [c for c in common_columns if c != "_materialized_at"]
        
        mismatches = []
        for col in common_columns:
            offline_vals = df_offline.set_index(entity_column)[col]
            online_vals = df_online.set_index(entity_column)[col]
            
            # Align indexes
            common_idx = offline_vals.index.intersection(online_vals.index)
            
            if len(common_idx) > 0:
                diff = (offline_vals.loc[common_idx] != online_vals.loc[common_idx]).sum()
                if diff > 0:
                    mismatches.append({"column": col, "mismatches": int(diff)})
        
        return {
            "offline_dataset": offline_dataset,
            "online_table": online_table,
            "sample_size": len(entity_ids),
            "common_columns": len(common_columns),
            "is_consistent": len(mismatches) == 0,
            "mismatches": mismatches
        }
