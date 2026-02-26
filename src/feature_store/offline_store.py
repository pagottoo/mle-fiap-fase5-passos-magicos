"""
Offline Store - batch storage for training datasets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import structlog

logger = structlog.get_logger()


class OfflineStore:
    """
    Offline store for training features.

    Uses Parquet files for efficient historical storage.
    Suitable for batch processing and model training.
    """
    
    def __init__(self, store_path: Optional[Path] = None):
        """
        Initialize offline store.

        Args:
            store_path: Directory where datasets are stored.
        """
        self.store_path = store_path or Path(__file__).parent.parent.parent / "feature_store" / "offline"
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_path = self.store_path / "metadata.json"
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata file."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {"datasets": {}, "created_at": datetime.now().isoformat()}
    
    def _save_metadata(self) -> None:
        """Persist metadata file."""
        self._metadata["updated_at"] = datetime.now().isoformat()
        with open(self.metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)
    
    def ingest(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        entity_column: str = "aluno_id",
        timestamp_column: Optional[str] = None,
        version: Optional[str] = None
    ) -> str:
        """
        Ingest data into offline store.

        Args:
            df: Feature DataFrame.
            dataset_name: Dataset name.
            entity_column: Entity identifier column.
            timestamp_column: Optional timestamp column.
            version: Dataset version (auto-generated when omitted).

        Returns:
            Saved file path.
        """
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create dataset directory
        dataset_dir = self.store_path / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Persist parquet file
        file_path = dataset_dir / f"{dataset_name}_{version}.parquet"
        df.to_parquet(file_path, index=False)
        
        # Update metadata
        if dataset_name not in self._metadata["datasets"]:
            self._metadata["datasets"][dataset_name] = {
                "versions": [],
                "entity_column": entity_column,
                "timestamp_column": timestamp_column,
                "created_at": datetime.now().isoformat()
            }
        
        self._metadata["datasets"][dataset_name]["versions"].append({
            "version": version,
            "file_path": str(file_path),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "created_at": datetime.now().isoformat()
        })
        
        self._save_metadata()
        
        logger.info(
            "data_ingested",
            dataset=dataset_name,
            version=version,
            rows=len(df),
            columns=len(df.columns)
        )
        
        return str(file_path)
    
    def get_dataset(
        self,
        dataset_name: str,
        version: Optional[str] = None,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Retrieve dataset from offline store.

        Args:
            dataset_name: Dataset name.
            version: Specific version (latest by default).
            columns: Optional selected columns.
            filters: Optional filter mapping.

        Returns:
            DataFrame with dataset rows.
        """
        if dataset_name not in self._metadata["datasets"]:
            raise ValueError(f"Dataset '{dataset_name}' não encontrado")
        
        dataset_meta = self._metadata["datasets"][dataset_name]
        
        # Select version
        if version:
            version_meta = next(
                (v for v in dataset_meta["versions"] if v["version"] == version),
                None
            )
            if not version_meta:
                raise ValueError(f"Versão '{version}' não encontrada")
        else:
            # Latest version
            version_meta = dataset_meta["versions"][-1]
        
        # Load dataset
        file_path = Path(version_meta["file_path"])
        df = pd.read_parquet(file_path, columns=columns)
        
        # Apply filters
        if filters:
            for col, value in filters.items():
                if col in df.columns:
                    if isinstance(value, list):
                        df = df[df[col].isin(value)]
                    else:
                        df = df[df[col] == value]
        
        logger.info(
            "dataset_retrieved",
            dataset=dataset_name,
            version=version_meta["version"],
            rows=len(df)
        )
        
        return df
    
    def get_features_for_training(
        self,
        dataset_name: str,
        feature_columns: List[str],
        target_column: str,
        version: Optional[str] = None
    ) -> tuple:
        """
        Return features and target for training.

        Args:
            dataset_name: Dataset name.
            feature_columns: Feature columns.
            target_column: Target column.
            version: Dataset version.

        Returns:
            Tuple `(X, y)` with features and target.
        """
        columns = feature_columns + [target_column]
        df = self.get_dataset(dataset_name, version=version, columns=columns)
        
        X = df[feature_columns]
        y = df[target_column]
        
        return X, y
    
    def list_datasets(self) -> List[str]:
        """List available datasets."""
        return list(self._metadata["datasets"].keys())
    
    def list_versions(self, dataset_name: str) -> List[Dict[str, Any]]:
        """List versions of a dataset."""
        if dataset_name not in self._metadata["datasets"]:
            return []
        return self._metadata["datasets"][dataset_name]["versions"]
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for one dataset."""
        return self._metadata["datasets"].get(dataset_name)
    
    def delete_version(self, dataset_name: str, version: str) -> bool:
        """
        Delete one dataset version.

        Args:
            dataset_name: Dataset name.
            version: Version to delete.

        Returns:
            True when deleted successfully.
        """
        if dataset_name not in self._metadata["datasets"]:
            return False
        
        versions = self._metadata["datasets"][dataset_name]["versions"]
        version_meta = next((v for v in versions if v["version"] == version), None)
        
        if not version_meta:
            return False
        
        # Remove parquet file
        file_path = Path(version_meta["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # Update metadata
        self._metadata["datasets"][dataset_name]["versions"] = [
            v for v in versions if v["version"] != version
        ]
        self._save_metadata()
        
        logger.info("version_deleted", dataset=dataset_name, version=version)
        
        return True
    
    def compute_statistics(self, dataset_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute descriptive statistics for a dataset.

        Args:
            dataset_name: Dataset name.
            version: Dataset version.

        Returns:
            Statistics dictionary.
        """
        df = self.get_dataset(dataset_name, version=version)
        
        stats = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": {}
        }
        
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_pct": round(df[col].isnull().mean() * 100, 2)
            }
            
            if np.issubdtype(df[col].dtype, np.number):
                col_stats.update({
                    "mean": round(df[col].mean(), 4) if not df[col].isnull().all() else None,
                    "std": round(df[col].std(), 4) if not df[col].isnull().all() else None,
                    "min": round(df[col].min(), 4) if not df[col].isnull().all() else None,
                    "max": round(df[col].max(), 4) if not df[col].isnull().all() else None,
                })
            else:
                col_stats["unique_count"] = int(df[col].nunique())
            
            stats["columns"][col] = col_stats
        
        return stats
