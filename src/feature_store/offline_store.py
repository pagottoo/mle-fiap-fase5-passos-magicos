"""
Offline Store - Armazenamento para treinamento (batch)
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
    Store offline para features de treinamento.
    
    Usa arquivos Parquet para armazenamento eficiente de dados históricos.
    Ideal para batch processing e treinamento de modelos.
    """
    
    def __init__(self, store_path: Optional[Path] = None):
        """
        Inicializa o offline store.
        
        Args:
            store_path: Diretório para armazenamento dos dados
        """
        self.store_path = store_path or Path(__file__).parent.parent.parent / "feature_store" / "offline"
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_path = self.store_path / "metadata.json"
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Carrega metadados do store."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {"datasets": {}, "created_at": datetime.now().isoformat()}
    
    def _save_metadata(self) -> None:
        """Salva metadados do store."""
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
        Ingere dados no offline store.
        
        Args:
            df: DataFrame com features
            dataset_name: Nome do dataset
            entity_column: Coluna de identificação da entidade
            timestamp_column: Coluna de timestamp (opcional)
            version: Versão do dataset (auto-gerada se não fornecida)
            
        Returns:
            Caminho do arquivo salvo
        """
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Criar diretório do dataset
        dataset_dir = self.store_path / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar em Parquet
        file_path = dataset_dir / f"{dataset_name}_{version}.parquet"
        df.to_parquet(file_path, index=False)
        
        # Atualizar metadados
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
        Obtém dataset do offline store.
        
        Args:
            dataset_name: Nome do dataset
            version: Versão específica (última se não fornecida)
            columns: Colunas a selecionar (todas se não fornecidas)
            filters: Filtros a aplicar
            
        Returns:
            DataFrame com os dados
        """
        if dataset_name not in self._metadata["datasets"]:
            raise ValueError(f"Dataset '{dataset_name}' não encontrado")
        
        dataset_meta = self._metadata["datasets"][dataset_name]
        
        # Selecionar versão
        if version:
            version_meta = next(
                (v for v in dataset_meta["versions"] if v["version"] == version),
                None
            )
            if not version_meta:
                raise ValueError(f"Versão '{version}' não encontrada")
        else:
            # Última versão
            version_meta = dataset_meta["versions"][-1]
        
        # Carregar dados
        file_path = Path(version_meta["file_path"])
        df = pd.read_parquet(file_path, columns=columns)
        
        # Aplicar filtros
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
        Obtém features e target para treinamento.
        
        Args:
            dataset_name: Nome do dataset
            feature_columns: Colunas de features
            target_column: Coluna target
            version: Versão do dataset
            
        Returns:
            Tupla (X, y) com features e target
        """
        columns = feature_columns + [target_column]
        df = self.get_dataset(dataset_name, version=version, columns=columns)
        
        X = df[feature_columns]
        y = df[target_column]
        
        return X, y
    
    def list_datasets(self) -> List[str]:
        """Lista datasets disponíveis."""
        return list(self._metadata["datasets"].keys())
    
    def list_versions(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Lista versões de um dataset."""
        if dataset_name not in self._metadata["datasets"]:
            return []
        return self._metadata["datasets"][dataset_name]["versions"]
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Obtém informações de um dataset."""
        return self._metadata["datasets"].get(dataset_name)
    
    def delete_version(self, dataset_name: str, version: str) -> bool:
        """
        Remove uma versão de dataset.
        
        Args:
            dataset_name: Nome do dataset
            version: Versão a remover
            
        Returns:
            True se removido com sucesso
        """
        if dataset_name not in self._metadata["datasets"]:
            return False
        
        versions = self._metadata["datasets"][dataset_name]["versions"]
        version_meta = next((v for v in versions if v["version"] == version), None)
        
        if not version_meta:
            return False
        
        # Remover arquivo
        file_path = Path(version_meta["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # Atualizar metadados
        self._metadata["datasets"][dataset_name]["versions"] = [
            v for v in versions if v["version"] != version
        ]
        self._save_metadata()
        
        logger.info("version_deleted", dataset=dataset_name, version=version)
        
        return True
    
    def compute_statistics(self, dataset_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Computa estatísticas de um dataset.
        
        Args:
            dataset_name: Nome do dataset
            version: Versão do dataset
            
        Returns:
            Dicionário com estatísticas
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
