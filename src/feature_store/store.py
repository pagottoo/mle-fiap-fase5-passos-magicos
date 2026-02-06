"""
Feature Store - Interface unificada para gestão de features
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
    Interface unificada para Feature Store.
    
    Centraliza o acesso às features para treinamento e inferência,
    garantindo consistência entre os dois ambientes.
    
    Componentes:
    - Registry: Metadados e definições de features
    - Offline Store: Armazenamento batch para treinamento (Parquet)
    - Online Store: Armazenamento low-latency para inferência (SQLite)
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Inicializa o Feature Store.
        
        Args:
            base_path: Diretório base para armazenamento
        """
        self.base_path = base_path or Path(__file__).parent.parent.parent / "feature_store"
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes
        self.registry = FeatureRegistry(self.base_path / "registry.json")
        self.offline_store = OfflineStore(self.base_path / "offline")
        self.online_store = OnlineStore(self.base_path / "online" / "features.db")
        
        logger.info("feature_store_initialized", path=str(self.base_path))
    
    # ==================== Registry Operations ====================
    
    def register_feature(self, feature: FeatureDefinition) -> None:
        """Registra uma nova feature."""
        self.registry.register_feature(feature)
    
    def register_group(self, group: FeatureGroup) -> None:
        """Registra um novo grupo de features."""
        self.registry.register_group(group)
    
    def get_feature_definition(self, name: str) -> Optional[FeatureDefinition]:
        """Obtém definição de uma feature."""
        return self.registry.get_feature(name)
    
    def get_group_features(self, group_name: str) -> List[FeatureDefinition]:
        """Obtém features de um grupo."""
        return self.registry.get_features_by_group(group_name)
    
    def list_features(self) -> List[str]:
        """Lista todas as features registradas."""
        return self.registry.list_features()
    
    def list_groups(self) -> List[str]:
        """Lista todos os grupos registrados."""
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
        Ingere dados de treinamento no offline store.
        
        Args:
            df: DataFrame com features
            dataset_name: Nome do dataset
            entity_column: Coluna de identificação
            version: Versão do dataset
            
        Returns:
            Caminho do arquivo salvo
        """
        return self.offline_store.ingest(df, dataset_name, entity_column, version=version)
    
    def get_training_data(
        self,
        dataset_name: str,
        version: Optional[str] = None,
        feature_group: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Obtém dados de treinamento do offline store.
        
        Args:
            dataset_name: Nome do dataset
            version: Versão do dataset
            feature_group: Grupo de features a selecionar
            
        Returns:
            DataFrame com os dados
        """
        columns = None
        if feature_group:
            group = self.registry.get_group(feature_group)
            if group:
                # Mapear nomes de features para colunas de origem
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
        Obtém features e target para treinamento.
        
        Args:
            dataset_name: Nome do dataset
            feature_group: Grupo de features
            target_column: Coluna target
            version: Versão do dataset
            
        Returns:
            Tupla (X, y) com features e target
        """
        # Obter colunas do grupo
        group = self.registry.get_group(feature_group)
        if not group:
            raise ValueError(f"Grupo '{feature_group}' não encontrado")
        
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
        Materializa features no online store para serving.
        
        Args:
            df: DataFrame com features
            table_name: Nome da tabela
            entity_column: Coluna de identificação
            feature_group: Grupo de features a materializar
            
        Returns:
            Número de registros materializados
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
        Obtém features para inferência do online store.
        
        Args:
            table_name: Nome da tabela
            entity_ids: IDs das entidades
            feature_group: Grupo de features
            
        Returns:
            DataFrame com features
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
        Obtém vetor de features para uma entidade.
        
        Args:
            table_name: Nome da tabela
            entity_id: ID da entidade
            
        Returns:
            Dicionário com features
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
        Sincroniza dados do offline store para o online store.
        
        Args:
            dataset_name: Nome do dataset no offline store
            table_name: Nome da tabela no online store
            entity_column: Coluna de identificação
            version: Versão do dataset
            
        Returns:
            Número de registros sincronizados
        """
        df = self.offline_store.get_dataset(dataset_name, version=version)
        return self.online_store.materialize(df, table_name, entity_column)
    
    # ==================== Utility Operations ====================
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtém status geral do Feature Store.
        
        Returns:
            Dicionário com status
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
        Valida consistência entre offline e online stores.
        
        Args:
            offline_dataset: Nome do dataset offline
            online_table: Nome da tabela online
            entity_column: Coluna de identificação
            sample_size: Tamanho da amostra para validação
            
        Returns:
            Relatório de validação
        """
        # Carregar amostra do offline
        df_offline = self.offline_store.get_dataset(offline_dataset)
        
        if len(df_offline) > sample_size:
            df_offline = df_offline.sample(sample_size, random_state=42)
        
        entity_ids = df_offline[entity_column].tolist()
        
        # Carregar do online
        df_online = self.online_store.get_features(online_table, entity_ids)
        
        # Comparar
        common_columns = list(set(df_offline.columns) & set(df_online.columns))
        common_columns = [c for c in common_columns if c != "_materialized_at"]
        
        mismatches = []
        for col in common_columns:
            offline_vals = df_offline.set_index(entity_column)[col]
            online_vals = df_online.set_index(entity_column)[col]
            
            # Alinhar índices
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
