"""
Online Store - Armazenamento para inferência (low latency)
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import structlog

logger = structlog.get_logger()


class OnlineStore:
    """
    Store online para features de inferência.
    
    Usa SQLite para armazenamento rápido e baixa latência.
    Ideal para serving de features em tempo real.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Inicializa o online store.
        
        Args:
            db_path: Caminho para o banco de dados SQLite
        """
        if db_path:
            self.db_path = db_path
            # Garantir que o diretório existe
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            store_dir = Path(__file__).parent.parent.parent / "feature_store" / "online"
            store_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = store_dir / "features.db"
        
        self._init_db()
    
    def _init_db(self) -> None:
        """Inicializa o banco de dados."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Tabela de metadados
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_tables (
                table_name TEXT PRIMARY KEY,
                entity_column TEXT,
                feature_columns TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Tabela de log de acesso
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT,
                entity_id TEXT,
                accessed_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Obtém conexão com o banco."""
        return sqlite3.connect(str(self.db_path))
    
    def materialize(
        self,
        df: pd.DataFrame,
        table_name: str,
        entity_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> int:
        """
        Materializa features no online store.
        
        Args:
            df: DataFrame com features
            table_name: Nome da tabela
            entity_column: Coluna de identificação da entidade
            feature_columns: Colunas de features (todas exceto entity se não fornecidas)
            
        Returns:
            Número de registros materializados
        """
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != entity_column]
        
        # Selecionar colunas
        columns = [entity_column] + feature_columns
        df_to_store = df[columns].copy()
        
        # Adicionar timestamp de materialização
        df_to_store["_materialized_at"] = datetime.now().isoformat()
        
        conn = self._get_connection()
        
        # Criar/substituir tabela
        df_to_store.to_sql(table_name, conn, if_exists="replace", index=False)
        
        # Criar índice na coluna de entidade
        cursor = conn.cursor()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{entity_column} ON {table_name}({entity_column})")
        
        # Atualizar metadados
        cursor.execute("""
            INSERT OR REPLACE INTO feature_tables (table_name, entity_column, feature_columns, created_at, updated_at)
            VALUES (?, ?, ?, COALESCE((SELECT created_at FROM feature_tables WHERE table_name = ?), ?), ?)
        """, (
            table_name,
            entity_column,
            json.dumps(feature_columns),
            table_name,
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(
            "features_materialized",
            table=table_name,
            rows=len(df_to_store),
            features=len(feature_columns)
        )
        
        return len(df_to_store)
    
    def get_features(
        self,
        table_name: str,
        entity_ids: List[Any],
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Obtém features para entidades específicas.
        
        Args:
            table_name: Nome da tabela
            entity_ids: IDs das entidades
            feature_columns: Colunas a retornar (todas se não fornecidas)
            
        Returns:
            DataFrame com features
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Obter metadados
        cursor.execute("SELECT entity_column FROM feature_tables WHERE table_name = ?", (table_name,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise ValueError(f"Tabela '{table_name}' não encontrada")
        
        entity_column = result[0]
        
        # Construir query
        if feature_columns:
            columns = [entity_column] + feature_columns
            columns_str = ", ".join(columns)
        else:
            columns_str = "*"
        
        placeholders = ", ".join(["?" for _ in entity_ids])
        query = f"SELECT {columns_str} FROM {table_name} WHERE {entity_column} IN ({placeholders})"
        
        df = pd.read_sql_query(query, conn, params=entity_ids)
        
        # Log de acesso
        for entity_id in entity_ids:
            cursor.execute(
                "INSERT INTO access_log (table_name, entity_id, accessed_at) VALUES (?, ?, ?)",
                (table_name, str(entity_id), datetime.now().isoformat())
            )
        
        conn.commit()
        conn.close()
        
        logger.debug(
            "features_retrieved",
            table=table_name,
            entities=len(entity_ids),
            rows=len(df)
        )
        
        return df
    
    def get_feature_vector(
        self,
        table_name: str,
        entity_id: Any,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Obtém vetor de features para uma única entidade.
        
        Args:
            table_name: Nome da tabela
            entity_id: ID da entidade
            feature_columns: Colunas a retornar
            
        Returns:
            Dicionário com features
        """
        df = self.get_features(table_name, [entity_id], feature_columns)
        
        if df.empty:
            return {}
        
        return df.iloc[0].to_dict()
    
    def list_tables(self) -> List[Dict[str, Any]]:
        """Lista tabelas disponíveis."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT table_name, entity_column, feature_columns, created_at, updated_at FROM feature_tables")
        rows = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                "table_name": row[0],
                "entity_column": row[1],
                "feature_columns": json.loads(row[2]),
                "created_at": row[3],
                "updated_at": row[4]
            }
            for row in rows
        ]
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Obtém informações de uma tabela."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT table_name, entity_column, feature_columns, created_at, updated_at FROM feature_tables WHERE table_name = ?",
            (table_name,)
        )
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Contar registros
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "table_name": row[0],
            "entity_column": row[1],
            "feature_columns": json.loads(row[2]),
            "num_rows": count,
            "created_at": row[3],
            "updated_at": row[4]
        }
    
    def delete_table(self, table_name: str) -> bool:
        """
        Remove uma tabela.
        
        Args:
            table_name: Nome da tabela
            
        Returns:
            True se removida com sucesso
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            cursor.execute("DELETE FROM feature_tables WHERE table_name = ?", (table_name,))
            conn.commit()
            logger.info("table_deleted", table=table_name)
            return True
        except Exception as e:
            logger.error("table_delete_failed", table=table_name, error=str(e))
            return False
        finally:
            conn.close()
    
    def get_access_stats(self, table_name: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        """
        Obtém estatísticas de acesso.
        
        Args:
            table_name: Filtrar por tabela (opcional)
            days: Número de dias para análise
            
        Returns:
            Estatísticas de acesso
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT 
                table_name,
                COUNT(*) as access_count,
                COUNT(DISTINCT entity_id) as unique_entities
            FROM access_log
            WHERE datetime(accessed_at) >= datetime('now', ?)
        """
        params = [f"-{days} days"]
        
        if table_name:
            query += " AND table_name = ?"
            params.append(table_name)
        
        query += " GROUP BY table_name"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        return {
            "period_days": days,
            "tables": [
                {
                    "table_name": row[0],
                    "access_count": row[1],
                    "unique_entities": row[2]
                }
                for row in rows
            ]
        }
