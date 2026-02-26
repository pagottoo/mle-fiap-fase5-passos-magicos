"""
Online Store - low-latency storage for inference.
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
    Online store for serving features.

    Uses SQLite for fast low-latency reads.
    Suitable for real-time inference workloads.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize online store.

        Args:
            db_path: SQLite database path.
        """
        if db_path:
            self.db_path = db_path
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            store_dir = Path(__file__).parent.parent.parent / "feature_store" / "online"
            store_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = store_dir / "features.db"
        
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize internal SQLite tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_tables (
                table_name TEXT PRIMARY KEY,
                entity_column TEXT,
                feature_columns TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Access log table
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
        """Return a SQLite connection."""
        return sqlite3.connect(str(self.db_path))
    
    def materialize(
        self,
        df: pd.DataFrame,
        table_name: str,
        entity_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> int:
        """
        Materialize features into online store.

        Args:
            df: Feature DataFrame.
            table_name: Target table name.
            entity_column: Entity identifier column.
            feature_columns: Feature columns (all except entity when omitted).

        Returns:
            Number of rows materialized.
        """
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != entity_column]
        
        # Select columns
        columns = [entity_column] + feature_columns
        df_to_store = df[columns].copy()
        
        # Add materialization timestamp
        df_to_store["_materialized_at"] = datetime.now().isoformat()
        
        conn = self._get_connection()
        
        # Create/replace table
        df_to_store.to_sql(table_name, conn, if_exists="replace", index=False)
        
        # Create index on entity column
        cursor = conn.cursor()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{entity_column} ON {table_name}({entity_column})")
        
        # Update metadata
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
        Fetch features for specific entities.

        Args:
            table_name: Source table.
            entity_ids: Entity IDs.
            feature_columns: Columns to return (all by default).

        Returns:
            Feature DataFrame.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Read metadata
        cursor.execute("SELECT entity_column FROM feature_tables WHERE table_name = ?", (table_name,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise ValueError(f"Tabela '{table_name}' não encontrada")
        
        entity_column = result[0]
        
        # Build SQL query
        if feature_columns:
            columns = [entity_column] + feature_columns
            columns_str = ", ".join(columns)
        else:
            columns_str = "*"
        
        placeholders = ", ".join(["?" for _ in entity_ids])
        query = f"SELECT {columns_str} FROM {table_name} WHERE {entity_column} IN ({placeholders})"
        
        df = pd.read_sql_query(query, conn, params=entity_ids)
        
        # Access log
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
        Fetch feature vector for a single entity.

        Args:
            table_name: Source table.
            entity_id: Entity ID.
            feature_columns: Columns to return.

        Returns:
            Feature dictionary.
        """
        df = self.get_features(table_name, [entity_id], feature_columns)
        
        if df.empty:
            return {}
        
        return df.iloc[0].to_dict()
    
    def list_tables(self) -> List[Dict[str, Any]]:
        """List available tables."""
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
        """Return metadata for a table."""
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
        
        # Count rows
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
        Drop table and delete its metadata.

        Args:
            table_name: Table name.

        Returns:
            True when removed successfully.
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
        Return access statistics.

        Args:
            table_name: Optional table filter.
            days: Number of days to analyze.

        Returns:
            Access statistics dictionary.
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
