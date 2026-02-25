"""
Data preprocessing module.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger()


class DataPreprocessor:
    """
    Handles preprocessing for Passos Magicos datasets.
    """
    
    def __init__(self):
        self.columns_to_drop = ["NOME"]  # Identifier columns
        self._fitted = False
        self._numeric_columns = []
        self._categorical_columns = []
    
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load data from CSV or Excel file.

        Args:
            file_path: Input data file path.

        Returns:
            Loaded DataFrame.
        """
        logger.info("loading_data", file_path=str(file_path))
        
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path, delimiter=";")
        elif file_path.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info("data_loaded", rows=len(df), columns=len(df.columns))
        return df
    
    def filter_columns_by_year(self, df: pd.DataFrame, year: str) -> pd.DataFrame:
        """
        Filter columns for a specific year and strip year suffix.

        Args:
            df: Original DataFrame.
            year: Year suffix to keep (e.g. `"2022"`).

        Returns:
            Filtered DataFrame with normalized column names.
        """
        logger.info("filtering_columns_by_year", year=year)
        
        # Keep selected-year columns or non-year columns
        year_suffix = f"_{year}"
        other_years = ["_2020", "_2021", "_2022", "_2023", "_2024"]
        other_years = [y for y in other_years if y != year_suffix]
        
        columns_to_keep = []
        for col in df.columns:
            # Keep if it has selected year suffix or no other year suffix
            has_year_suffix = any(y in col for y in other_years)
            if not has_year_suffix:
                columns_to_keep.append(col)
        
        df_filtered = df[columns_to_keep].copy()
        
        # Strip selected year suffix from column names
        df_filtered.columns = [
            col.replace(year_suffix, "") for col in df_filtered.columns
        ]
        
        logger.info("columns_filtered", columns=len(df_filtered.columns))
        return df_filtered
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset by removing low-quality rows.

        Args:
            df: DataFrame to clean.

        Returns:
            Cleaned DataFrame.
        """
        logger.info("cleaning_data", initial_rows=len(df))
        
        # Drop identifier columns
        cols_to_drop = [c for c in self.columns_to_drop if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        
        # Drop rows where all numeric columns are null
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.dropna(subset=numeric_cols, how="all")
        
        # Drop rows with more than 50% missing values
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)
        
        logger.info("data_cleaned", final_rows=len(df))
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.

        Args:
            df: DataFrame with missing values.

        Returns:
            DataFrame with imputed values.
        """
        logger.info("handling_missing_values")
        
        df = df.copy()
        
        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        
        # Categorical columns: fill with "Desconhecido"
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna("Desconhecido")
        
        logger.info("missing_values_handled")
        return df
    
    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit preprocessor and persist schema statistics.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        self._numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
        self._fitted = True
        logger.info("preprocessor_fitted", 
                   numeric_cols=len(self._numeric_columns),
                   categorical_cols=len(self._categorical_columns))
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transforms.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """
        if not self._fitted:
            raise ValueError("DataPreprocessor must be fitted before transform.")
        
        df = self.clean_data(df)
        df = self.handle_missing_values(df)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one operation.

        Args:
            df: DataFrame to process.

        Returns:
            Processed DataFrame.
        """
        self.fit(df)
        return self.transform(df)
    
    def prepare_dataset(
        self, 
        file_path: Path, 
        year: str = "2022"
    ) -> pd.DataFrame:
        """
        Full dataset preparation pipeline.

        Args:
            file_path: Source data file.
            year: Year suffix to process.

        Returns:
            Prepared DataFrame.
        """
        logger.info("preparing_dataset", file_path=str(file_path), year=year)
        
        df = self.load_data(file_path)
        df = self.filter_columns_by_year(df, year)
        df = self.fit_transform(df)
        
        logger.info("dataset_prepared", rows=len(df), columns=len(df.columns))
        return df
