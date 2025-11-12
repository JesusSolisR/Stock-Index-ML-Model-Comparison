"""
This module provides functions to load and preprocess stock index data from CSV files.
It includes a DataCleaner class for modular data cleaning operations.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


class DataCleaner:
    """Lightweight data cleaning helper exposing small, specific methods."""
    
    # Renaming columns and stock index values
    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Column mappings
        column_mapping = {
            'Index': 'stock_index',
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        # Stock index mappings
        stock_index_mapping = {
            'NYA': 'New York',
            'IXIC': 'NASDAQ',
            'HSI': 'Hong Kong',
            '000001.SS': 'Shanghai',
            'N225': 'Tokyo',
            'N100': 'Euronext',
            '399001.SZ': 'Shenzhen',
            'GSPTSE': 'Toronto',
            'NSEI': 'India',
            'GDAXI': 'Frankfurt',
            'KS11': 'Korea',
            'SSMI': 'Switzerland',
            'TWII': 'Taiwan',
            'J203.JO': 'Johannesburg'
        }
        
        # Rename columns and stock index values in one go
        df = df.rename(columns=column_mapping).replace({'stock_index': stock_index_mapping})

        return df

    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        return df

    def filter_by_dates(self, df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        return df

    def basic_data_preprocessing(self, df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        df = self.rename_columns(df)
        df = self.parse_dates(df)
        df = self.filter_by_dates(df, start_date, end_date)
        return df
    
    def set_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('date')
        df.sort_index(inplace=True)
        return df
    
    def drop_rows_with_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        core_columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=core_columns)
        return df
    
    def drop_rows_with_zero_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['volume'] > 0]
        return df

def load_raw_data(stock_index=None, start_date=None, end_date=None, file_path='../data/indexData.csv'):
    """Load raw data from a given file path."""
    
    print(f"Loading raw data from {file_path}...")
    
    # read CSV
    path = Path(file_path)
    # Error handling for file not found
    if not path.exists():
        raise FileNotFoundError(f"The file not found: {path}")
    # Read file to DataFrame
    df_raw = pd.read_csv(path)

    # filter for a specific stock index if provided
    if stock_index:
        df_raw = df_raw[df_raw['Index'] == stock_index]

    # Delegate cleaning to DataCleaner
    cleaner = DataCleaner()
    df = cleaner.basic_data_preprocessing(df_raw, start_date, end_date)
    return df

def load_data(stock_index=None, start_date=None, end_date=None, file_path='../data/indexData.csv'):
    """Load data from a given file path."""
    print(f"Loading data from {file_path}...")
    
    # read CSV
    path = Path(file_path)
    # Error handling for file not found
    if not path.exists():
        raise FileNotFoundError(f"The file not found: {path}")
    # Read file to DataFrame
    df = pd.read_csv(path)
    
    # Filter for specific stock index if provided
    if stock_index:
        df = df[df['Index'] == stock_index]
    
    # Delegate cleaning to DataCleaner
    cleaner = DataCleaner()
    df = cleaner.basic_data_preprocessing(df, start_date, end_date)
    
    # Sort and set index via DataCleaner
    df = cleaner.set_and_sort(df)
    
    # Drop rows with missing core data via DataCleaner
    df = cleaner.drop_rows_with_missing_data(df)
    
    # Drop rows with zero volume
    df = cleaner.drop_rows_with_zero_volume(df)
    
    return df
