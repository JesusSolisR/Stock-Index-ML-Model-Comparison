"""Time-series aware train/test splitting for temporal data.

This module provides a TimeSeriesSplit class for splitting time-indexed data
chronologically without data leakage. Perfect for stock price and financial
time-series data where temporal order must be preserved.
"""

from typing import Tuple
import pandas as pd


class TimeSeriesSplit:
    """Chronological train/test split for time-series data.
    
    Splits data at a cutoff point, ensuring training data comes before
    test data temporally. No random shuffling, no data leakage.
    
    Params:
        test_size (float): proportion of data for testing (0.0 to 1.0).
                          Default: 0.2 (80/20 split)
    """
    
    def __init__(self, test_size: float = 0.2):
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be in (0.0, 1.0), got {test_size}")
        self.test_size = test_size
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate dataframe is suitable for time-series splitting."""
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex (temporal)")
        if len(df) < 2:
            raise ValueError("DataFrame must have at least 2 rows for train/test split")
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data chronologically into train and test sets.
        
        Args:
            df: DataFrame with DatetimeIndex, sorted by date (ascending)
        
        Returns:
            (train_df, test_df): Train set (earlier dates), test set (later dates)
        
        Raises:
            ValueError: If DataFrame is None, empty, lacks DatetimeIndex, or too small
        """
        self._validate_input(df)
        
        # Calculate split point
        n = len(df)
        split_point = int(n * (1 - self.test_size))
        
        # Chronological split: no shuffling, preserves temporal order
        train = df.iloc[:split_point]
        test = df.iloc[split_point:]
        
        return train, test


__all__ = ["TimeSeriesSplit"]
