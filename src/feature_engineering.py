"""Feature engineering utilities.

This file provides a compact FeatureEngineering class with a small set of
well-tested methods most pipelines need: pct change, SMA/EMA, RSI, MACD,
lags, simple temporal features and target creation. Each method returns a
new DataFrame copy and keeps the implementation minimal and readable.
"""

from typing import Optional
import pandas as pd


class FeatureEngineering:
    """Small, focused feature engineering helper.

    Params:
        short (int): short window for moving averages
        long (int): long window for moving averages
        rsi (int): period for RSI
    """

    def __init__(self, short: int = 5, long: int = 20, rsi: int = 14):
        self.short = short
        self.long = long
        self.rsi = rsi

    def pct_change(self, df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
        df = df.copy()
        df["pct_change"] = df[col].pct_change() * 100
        return df

    def sma(self, df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
        df = df.copy()
        df[f"sma_{self.short}"] = df[col].rolling(self.short).mean()
        df[f"sma_{self.long}"] = df[col].rolling(self.long).mean()
        return df

    def ema(self, df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
        df = df.copy()
        df[f"ema_{self.short}"] = df[col].ewm(span=self.short, adjust=False).mean()
        df[f"ema_{self.long}"] = df[col].ewm(span=self.long, adjust=False).mean()
        return df

    def rsi_simple(self, df: pd.DataFrame, col: str = "close", period: Optional[int] = None) -> pd.DataFrame:
        period = period or self.rsi
        df = df.copy()
        delta = df[col].diff()
        up = delta.clip(lower=0).rolling(period).mean()
        down = -delta.clip(upper=0).rolling(period).mean()
        rs = up / down
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def macd(self, df: pd.DataFrame, col: str = "close", fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
        df = df.copy()
        ema_fast = df[col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[col].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=sig, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    def lag(self, df: pd.DataFrame, col: str = "pct_change", n: int = 3) -> pd.DataFrame:
        df = df.copy()
        for i in range(1, n + 1):
            df[f"lag_{i}"] = df[col].shift(i)
        return df

    def temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df["dow"] = df.index.dayofweek
        df["month"] = df.index.month
        return df

    def add_targets(self, df: pd.DataFrame, col: str = "close", shift: int = 1) -> pd.DataFrame:
        df = df.copy()
        df["future_return"] = df[col].pct_change(periods=shift).shift(-shift) * 100
        df["price_up"] = (df["future_return"] > 0).astype(float)
        return df

    def engineer(self, df: pd.DataFrame, include_target: bool = True, lags: int = 3) -> pd.DataFrame:
        """Run a compact pipeline and return a new DataFrame."""
        df = df.copy()
        df = self.pct_change(df)
        df = self.sma(df)
        df = self.ema(df)
        df = self.rsi_simple(df)
        df = self.macd(df)
        df = self.lag(df, col="pct_change", n=lags)
        df = self.temporal(df)
        if include_target:
            df = self.add_targets(df)
        return df


__all__ = ["FeatureEngineering"]
