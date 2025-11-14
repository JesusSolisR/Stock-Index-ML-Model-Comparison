"""Minimal trainer base class for classification models.

Provides a small, testable interface used by concrete trainers
to prepare data, build a pipeline, fit, evaluate and persist models.
"""
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import joblib

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


class BaseTrainer:
    """Base trainer with small, stable API.

    Concrete trainers must implement `build_pipeline`.
    """

    def __init__(self,
                 features: List[str],
                 target: str = "price_up",
                 test_size: float = 0.2,
                 random_state: int = 42):
        self.features = features
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        self.pipeline: Optional[Pipeline] = None
        self.metrics: Optional[Dict] = None

    def prepare_data(self, df: pd.DataFrame, ts_split) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare X/y and apply a time-series split instance.

        `ts_split` is expected to be an object with a `split(df)` method
        (for example the project's `TimeSeriesSplit`).
        """
        X = df[self.features]
        y = df[self.target].astype(int)
        X_train, X_test = ts_split.split(X)
        y_train, y_test = y.loc[X_train.index], y.loc[X_test.index]
        return X_train, X_test, y_train, y_test

    def build_pipeline(self) -> Pipeline:
        raise NotImplementedError("Concrete trainers must implement build_pipeline")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        if self.pipeline is None:
            self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        y_pred = self.pipeline.predict(X_test)
        y_proba = None
        try:
            y_proba = self.pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred, digits=4),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "roc_auc": float(roc_auc_score(y_test, y_proba)) if (y_proba is not None and len(set(y_test)) > 1) else None,
        }
        self.metrics = metrics
        return metrics

    def predict(self, X: pd.DataFrame):
        return self.pipeline.predict(X)

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, p)

    @classmethod
    def load_pipeline(cls, path: str):
        return joblib.load(path)
