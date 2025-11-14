"""Logistic regression trainer backed by `BaseTrainer`.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseTrainer


class LogisticTrainer(BaseTrainer):
    """Simple logistic regression trainer with a scaler + classifier pipeline."""

    def __init__(self, features, target: str = "price_up", test_size: float = 0.2, random_state: int = 42, solver: str = "liblinear"):
        super().__init__(features=features, target=target, test_size=test_size, random_state=random_state)
        self.solver = solver

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver=self.solver, random_state=self.random_state))
        ])
