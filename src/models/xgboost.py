"""XGBoost trainer wrapper implementing the BaseTrainer API.

Note: xgboost must be installed in the environment to use this trainer.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseTrainer

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - heavy dep may not exist in tests
    XGBClassifier = None


class XGBTrainer(BaseTrainer):
    def __init__(self, features, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1, target: str = "price_up", test_size: float = 0.2, random_state: int = 42):
        super().__init__(features=features, target=target, test_size=test_size, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def build_pipeline(self) -> Pipeline:
        if XGBClassifier is None:
            raise ImportError("xgboost is required to use XGBTrainer. Install with `pip install xgboost`.")
        clf = XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, random_state=self.random_state, eval_metric='logloss')
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
