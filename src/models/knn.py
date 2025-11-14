"""K-Nearest Neighbors trainer using the BaseTrainer API."""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseTrainer


class KNNTrainer(BaseTrainer):
    """KNN trainer with a simple scaler + KNN pipeline."""

    def __init__(self, features, n_neighbors: int = 5, target: str = "price_up", test_size: float = 0.2, random_state: int = 42):
        super().__init__(features=features, target=target, test_size=test_size, random_state=random_state)
        self.n_neighbors = n_neighbors

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=self.n_neighbors))
        ])
