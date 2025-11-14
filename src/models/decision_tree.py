"""Decision Tree trainer that handles numeric + categorical preprocessing."""
from typing import List, Optional

from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from .base import BaseTrainer


class DecisionTreeTrainer(BaseTrainer):
    """Decision tree trainer with a ColumnTransformer preprocessor.

    Provide `numeric_features` and `categorical_features` lists when
    instantiating; the trainer will apply one-hot encoding to categoricals
    and passthrough numeric features.
    """

    def __init__(self,
                 features: List[str],
                 numeric_features: Optional[List[str]] = None,
                 categorical_features: Optional[List[str]] = None,
                 target: str = "price_up",
                 max_depth: Optional[int] = 10,
                 test_size: float = 0.2,
                 random_state: int = 42):
        super().__init__(features=features, target=target, test_size=test_size, random_state=random_state)
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.max_depth = max_depth

    def build_pipeline(self) -> Pipeline:
        transformers = []
        if self.numeric_features:
            transformers.append(("num", "passthrough", self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), self.categorical_features))

        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

        return Pipeline([
            ("preprocessor", preprocessor),
            ("clf", DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state))
        ])
