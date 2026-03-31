from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """Minimal interface all models must implement."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel": ...
