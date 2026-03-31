import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from config.settings import RANDOM_STATE
from src.models.base_model import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM binary classifier wrapper.

    LightGBM uses leaf-wise tree growth (best-first) instead of XGBoost's
    level-wise growth, allowing deeper, more targeted splits that often
    achieve better AUC on tabular data with the same number of leaves.
    """

    _INTERNAL_VAL_FRAC = 0.15

    def __init__(self, **kwargs):
        params = dict(
            n_estimators=1000,
            max_depth=-1,           # no depth limit — controlled by num_leaves
            num_leaves=31,          # key parameter: controls model complexity
            learning_rate=0.05,
            subsample=0.8,
            subsample_freq=1,       # required for subsample to take effect
            colsample_bytree=0.8,
            min_child_samples=20,   # LightGBM equiv of min_child_weight
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1,     # updated at fit-time for class imbalance
            early_stopping_rounds=50,
            eval_metric="average_precision",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        params.update(kwargs)
        self.model = LGBMClassifier(**params)
        self.threshold: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None) -> None:
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        if n_pos > 0:
            self.model.set_params(scale_pos_weight=n_neg / n_pos)

        if eval_set is None:
            n_val = max(30, int(len(X) * self._INTERNAL_VAL_FRAC))
            X_fit, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
            y_fit, y_val = y.iloc[:-n_val], y.iloc[-n_val:]
            eval_set = [(X_val, y_val)]
        else:
            X_fit, y_fit = X, y

        import lightgbm as lgb
        self.model.fit(
            X_fit, y_fit,
            eval_set=eval_set,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        proba = self.model.predict_proba(X)[:, 1]
        preds = (proba >= self.threshold).astype(int)
        return pd.Series(preds, index=X.index, name="prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        proba = self.model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=X.index, name="proba_buy")

    def get_feature_importance(self, feature_names: list[str]) -> pd.Series:
        scores = self.model.feature_importances_
        return pd.Series(scores, index=feature_names, name="importance").sort_values(ascending=False)

    def set_params(self, **params) -> None:
        self.model.set_params(**params)

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        instance = cls.__new__(cls)
        payload = joblib.load(path)
        instance.model     = payload["model"]
        instance.threshold = payload.get("threshold", 0.5)
        return instance
