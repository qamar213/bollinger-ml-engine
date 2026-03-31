import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from config.settings import RANDOM_STATE
from src.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost binary classifier wrapper.

    Default hyperparameters are reasonable starting points for daily OHLCV data.
    The tuner can override these via set_params().
    """

    _INTERNAL_VAL_FRAC = 0.15

    def __init__(self, **kwargs):
        params = dict(
            n_estimators=1000,          # high ceiling — early stopping decides actual count
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1.0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1,         # updated at fit-time for class imbalance
            early_stopping_rounds=50,   # stop if aucpr hasn't improved for 50 rounds
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        params.update(kwargs)
        self.model = XGBClassifier(**params)
        # Platt scaling — logistic regression fitted on held-out raw probabilities.
        # Produces smooth, continuous calibrated probabilities (unlike isotonic which
        # creates step-function distributions). Set to None until calibrate() is called.
        self._platt: LogisticRegression | None = None
        # Decision threshold — default 0.5, overridden after threshold optimisation
        self.threshold: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None) -> None:
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        if n_pos > 0:
            self.model.set_params(scale_pos_weight=n_neg / n_pos)

        if eval_set is None:
            # Hold out the most recent rows as a validation set for early stopping.
            # Using the last N% preserves time order — no future data leaks into training.
            n_val = max(30, int(len(X) * self._INTERNAL_VAL_FRAC))
            X_fit, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
            y_fit, y_val = y.iloc[:-n_val], y.iloc[-n_val:]
            eval_set = [(X_val, y_val)]
        else:
            X_fit, y_fit = X, y

        self.model.fit(X_fit, y_fit, eval_set=eval_set, verbose=False)
        # Reset calibration when refitted
        self._iso = None

    def calibrate(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit Platt scaling on a held-out set AFTER the base model is trained.
        Logistic regression on raw log-odds produces smooth calibrated probabilities
        across the full [0,1] range, avoiding the step-function artefacts of isotonic
        regression. Must be called with data the base model was NOT trained on.
        """
        raw_proba = self.model.predict_proba(X)[:, 1].reshape(-1, 1)
        self._platt = LogisticRegression(C=1.0, max_iter=1000)
        self._platt.fit(raw_proba, y.values)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        proba = self.model.predict_proba(X)[:, 1]
        preds = (proba >= self.threshold).astype(int)
        return pd.Series(preds, index=X.index, name="prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        raw = self.model.predict_proba(X)[:, 1]
        if self._platt is not None:
            proba = self._platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            proba = raw
        return pd.Series(proba, index=X.index, name="proba_buy")

    def get_feature_importance(self, feature_names: list[str]) -> pd.Series:
        scores = self.model.feature_importances_
        return pd.Series(scores, index=feature_names, name="importance").sort_values(ascending=False)

    def set_params(self, **params) -> None:
        self.model.set_params(**params)

    def save(self, path: str) -> None:
        joblib.dump({
            "model": self.model,
            "platt": self._platt,
            "threshold": self.threshold,
        }, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        instance = cls.__new__(cls)
        payload = joblib.load(path)
        instance.model     = payload["model"]
        instance._platt    = payload.get("platt")
        instance.threshold = payload.get("threshold", 0.5)
        return instance
