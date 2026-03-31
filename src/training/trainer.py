import json
import logging
from pathlib import Path

import pandas as pd

from config.settings import MODELS_DIR
from src.data.preprocessor import get_xy, get_cv_splitter
from src.models.xgboost_model import XGBoostModel
from src.training.evaluator import evaluate, find_best_threshold

log = logging.getLogger(__name__)


def train_cv(train_df: pd.DataFrame) -> dict:
    """
    Run time-series cross-validation on the training set.
    Returns a dict with per-fold metrics and the mean across folds.
    """
    X, y = get_xy(train_df)
    cv   = get_cv_splitter()

    fold_metrics = []
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = XGBoostModel()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

        metrics = evaluate(model, X_val, y_val)
        metrics["fold"] = fold
        fold_metrics.append(metrics)

        log.info(
            "Fold %s | precision=%.3f recall=%.3f f1=%.3f roc_auc=%.3f",
            fold, metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["roc_auc"],
        )

    mean_metrics = {
        k: sum(m[k] for m in fold_metrics) / len(fold_metrics)
        for k in fold_metrics[0]
        if k != "fold"
    }
    log.info(
        "CV mean | precision=%.3f recall=%.3f f1=%.3f roc_auc=%.3f",
        mean_metrics["precision"], mean_metrics["recall"],
        mean_metrics["f1"], mean_metrics["roc_auc"],
    )
    results = {"folds": fold_metrics, "mean": mean_metrics}

    path = MODELS_DIR.parent / "results" / "cv_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("CV metrics saved to %s.", path)

    return results


def train_final(train_df: pd.DataFrame, hyperparams: dict | None = None) -> XGBoostModel:
    """
    Train a final model, then find the optimal decision threshold.

    The last 15% of training rows (chronologically) are held out as a threshold-search
    set — the model never sees these rows during fit, so the threshold found is unbiased.

    hyperparams: optional dict of XGBoost params (e.g. from the Optuna tuner).
                 If None, the model's defaults are used.
    """
    X, y = get_xy(train_df)

    n_val   = max(30, int(len(X) * 0.15))
    X_fit,  X_thresh = X.iloc[:-n_val],  X.iloc[-n_val:]
    y_fit,  y_thresh = y.iloc[:-n_val],  y.iloc[-n_val:]

    model = XGBoostModel(**(hyperparams or {}))
    model.fit(X_fit, y_fit, eval_set=[(X_thresh, y_thresh)])
    model.threshold = find_best_threshold(model, X_thresh, y_thresh)
    log.info("Final model trained on %s rows. Threshold=%.2f", len(X_fit), model.threshold)
    return model


def save_model(model: XGBoostModel, name: str = "lgbm_final") -> Path:
    path = MODELS_DIR / f"{name}.joblib"
    model.save(str(path))
    log.info("Model saved to %s.", path)
    return path
