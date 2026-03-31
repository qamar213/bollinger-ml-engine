import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    accuracy_score,
    confusion_matrix, classification_report,
)

from config.settings import RESULTS_DIR
from src.models.xgboost_model import XGBoostModel

log = logging.getLogger(__name__)


def find_best_threshold(
    model: XGBoostModel,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """
    Find the threshold that maximises accuracy on the validation set.

    For volatility regime prediction the classes are ~50/50, so accuracy
    is a direct and symmetric metric — the best threshold is simply the
    one that correctly classifies the most rows.
    """
    proba = model.predict_proba(X)

    best_thresh, best_acc = 0.5, 0.0

    # Diagnostic: log accuracy at key thresholds
    log.info("Accuracy curve on threshold validation set:")
    for t_diag in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        preds = (proba >= t_diag).astype(int)
        acc = accuracy_score(y, preds)
        log.info("  t=%.2f | accuracy=%.3f", t_diag, acc)

    for t in np.arange(0.20, 0.81, 0.01):
        preds = (proba >= t).astype(int)
        acc = accuracy_score(y, preds)
        if acc > best_acc:
            best_acc, best_thresh = acc, float(t)

    log.info(
        "Threshold %.2f → accuracy=%.3f on validation set.",
        best_thresh, best_acc,
    )
    return best_thresh


def evaluate(model: XGBoostModel, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute classification metrics for a fitted model on (X, y)."""
    preds = model.predict(X)
    proba = model.predict_proba(X)

    return {
        "accuracy":  accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall":    recall_score(y, preds, zero_division=0),
        "f1":        f1_score(y, preds, zero_division=0),
        "roc_auc":   roc_auc_score(y, proba),
    }


def evaluate_test(model: XGBoostModel, test_df: pd.DataFrame) -> dict:
    """Full evaluation on the held-out test set — logs and returns metrics."""
    from src.data.preprocessor import get_xy
    X, y = get_xy(test_df)

    preds   = model.predict(X)
    metrics = evaluate(model, X, y)

    log.info("Test set evaluation:")
    log.info("  accuracy  : %.4f", metrics["accuracy"])
    log.info("  precision : %.4f", metrics["precision"])
    log.info("  recall    : %.4f", metrics["recall"])
    log.info("  f1        : %.4f", metrics["f1"])
    log.info("  roc_auc   : %.4f", metrics["roc_auc"])
    log.info("\n%s", classification_report(y, preds, target_names=["contracting", "expanding"]))

    cm = confusion_matrix(y, preds)
    log.info("Confusion matrix:\n%s", cm)

    return metrics


def save_metrics(metrics: dict, name: str = "test_metrics") -> Path:
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved to %s.", path)
    return path


def save_feature_importance(model: XGBoostModel, feature_names: list[str], name: str = "feature_importance") -> Path:
    importance = model.get_feature_importance(feature_names)
    path = RESULTS_DIR / f"{name}.csv"
    importance.to_csv(path, header=True)
    log.info("Feature importance saved to %s.", path)
    return path
