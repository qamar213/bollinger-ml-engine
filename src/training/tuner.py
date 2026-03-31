"""
tuner.py — Optuna-based hyperparameter search for XGBoostModel.

Usage:
    from src.training.tuner import run_hyperparameter_search
    best_params = run_hyperparameter_search(train_df, n_trials=50)

The tuner optimises mean ROC-AUC across TimeSeriesSplit folds.
After finding the best params it returns them so train_final() can
use them when training the production model.
"""
import json
import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from config.settings import MODELS_DIR, N_SPLITS, RANDOM_STATE
from src.data.preprocessor import get_xy
from src.models.xgboost_model import XGBoostModel

log = logging.getLogger(__name__)

# Suppress Optuna's per-trial INFO noise — only show WARNING and above from it
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Single Optuna trial: sample hyperparams, run CV, return mean ROC-AUC."""
    params = dict(
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
        # scale_pos_weight < naive_ratio → favours precision (fewer FP)
        # scale_pos_weight > naive_ratio → favours recall  (fewer FN)
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 0.5, 8.0),
    )

    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    auc_scores = []

    for tr_idx, val_idx in cv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = XGBoostModel(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

        proba = model.predict_proba(X_val)
        if y_val.nunique() < 2:
            continue  # skip degenerate fold
        auc_scores.append(roc_auc_score(y_val, proba))

    return float(np.mean(auc_scores)) if auc_scores else 0.0


def run_hyperparameter_search(
    train_df: pd.DataFrame,
    n_trials: int = 50,
    timeout: int | None = None,
) -> dict:
    """
    Run Bayesian hyperparameter search over n_trials.

    Parameters
    ----------
    train_df  : full training DataFrame (features + label)
    n_trials  : number of Optuna trials (default 50, ~5-10 min on 50 tickers)
    timeout   : optional wall-clock limit in seconds

    Returns
    -------
    best_params : dict of the best hyperparameter values found
    """
    X, y = get_xy(train_df)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: _objective(trial, X, y),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    best = study.best_trial
    log.info(
        "Hyperparameter search complete. Best ROC-AUC=%.4f in %d trials.",
        best.value, len(study.trials),
    )
    log.info("Best params: %s", best.params)

    # Save best params and full trial history
    results_dir = MODELS_DIR.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    params_path = results_dir / "best_hyperparams.json"
    with open(params_path, "w") as f:
        json.dump({"best_roc_auc": best.value, "params": best.params}, f, indent=2)
    log.info("Best params saved to %s.", params_path)

    history = [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in study.trials
        if t.value is not None
    ]
    history_path = results_dir / "tuning_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Trial history saved to %s.", history_path)

    return best.params
