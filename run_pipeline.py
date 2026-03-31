"""
run_pipeline.py — end-to-end training pipeline

Steps:
  1. Load raw OHLCV data from disk
  2. Attach labels (forward return classification)
  3. Build features (Bollinger, RSI, MACD, ATR, volume, price, SMA trend, 52w anchors)
  4. Build combined dataset, split train/test, save to disk
  5. Cross-validate on training set
  6. Hyperparameter search (Optuna, 50 trials) to maximise ROC-AUC
  7. Train final model with best hyperparams
  8. Evaluate on held-out test set
  9. Save model, metrics, and feature importance
"""

import argparse
import logging

from src.data.fetcher       import load_all
from src.data.labeler       import label_volatility_all
from src.features.pipeline  import build_features_all, FEATURE_COLS
from src.features.market    import build_market_features_all
from src.data.preprocessor  import (
    build_dataset, split_dataset,
    save_processed, save_splits,
)
from src.training.trainer   import train_cv, train_final, save_model
from src.training.tuner     import run_hyperparameter_search
from src.training.evaluator import evaluate_test, save_metrics, save_feature_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


def main(tune: bool = True, n_trials: int = 50):
    # ── 1. Load raw data ─────────────────────────────────────────────────────
    log.info("=== Step 1: Loading raw data ===")
    raw = load_all()

    # ── 2. Label ─────────────────────────────────────────────────────────────
    log.info("=== Step 2: Labeling ===")
    labeled = label_volatility_all(raw)

    # ── 3. Feature engineering ───────────────────────────────────────────────
    log.info("=== Step 3: Feature engineering ===")
    featured = build_features_all(labeled)
    featured = build_market_features_all(featured)

    # ── 4. Build dataset & split ─────────────────────────────────────────────
    log.info("=== Step 4: Building dataset & splitting ===")
    dataset = build_dataset(featured)
    save_processed(dataset, "all_tickers")

    train_df, test_df = split_dataset(dataset)
    save_splits(train_df, test_df)

    # ── 5. Cross-validation ──────────────────────────────────────────────────
    log.info("=== Step 5: Cross-validation ===")
    train_cv(train_df)

    # ── 6. Hyperparameter search ─────────────────────────────────────────────
    hyperparams = None
    if tune:
        log.info("=== Step 6: Hyperparameter search (%d trials) ===", n_trials)
        hyperparams = run_hyperparameter_search(train_df, n_trials=n_trials)
    else:
        log.info("=== Step 6: Skipping hyperparameter search (--no-tune) ===")

    # ── 7. Train final model ─────────────────────────────────────────────────
    log.info("=== Step 7: Training final model ===")
    model = train_final(train_df, hyperparams=hyperparams)
    save_model(model)

    # ── 8 & 9. Evaluate & save artifacts ────────────────────────────────────
    log.info("=== Step 8: Evaluating on test set ===")
    test_metrics = evaluate_test(model, test_df)
    save_metrics(test_metrics)
    save_feature_importance(model, FEATURE_COLS)

    log.info("=== Pipeline complete ===")
    log.info(
        "Test Results — Accuracy: %.4f  Precision: %.4f  Recall: %.4f  F1: %.4f  ROC-AUC: %.4f",
        test_metrics["accuracy"],  test_metrics["precision"], test_metrics["recall"],
        test_metrics["f1"],        test_metrics["roc_auc"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter search")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trial count")
    args = parser.parse_args()
    main(tune=not args.no_tune, n_trials=args.trials)
