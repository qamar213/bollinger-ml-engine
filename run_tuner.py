"""
run_tuner.py — standalone hyperparameter search script.

Loads the saved train split and runs Optuna to find XGBoost params
that maximise cross-validated ROC-AUC, then saves the best params to
results/best_hyperparams.json.

Usage:
    python run_tuner.py [--trials N]
"""
import argparse
import logging

from src.data.preprocessor import load_splits
from src.training.tuner import run_hyperparameter_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()

    log.info("Loading train split from disk...")
    train_df, _ = load_splits()

    log.info("Starting hyperparameter search (%d trials)...", args.trials)
    best_params = run_hyperparameter_search(train_df, n_trials=args.trials)

    log.info("=== Best hyperparameters ===")
    for k, v in best_params.items():
        log.info("  %-20s = %s", k, v)


if __name__ == "__main__":
    main()
