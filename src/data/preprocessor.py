import logging

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from config.settings import (
    DATA_PROC_DIR, DATA_SPLITS_DIR,
    TEST_SIZE, N_SPLITS, SECTOR_MAP,
)
from src.features.pipeline import FEATURE_COLS

log = logging.getLogger(__name__)


def build_dataset(labeled_features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all tickers into one flat DataFrame with a 'ticker' column.
    Keeps only FEATURE_COLS + label. Drops any rows with NaN features.
    """
    frames = []
    for ticker, df in labeled_features.items():
        if "label" not in df.columns:
            log.warning("%s: missing label column, skipping.", ticker)
            continue

        # These columns are generated inside build_dataset or attached via market features,
        # not by the per-ticker feature pipeline — exclude from the pre-check
        _generated_cols = {"ticker_encoded", "sector"}
        cols_to_check = [c for c in FEATURE_COLS if c not in _generated_cols]
        needed = cols_to_check + ["label"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            log.warning("%s: missing columns %s, skipping.", ticker, missing)
            continue

        sub = df[needed].copy()
        sub["ticker"] = ticker
        frames.append(sub)

    if not frames:
        raise ValueError("No valid ticker data to combine.")

    combined = pd.concat(frames)
    combined.sort_index(inplace=True)   # sort by date so iloc split is truly chronological

    # Encode ticker as an integer so XGBoost can use it as a categorical split feature.
    # Fitted on the full dataset so train/test share the same mapping.
    le = LabelEncoder()
    combined["ticker_encoded"] = le.fit_transform(combined["ticker"])
    combined["sector"] = combined["ticker"].map(SECTOR_MAP).fillna(-1).astype(int)
    log.info("Ticker encoding: %s unique tickers.", len(le.classes_))

    before = len(combined)
    combined.dropna(inplace=True)
    log.info("Dataset: %s rows after dropping NaNs (removed %s).", len(combined), before - len(combined))
    return combined


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split using TEST_SIZE fraction.
    Because the data is time-series, we split by position, not randomly.
    Returns (train_df, test_df).
    """
    n       = len(df)
    cutoff  = int(n * (1 - TEST_SIZE))
    train   = df.iloc[:cutoff].copy()
    test    = df.iloc[cutoff:].copy()
    log.info("Split: %s train rows, %s test rows.", len(train), len(test))
    return train, test


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and label series y from a dataset DataFrame."""
    X = df[FEATURE_COLS]
    y = df["label"]
    return X, y


def save_processed(df: pd.DataFrame, name: str) -> None:
    path = DATA_PROC_DIR / f"{name}.parquet"
    df.to_parquet(path, index=True)
    log.info("Saved %s to %s.", name, path)


def save_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:
    for name, df in [("train", train), ("test", test)]:
        path = DATA_SPLITS_DIR / f"{name}.parquet"
        df.to_parquet(path, index=True)
        log.info("Saved %s split (%s rows) to %s.", name, len(df), path)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA_SPLITS_DIR / "train.parquet")
    test  = pd.read_parquet(DATA_SPLITS_DIR / "test.parquet")
    log.info("Loaded splits: %s train, %s test.", len(train), len(test))
    return train, test


def get_cv_splitter() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=N_SPLITS)
