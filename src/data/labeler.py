import logging

import pandas as pd
import numpy as np

from config.settings import (
    PRICE_COL, FORWARD_WINDOW, BUY_THRESHOLD, SELL_THRESHOLD
)

log = logging.getLogger(__name__)


def compute_forward_return(df: pd.DataFrame, window: int = FORWARD_WINDOW) -> pd.Series:
    """
    Compute the forward return over `window` days.
    forward_return[t] = (price[t + window] - price[t]) / price[t]
    The last `window` rows will be NaN since we don't have future data.
    """
    price = df[PRICE_COL].squeeze()
    forward_return = price.shift(-window) / price - 1
    return forward_return.rename("forward_return")


def compute_labels(df: pd.DataFrame, window: int = FORWARD_WINDOW) -> pd.Series:
    """
    Generate binary labels from forward returns:
        1 = buy signal  (forward return >= BUY_THRESHOLD)
        0 = no signal   (forward return between thresholds or below SELL_THRESHOLD)

    Rows where the forward return is NaN (end of series) are dropped downstream.
    """
    fwd = compute_forward_return(df, window)
    labels = (fwd >= BUY_THRESHOLD).astype(int)
    labels.name = "label"
    return labels


def attach_labels(df: pd.DataFrame, window: int = FORWARD_WINDOW) -> pd.DataFrame:
    """
    Attach forward return and binary label columns to the DataFrame.
    Drops rows at the end where future data is unavailable.
    """
    fwd    = compute_forward_return(df, window)
    labels = compute_labels(df, window)

    out = df.copy()
    out["forward_return"] = fwd
    out["label"]          = labels
    out.dropna(subset=["forward_return"], inplace=True)

    n_buy  = (out["label"] == 1).sum()
    n_sell = (out["label"] == 0).sum()
    pct_buy = n_buy / len(out) * 100

    log.info(
        "Labels: %s rows | buy=%s (%.1f%%) | no-signal=%s (%.1f%%)",
        len(out), n_buy, pct_buy, n_sell, 100 - pct_buy
    )
    return out


def label_all(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply attach_labels to every ticker in the data dict."""
    labeled = {}
    for ticker, df in data.items():
        log.info("Labeling %s...", ticker)
        labeled[ticker] = attach_labels(df)
    log.info("Labeled %s tickers.", len(labeled))
    return labeled


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from src.data.fetcher import load_all

    data = load_all()
    labeled = label_all(data)

    # Quick sanity check on one ticker
    sample = labeled["AAPL"]
    print(sample[["Close", "forward_return", "label"]].tail(10))