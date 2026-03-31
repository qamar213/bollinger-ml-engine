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


def attach_volatility_labels(
    df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Volatility regime labeling: predict whether realized volatility INCREASES
    over the next `window` trading days relative to the current `window` days.

    current_vol[t]  = std(log_returns[t-window+1 : t])
    future_vol[t]   = std(log_returns[t+1 : t+window])
    label[t]        = 1 if future_vol > current_vol  (volatility expanding)
                    = 0 if future_vol <= current_vol  (volatility contracting/flat)

    Why this works: volatility is autocorrelated (GARCH effect) — today's ATR,
    Bollinger bandwidth, and recent return dispersion are strong predictors of
    the next volatility state. This is a ~50/50 balanced binary problem, so
    accuracy is a direct and meaningful metric (no class-imbalance games).
    """
    close = df[PRICE_COL].squeeze()
    log_ret = np.log(close / close.shift(1))

    # Current realized vol: std of last `window` log returns (known at time t)
    current_vol = log_ret.rolling(window).std()

    # Future realized vol: std of next `window` log returns (unknown at time t)
    # Build it from shifted series to avoid using a rolling window on the future
    future_vol = pd.concat(
        [log_ret.shift(-i) for i in range(1, window + 1)], axis=1
    ).std(axis=1)

    label = (future_vol > current_vol).astype(int)
    label.name = "label"

    out = df.copy()
    out["current_vol"] = current_vol
    out["future_vol"]  = future_vol
    out["label"]       = label
    # Drop rows where either vol is NaN (start or end of series)
    out.dropna(subset=["current_vol", "future_vol"], inplace=True)

    n_expand  = (out["label"] == 1).sum()
    n_contract = (out["label"] == 0).sum()
    log.info(
        "Vol labels: %s rows | expanding=%s (%.1f%%) | contracting=%s (%.1f%%)",
        len(out), n_expand, n_expand / len(out) * 100,
        n_contract, n_contract / len(out) * 100,
    )
    return out


def label_volatility_all(
    data: dict[str, pd.DataFrame],
    window: int = 5,
) -> dict[str, pd.DataFrame]:
    """Apply attach_volatility_labels to every ticker in the data dict."""
    labeled = {}
    for ticker, df in data.items():
        log.info("Labeling volatility regime for %s...", ticker)
        labeled[ticker] = attach_volatility_labels(df, window=window)
    log.info("Volatility labels done for %s tickers.", len(labeled))
    return labeled


def label_all_cross_sectional(
    data: dict[str, pd.DataFrame],
    top_pct: float = 0.20,
    window: int = FORWARD_WINDOW,
) -> dict[str, pd.DataFrame]:
    """
    Cross-sectional labeling: on each date, label the top `top_pct` fraction of
    stocks by forward return as buy signals (label=1).

    This removes the ATR bias of a fixed absolute threshold. With a 5% threshold,
    volatile stocks (TSLA, NVDA) dominate the positives simply because they're more
    likely to move 5%+. Cross-sectional ranking forces the model to learn WHICH
    stocks will outperform their peers on a given day, not just WHICH stocks are
    volatile — a much cleaner directional signal.

    top_pct=0.20 with 50 tickers ≈ 10 buy signals per day, ~20% positive rate.
    """
    # Step 1: compute forward returns for every ticker
    fwd_series: dict[str, pd.Series] = {}
    for ticker, df in data.items():
        fwd_series[ticker] = compute_forward_return(df, window)

    # Step 2: build a cross-sectional matrix (dates × tickers)
    fwd_matrix = pd.DataFrame(fwd_series)  # shape: (dates, n_tickers)

    # Step 3: per-date threshold = the (1 - top_pct) quantile of forward returns
    # A stock is a buy if its forward return is >= this date's threshold.
    per_date_threshold = fwd_matrix.quantile(1.0 - top_pct, axis=1)

    # Step 4: assign labels ticker-by-ticker
    out: dict[str, pd.DataFrame] = {}
    for ticker, df in data.items():
        labeled = df.copy()
        fwd = fwd_series[ticker]
        labeled["forward_return"] = fwd

        thresh = per_date_threshold.reindex(fwd.index)
        labeled["label"] = (fwd >= thresh).astype(int)
        labeled.dropna(subset=["forward_return"], inplace=True)

        n_buy = (labeled["label"] == 1).sum()
        pct_buy = n_buy / len(labeled) * 100
        log.info(
            "Labels: %s rows | buy=%s (%.1f%%) | no-signal=%s (%.1f%%)",
            len(labeled), n_buy, pct_buy,
            (labeled["label"] == 0).sum(), 100 - pct_buy,
        )
        out[ticker] = labeled

    log.info("Labeled %s tickers (cross-sectional top %.0f%%).", len(out), top_pct * 100)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from src.data.fetcher import load_all

    data = load_all()
    labeled = label_all(data)

    # Quick sanity check on one ticker
    sample = labeled["AAPL"]
    print(sample[["Close", "forward_return", "label"]].tail(10))