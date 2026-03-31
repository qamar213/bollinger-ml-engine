import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import ROOT_DIR, START_DATE, END_DATE, SECTOR_MAP

log = logging.getLogger(__name__)

_SPY_CACHE = ROOT_DIR / ".cache" / "spy.parquet"

# Sector index → ETF ticker
_SECTOR_ETFS = {
    0: "XLK",   # Technology
    1: "XLC",   # Communication Services
    2: "XLY",   # Consumer Discretionary
    3: "XLP",   # Consumer Staples
    4: "XLF",   # Financials
    5: "XLV",   # Health Care
    6: "XLI",   # Industrials
    7: "XLE",   # Energy
    8: "XLU",   # Utilities
    9: "XLB",   # Materials
}


def _load_spy() -> pd.Series:
    """
    Load SPY daily closes, using a local cache to avoid re-fetching.
    Returns a Series indexed by date.
    """
    if _SPY_CACHE.exists():
        spy = pd.read_parquet(_SPY_CACHE)["Close"]
    else:
        log.info("Fetching SPY data from Yahoo Finance...")
        raw = yf.download("SPY", start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index)
        _SPY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(_SPY_CACHE)
        spy = raw["Close"]
        log.info("SPY fetched: %s rows.", len(spy))

    spy.index = pd.to_datetime(spy.index)
    return spy


def _build_spy_features(spy: pd.Series) -> pd.DataFrame:
    """
    Compute market regime signals from SPY closes:
      spy_ret_5d    — SPY 5-day log return (short-term momentum)
      spy_ret_20d   — SPY 20-day log return (medium-term trend)
      spy_above_200 — 1 if SPY close > 200-day SMA (broad bull/bear regime)
    """
    log_ret = np.log(spy / spy.shift(1))
    sma200  = spy.rolling(200).mean()

    features = pd.DataFrame({
        "spy_ret_5d":    log_ret.rolling(5).sum(),
        "spy_ret_20d":   log_ret.rolling(20).sum(),
        "spy_above_200": (spy > sma200).astype(int),
    }, index=spy.index)
    return features


def attach_market_features(df: pd.DataFrame, spy_features: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join SPY market regime features onto a ticker DataFrame by date index.
    Also computes relative-strength features (stock return minus SPY return) which
    capture whether the stock is outperforming/underperforming the market.
    Dates with no SPY data (holidays, etc.) are forward-filled from the prior day.
    """
    out = df.join(spy_features, how="left")
    mkt_cols = ["spy_ret_5d", "spy_ret_20d", "spy_above_200"]
    out[mkt_cols] = out[mkt_cols].ffill()

    # Relative strength vs market — directional alpha signal.
    # A stock with rs_vs_spy_5d > 0 is outperforming SPY over the last week;
    # combined with oversold indicators this can signal a recovery/reversal.
    out["rs_vs_spy_5d"]  = out["ret_5d"]  - out["spy_ret_5d"]
    out["rs_vs_spy_20d"] = out["ret_20d"] - out["spy_ret_20d"]
    return out


def _load_sector_etf(etf: str) -> pd.Series:
    """Load a sector ETF's daily closes, cached locally."""
    cache = ROOT_DIR / ".cache" / f"{etf.lower()}.parquet"
    if cache.exists():
        closes = pd.read_parquet(cache)["Close"]
    else:
        log.info("Fetching %s from Yahoo Finance...", etf)
        raw = yf.download(etf, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index)
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache)
        closes = raw["Close"]
    closes.index = pd.to_datetime(closes.index)
    return closes


def _sector_log_returns(closes: pd.Series) -> pd.DataFrame:
    """Compute 5d, 20d log returns and trend indicator for a sector ETF."""
    log_ret = np.log(closes / closes.shift(1))
    sma50   = closes.rolling(50).mean()
    sma200  = closes.rolling(200).mean()
    return pd.DataFrame({
        "ret_5d":        log_ret.rolling(5).sum(),
        "ret_20d":       log_ret.rolling(20).sum(),
        "above_sma50":   (closes > sma50).astype(int),   # sector in short-term uptrend
        "above_sma200":  (closes > sma200).astype(int),  # sector in long-term uptrend
    }, index=closes.index)


def attach_sector_features(df: pd.DataFrame, ticker: str,
                            sector_rets: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Attach sector-relative strength features for a ticker.
    rs_vs_sector_5d  = stock 5d return  - sector ETF 5d return
    rs_vs_sector_20d = stock 20d return - sector ETF 20d return

    Stocks outperforming their sector are exhibiting true alpha, not just
    riding sector tailwinds — a stronger directional signal than SPY-relative RS.
    """
    sector_id = SECTOR_MAP.get(ticker, -1)
    if sector_id == -1 or sector_id not in sector_rets:
        df["rs_vs_sector_5d"]  = 0.0
        df["rs_vs_sector_20d"] = 0.0
        return df

    sec = sector_rets[sector_id].reindex(df.index).ffill()
    df["rs_vs_sector_5d"]    = df["ret_5d"]  - sec["ret_5d"]
    df["rs_vs_sector_20d"]   = df["ret_20d"] - sec["ret_20d"]
    df["sector_above_sma50"]  = sec["above_sma50"]
    df["sector_above_sma200"] = sec["above_sma200"]
    return df


def build_market_features_all(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Attach SPY market regime + sector-relative strength features to every ticker."""
    spy = _load_spy()
    spy_features = _build_spy_features(spy)

    # Load all unique sector ETFs once
    sector_rets: dict[int, pd.DataFrame] = {}
    for sector_id, etf in _SECTOR_ETFS.items():
        try:
            closes = _load_sector_etf(etf)
            sector_rets[sector_id] = _sector_log_returns(closes)
        except Exception as e:
            log.warning("Could not load sector ETF %s: %s", etf, e)

    out = {}
    for ticker, df in data.items():
        df = attach_market_features(df, spy_features)
        df = attach_sector_features(df, ticker, sector_rets)
        out[ticker] = df
    log.info("Market regime + sector features attached to %s tickers.", len(out))
    return out
