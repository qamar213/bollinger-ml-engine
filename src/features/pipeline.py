import logging

import pandas as pd

from src.features.bollinger  import add_bollinger_bands
from src.features.momentum   import add_momentum
from src.features.volatility import add_volatility
from src.features.volume     import add_volume
from src.features.price      import add_price_features

log = logging.getLogger(__name__)

# Base feature columns (current-day values)
_BASE_FEATURE_COLS = [
    # Ticker identity & sector — added by build_dataset(), not the feature pipeline
    "ticker_encoded",
    "sector",
    # Market regime — SPY-based signals added by build_market_features_all()
    "spy_ret_5d", "spy_ret_20d", "spy_above_200",
    # Relative strength vs SPY — alpha signal (stock outperforming/underperforming market)
    "rs_vs_spy_5d", "rs_vs_spy_20d",
    # Relative strength vs sector ETF — purer alpha, removes sector tailwinds
    "rs_vs_sector_5d", "rs_vs_sector_20d",
    # Sector trend state — is the sector itself in an uptrend?
    "sector_above_sma50", "sector_above_sma200",
    # Bollinger
    "bb_pct_b", "bb_bandwidth", "bb_squeeze",
    # Momentum — macd_pct/macd_signal_pct are price-normalised so they're comparable across tickers
    "rsi", "macd_pct", "macd_signal_pct", "macd_hist",
    # Stochastic oscillator — close position within recent high/low range
    "stoch_k", "stoch_d",
    # Volatility
    "atr_pct",
    # ADX — trend strength (high = strong trend, low = choppy/ranging)
    "adx",
    # Volume — obv_divergence (OBV minus its EMA) centres around 0, comparable across tickers
    "volume_ratio", "obv_divergence",
    # Money Flow Index — volume-weighted RSI confirms price moves with volume
    "mfi",
    # Price
    "log_return", "ret_5d", "ret_10d", "ret_20d", "hl_range", "gap",
    # Trend position — captures whether the stock is above/below key moving averages.
    # These are critical for directional prediction: volatile stocks in uptrends are
    # far more likely to hit the buy threshold than volatile stocks in downtrends.
    "close_to_sma20", "close_to_sma50", "close_to_sma200", "sma20_slope",
    # 52-week anchoring — distance from yearly extremes captures momentum regime
    "dist_52w_high", "dist_52w_low",
    # Medium/long-term momentum — Jegadeesh-Titman factor (one of the most robust in finance)
    "ret_63d", "ret_126d", "mom_12_1",
]

# Which columns get lagged versions (the ones where direction/change matters most)
_LAG_COLS = [
    "rsi", "macd_hist", "macd_pct", "bb_pct_b", "bb_bandwidth",
    "volume_ratio", "obv_divergence", "atr_pct", "log_return",
    "close_to_sma20", "sma20_slope",
    "stoch_k", "mfi", "adx",
]

# Lag windows to create
_LAGS = [1, 2, 3, 5]


def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each column in _LAG_COLS, add lag-1, lag-2, lag-3, lag-5 columns.
    E.g. rsi_lag1 = rsi shifted by 1 day.
    This lets the model see whether an indicator is rising or falling,
    which is far more predictive than the raw level alone.
    """
    out = df.copy()
    for col in _LAG_COLS:
        if col not in df.columns:
            continue
        for lag in _LAGS:
            out[f"{col}_lag{lag}"] = df[col].shift(lag)
    return out


def _lag_col_names() -> list[str]:
    return [f"{col}_lag{lag}" for col in _LAG_COLS for lag in _LAGS]


# Full list of feature columns fed to the model
FEATURE_COLS = _BASE_FEATURE_COLS + _lag_col_names()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a single ticker DataFrame.
    Returns a DataFrame with OHLCV + base features + lagged features.
    """
    df = add_bollinger_bands(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_volume(df)
    df = add_price_features(df)
    df = _add_lags(df)
    return df


def build_features_all(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply build_features to every ticker in the data dict."""
    out = {}
    for ticker, df in data.items():
        log.info("Building features for %s...", ticker)
        out[ticker] = build_features(df)
    log.info("Feature engineering done for %s tickers.", len(out))
    return out
