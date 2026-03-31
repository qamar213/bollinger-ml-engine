import numpy as np
import pandas as pd


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds price-derived features:
      log_return      — daily log return
      ret_5d          — 5-day cumulative log return
      ret_10d         — 10-day cumulative log return
      ret_20d         — 20-day cumulative log return
      hl_range        — (High - Low) / Close  — intraday range normalised
      gap             — (Open - prev Close) / prev Close  — overnight gap
      close_to_sma20  — close / 20-day SMA - 1  (short-term trend position)
      close_to_sma50  — close / 50-day SMA - 1  (medium-term trend position)
      close_to_sma200 — close / 200-day SMA - 1 (long-term trend position per ticker)
      sma20_slope     — (SMA20 - SMA20.shift(5)) / SMA20.shift(5)  (trend momentum)
      dist_52w_high   — close / 252-day high - 1  (how far below the yearly high, negative = drawdown)
      dist_52w_low    — close / 252-day low - 1   (how far above the yearly low, positive = recovery)
    """
    close = df["Close"]
    log_ret = np.log(close / close.shift(1))

    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    out = df.copy()
    out["log_return"]      = log_ret
    out["ret_5d"]          = log_ret.rolling(5).sum()
    out["ret_10d"]         = log_ret.rolling(10).sum()
    out["ret_20d"]         = log_ret.rolling(20).sum()
    out["hl_range"]        = (df["High"] - df["Low"]) / close
    out["gap"]             = (df["Open"] - close.shift(1)) / close.shift(1)
    out["close_to_sma20"]  = close / sma20 - 1
    out["close_to_sma50"]  = close / sma50 - 1
    out["close_to_sma200"] = close / sma200 - 1
    out["sma20_slope"]     = (sma20 - sma20.shift(5)) / sma20.shift(5)
    # Distance from 52-week high/low — anchoring momentum signal.
    # Stocks near 52w highs tend to continue (momentum); stocks near 52w lows
    # are potential mean-reversion buys if combined with oversold signals.
    out["dist_52w_high"]   = close / close.rolling(252).max() - 1
    out["dist_52w_low"]    = close / close.rolling(252).min() - 1

    # Medium and long-term momentum — among the most robust equity return predictors
    # (Jegadeesh & Titman 1993, Carhart 1997).
    out["ret_63d"]  = log_ret.rolling(63).sum()   # 3-month
    out["ret_126d"] = log_ret.rolling(126).sum()  # 6-month
    # 12-1 month momentum: skip last 21 days to avoid short-term reversal.
    # ret from t-252 to t-21 = log(close[t-21] / close[t-252])
    out["mom_12_1"] = np.log(close.shift(21) / close.shift(252))
    return out
