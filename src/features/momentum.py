import pandas as pd

from config.settings import RSI_WINDOW, MACD_FAST, MACD_SLOW, MACD_SIGNAL

_STOCH_WINDOW = 14
_STOCH_D_SMOOTH = 3


def _rsi(series: pd.Series, window: int) -> pd.Series:
    """Wilder's RSI — uses exponential smoothing (alpha=1/window), not simple mean."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds momentum features:
      rsi            — Relative Strength Index (RSI_WINDOW)
      macd           — MACD line (fast EMA - slow EMA)
      macd_signal    — 9-period EMA of MACD
      macd_hist      — MACD histogram (macd - macd_signal)
      stoch_k        — Stochastic %K: close position within n-day high/low range
      stoch_d        — Stochastic %D: 3-period SMA of %K (signal line)
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd     = ema_fast - ema_slow
    signal   = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()

    # Stochastic oscillator — measures where close sits within the recent high/low range.
    # Values near 0: close is near period low (oversold signal)
    # Values near 100: close is near period high (overbought)
    low_n   = low.rolling(_STOCH_WINDOW).min()
    high_n  = high.rolling(_STOCH_WINDOW).max()
    stoch_k = (close - low_n) / (high_n - low_n).replace(0, float("nan")) * 100
    stoch_d = stoch_k.rolling(_STOCH_D_SMOOTH).mean()

    out = df.copy()
    out["rsi"]          = _rsi(close, RSI_WINDOW)
    out["macd"]         = macd
    out["macd_signal"]  = signal
    out["macd_hist"]    = macd - signal
    out["macd_pct"]         = macd   / close
    out["macd_signal_pct"]  = signal / close
    out["stoch_k"]      = stoch_k
    out["stoch_d"]      = stoch_d
    return out
