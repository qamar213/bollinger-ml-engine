import pandas as pd

from config.settings import ATR_WINDOW

_ADX_WINDOW = 14


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds volatility features:
      atr            — Average True Range (ATR_WINDOW)
      atr_pct        — ATR as % of close price (normalised)
      adx            — Average Directional Index: trend strength (0-100, >25 = strong trend)
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    prev  = close.shift(1)

    true_range = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs(),
    ], axis=1).max(axis=1)

    atr = true_range.rolling(ATR_WINDOW).mean()

    # ADX — measures trend strength regardless of direction.
    # High ADX = strong trend (either up or down), useful for confirming momentum signals.
    # Computed from directional movement indicators (DM+ and DM-).
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    dm_pos = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_neg = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr_smooth  = true_range.ewm(alpha=1 / _ADX_WINDOW, min_periods=_ADX_WINDOW, adjust=False).mean()
    di_pos = 100 * dm_pos.ewm(alpha=1 / _ADX_WINDOW, min_periods=_ADX_WINDOW, adjust=False).mean() / atr_smooth.replace(0, float("nan"))
    di_neg = 100 * dm_neg.ewm(alpha=1 / _ADX_WINDOW, min_periods=_ADX_WINDOW, adjust=False).mean() / atr_smooth.replace(0, float("nan"))
    dx  = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, float("nan"))
    adx = dx.ewm(alpha=1 / _ADX_WINDOW, min_periods=_ADX_WINDOW, adjust=False).mean()

    out = df.copy()
    out["atr"]     = atr
    out["atr_pct"] = atr / close
    out["adx"]     = adx
    return out
