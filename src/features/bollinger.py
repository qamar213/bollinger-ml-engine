import pandas as pd

from config.settings import BB_WINDOW, BB_STD


def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Bollinger Band features:
      bb_upper, bb_lower, bb_mid  — raw band levels
      bb_pct_b                    — %B: where price sits within the bands (0=lower, 1=upper)
      bb_bandwidth                — (upper - lower) / mid, normalised width
      bb_squeeze                  — 1 if bandwidth is in its lowest 20th percentile over 50 days,
                                    indicating a volatility compression (Bollinger squeeze) that
                                    often precedes a directional breakout
    """
    close = df["Close"]
    mid   = close.rolling(BB_WINDOW).mean()
    std   = close.rolling(BB_WINDOW).std()
    upper = mid + BB_STD * std
    lower = mid - BB_STD * std

    out = df.copy()
    out["bb_upper"]     = upper
    out["bb_lower"]     = lower
    out["bb_mid"]       = mid
    band_width = (upper - lower).replace(0, float("nan"))  # guard: flat markets collapse bands to zero
    out["bb_pct_b"]     = (close - lower) / band_width
    bandwidth = (upper - lower) / mid
    out["bb_bandwidth"] = bandwidth
    # Squeeze: bandwidth percentile rank over 50-day rolling window (0=tightest ever, 1=widest ever)
    out["bb_squeeze"]   = bandwidth.rolling(50).rank(pct=True)
    return out
