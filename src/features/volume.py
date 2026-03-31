import numpy as np
import pandas as pd

from config.settings import VOLUME_WINDOW

_MFI_WINDOW = 14


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds volume features:
      volume_ma      — rolling mean volume (VOLUME_WINDOW)
      volume_ratio   — today's volume / rolling mean (surge detector)
      obv            — On-Balance Volume: cumulative signed volume
      obv_ema        — 20-period EMA of OBV (smoothed trend)
      obv_divergence — OBV minus its EMA (centred around 0)
      mfi            — Money Flow Index: volume-weighted RSI (0-100)
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    vol    = df["Volume"]
    vol_ma = vol.rolling(VOLUME_WINDOW).mean()

    # OBV: add volume on up days, subtract on down days
    direction = np.sign(close.diff())
    obv       = (vol * direction).cumsum()
    obv_ema   = obv.ewm(span=VOLUME_WINDOW, adjust=False).mean()

    # Money Flow Index — volume-weighted RSI.
    # Typical price captures the average price of the bar; multiplied by volume gives
    # raw money flow. Separating into positive (up days) vs negative (down days) and
    # taking a ratio gives a 0-100 oscillator that confirms price moves with volume.
    typical_price = (high + low + close) / 3
    raw_mf        = typical_price * vol
    tp_diff       = typical_price.diff()
    pos_mf = raw_mf.where(tp_diff > 0, 0.0).rolling(_MFI_WINDOW).sum()
    neg_mf = raw_mf.where(tp_diff < 0, 0.0).rolling(_MFI_WINDOW).sum()
    mfi = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, float("nan"))))

    out = df.copy()
    out["volume_ma"]      = vol_ma
    out["volume_ratio"]   = vol / vol_ma
    out["obv"]            = obv
    out["obv_ema"]        = obv_ema
    out["obv_divergence"] = obv - obv_ema
    out["mfi"]            = mfi
    return out
