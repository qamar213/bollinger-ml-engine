import logging
import yfinance as yf
import pandas as pd
from pathlib import Path
from config.settings import (
    TICKERS, START_DATE, END_DATE, DATA_RAW_DIR, MIN_SAMPLES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Download OHLCV data for a single ticker. Returns None on failure."""
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty or len(df) < MIN_SAMPLES:
            log.warning(f"{ticker}: insufficient data ({len(df)} rows), skipping.")
            return None
        df.index = pd.to_datetime(df.index)
        df.dropna(inplace=True)
        log.info(f"{ticker}: fetched {len(df)} rows.")
        return df
    except Exception as e:
        log.error(f"{ticker}: fetch failed — {e}")
        return None

# python3 -m pip install pandas-stubs
def fetch_all(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all tickers.
    Returns a dict mapping ticker -> DataFrame.
    Optionally saves each ticker as a parquet file in DATA_RAW_DIR.
    """
    data = {}
    for ticker in tickers:
        df = fetch_ticker(ticker, start, end)
        if df is not None:
            data[ticker] = df
            if save:
                path = DATA_RAW_DIR / f"{ticker}.parquet"
                df.to_parquet(path)
                log.info(f"{ticker}: saved to {path}")

    log.info(f"Fetched {len(data)}/{len(tickers)} tickers successfully.")
    return data


def load_ticker(ticker: str) -> pd.DataFrame | None:
    """Load a previously saved ticker from parquet."""
    path = DATA_RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        log.warning(f"{ticker}: no saved data found at {path}.")
        return None
    return pd.read_parquet(path)


def load_all(tickers: list[str] = TICKERS) -> dict[str, pd.DataFrame]:
    """Load all saved tickers from parquet files."""
    data = {}
    for ticker in tickers:
        df = load_ticker(ticker)
        if df is not None:
            data[ticker] = df
    log.info(f"Loaded {len(data)}/{len(tickers)} tickers from disk.")
    return data


if __name__ == "__main__":
    log.info("Starting data fetch for all tickers...")
    fetch_all()
    log.info("Done.")