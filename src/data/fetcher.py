import logging

import pandas as pd
import yfinance as yf
from yfinance import cache as yf_cache

from config.settings import ROOT_DIR, TICKERS, START_DATE, END_DATE, DATA_RAW_DIR, MIN_SAMPLES


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


YF_CACHE_DIR = ROOT_DIR / ".cache" / "yfinance"
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf_cache.set_cache_location(str(YF_CACHE_DIR))
yf.set_tz_cache_location(str(YF_CACHE_DIR))


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Download OHLCV data for a single ticker. Returns None on failure."""
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty or len(df) < MIN_SAMPLES:
            log.warning(
                "%s: insufficient data (%s rows, minimum %s), skipping.",
                ticker,
                len(df),
                MIN_SAMPLES,
            )
            return None

        df.index = pd.to_datetime(df.index)
        df.dropna(inplace=True)
        log.info("%s: fetched %s rows.", ticker, len(df))
        return df
    except Exception as e:
        log.error("%s: fetch failed - %s", ticker, e)
        return None


def fetch_all(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all tickers.
    Returns a dict mapping ticker -> DataFrame.
    Optionally saves each ticker as a csv file in DATA_RAW_DIR.
    """
    data = {}
    skipped = []

    for ticker in tickers:
        df = fetch_ticker(ticker, start, end)
        if df is None:
            skipped.append(ticker)
            continue

        data[ticker] = df
        if save:
            path = DATA_RAW_DIR / f"{ticker}.csv"
            df.to_csv(path)
            log.info("%s: saved to %s", ticker, path)

    log.info("Fetched %s/%s tickers successfully.", len(data), len(tickers))
    if skipped:
        log.warning("Skipped tickers: %s", ", ".join(skipped))
    return data


def load_ticker(ticker: str) -> pd.DataFrame | None:
    """Load a previously saved ticker from csv."""
    path = DATA_RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        log.warning("%s: no saved data found at %s.", ticker, path)
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_all(tickers: list[str] = TICKERS) -> dict[str, pd.DataFrame]:
    """Load all saved tickers from csv files."""
    data = {}
    for ticker in tickers:
        df = load_ticker(ticker)
        if df is not None:
            data[ticker] = df
    log.info("Loaded %s/%s tickers from disk.", len(data), len(tickers))
    return data


if __name__ == "__main__":
    log.info("Starting data fetch for all tickers...")
    fetch_all()
    log.info("Done.")
