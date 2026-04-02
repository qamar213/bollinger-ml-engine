"""
app/utils.py — shared helpers for all dashboard pages.

Handles: model loading, live data fetching, signal generation,
metrics loading, and feature building for real-time inference.
"""
import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM",   "V",    "UNH",
    "JNJ",  "WMT",  "XOM",   "PG",   "MA",
    "HD",   "CVX",  "MRK",   "ABBV", "PFE",
    "BAC",  "KO",   "PEP",   "AVGO", "COST",
    "MCD",  "TMO",  "CSCO",  "ACN",  "ABT",
    "NKE",  "DHR",  "NEE",   "LIN",  "TXN",
    "PM",   "ORCL", "BMY",   "RTX",  "AMGN",
    "QCOM", "HON",  "UPS",   "SBUX", "GS",
    "BLK",  "CAT",  "IBM",   "GE",   "MMM",
]


# ── Model ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved XGBoost model. Cached so it only loads once per session."""
    try:
        from src.models.xgboost_model import XGBoostModel
        path = ROOT / "models" / "lgbm_final.joblib"
        if not path.exists():
            return None
        return XGBoostModel.load(str(path))
    except Exception as e:
        log.warning("Could not load model: %s", e)
        return None


# ── Metrics ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def get_model_metrics() -> dict | None:
    path = ROOT / "results" / "test_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False, ttl=3600)
def get_feature_importance() -> pd.DataFrame | None:
    path = ROOT / "results" / "feature_importance.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


@st.cache_data(show_spinner=False, ttl=3600)
def get_cv_metrics() -> dict | None:
    path = ROOT / "results" / "cv_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Live data ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=900)  # cache 15 min — intraday is fine
def fetch_ticker_history(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """Download OHLCV history for a single ticker via yfinance."""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return None
        df.columns = df.columns.get_level_values(0)  # flatten MultiIndex if present
        return df
    except Exception as e:
        log.warning("yfinance error for %s: %s", ticker, e)
        return None


def _build_features_for_inference(df: pd.DataFrame, ticker: str, ticker_idx: int) -> pd.DataFrame | None:
    """
    Run the full feature pipeline on a live OHLCV DataFrame.
    Returns a single-row DataFrame with all FEATURE_COLS ready for the model.
    """
    try:
        from src.features.pipeline import build_features, FEATURE_COLS
        from src.features.market import build_market_features_all
        from config.settings import SECTOR_MAP

        featured = build_features(df.copy())

        # Market features need SPY for relative-strength columns
        spy_df = fetch_ticker_history("SPY", period="5y")
        ticker_map = {ticker: featured}
        if spy_df is not None:
            ticker_map["SPY"] = build_features(spy_df.copy())
        featured_dict = build_market_features_all(ticker_map)
        featured = featured_dict[ticker]

        # Add ticker identity columns
        featured["ticker_encoded"] = ticker_idx
        featured["sector"]         = SECTOR_MAP.get(ticker, -1)

        # Drop rows missing any feature
        featured = featured.dropna(subset=FEATURE_COLS)
        if featured.empty:
            return None

        return featured[FEATURE_COLS].iloc[[-1]]  # most recent row only
    except Exception as e:
        log.warning("Feature build failed for %s: %s", ticker, e)
        return None


@st.cache_data(show_spinner=False, ttl=900)
def get_live_signals() -> pd.DataFrame | None:
    """
    Run the model on today's live data for all 50 tickers.
    Returns a DataFrame of tickers predicted to enter an expanding volatility
    regime (label=1), with columns: ticker, proba, rsi, bb_pct_b, atr_pct.
    """
    model = load_model()
    if model is None:
        return None

    results = []
    ticker_index = {t: i for i, t in enumerate(TICKERS)}

    for ticker in TICKERS:
        df = fetch_ticker_history(ticker, period="2y")
        if df is None or len(df) < 250:
            continue

        X = _build_features_for_inference(df, ticker, ticker_index[ticker])
        if X is None:
            continue

        proba = model.predict_proba(X).iloc[0]
        pred  = model.predict(X).iloc[0]

        if pred == 1:
            row = X.iloc[0]
            results.append({
                "ticker":    ticker,
                "proba":     proba,
                "rsi":       row.get("rsi", float("nan")),
                "bb_pct_b":  row.get("bb_pct_b", float("nan")),
                "atr_pct":   row.get("atr_pct", float("nan")),
            })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("proba", ascending=False).reset_index(drop=True)
