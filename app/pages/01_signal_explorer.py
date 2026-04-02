"""
pages/01_signal_explorer.py — Signal Explorer

Pick any of the 50 tickers and see:
  · Candlestick chart with Bollinger Bands overlaid
  · Vol-expanding signal markers on the timeline
  · Current indicator readings (RSI, MACD, ATR, OBV)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.styles import GLOBAL_CSS
from app.utils import TICKERS, fetch_ticker_history, load_model, _build_features_for_inference

st.set_page_config(page_title="Signal Explorer · VolCast", page_icon="📈", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Top nav ───────────────────────────────────────────────────────────────────
st.markdown('<div class="topnav-wrap">', unsafe_allow_html=True)
nav_logo, nav_home, nav_sig, nav_perf, nav_pad = st.columns([2, 1, 1.3, 1.5, 3])
with nav_logo:
    st.markdown('<span class="topnav-logo">VolCast</span>', unsafe_allow_html=True)
with nav_home:
    st.markdown('<a href="/?e=1" target="_self" class="nav-plain-link">Dashboard</a>', unsafe_allow_html=True)
with nav_sig:
    st.markdown('<a href="/signal_explorer" target="_self" class="nav-plain-link">Signal Explorer</a>', unsafe_allow_html=True)
with nav_perf:
    st.markdown('<a href="/model_performance" target="_self" class="nav-plain-link">Model Performance</a>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Page heading + controls ───────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <p class="hero-title">Signal Explorer</p>
  <p class="hero-sub">Bollinger Bands · Vol Signals · Indicators</p>
</div>
""", unsafe_allow_html=True)

ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
with ctrl1:
    ticker = st.selectbox("Ticker", TICKERS, index=0)
with ctrl2:
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2)
with ctrl3:
    show_vol = st.checkbox("Show Volume", value=True)

with st.spinner(f"Loading {ticker}..."):
    df = fetch_ticker_history(ticker, period=period)
    df_full = fetch_ticker_history(ticker, period="5y")  # always 5y for feature computation

if df is None or df.empty:
    st.error(f"Could not fetch data for {ticker}.")
    st.stop()
if df_full is None or df_full.empty:
    df_full = df

# ── Build features for signal markers ────────────────────────────────────────
model = load_model()
ticker_index = {t: i for i, t in enumerate(TICKERS)}

@st.cache_data(show_spinner=False, ttl=900)
def get_signals_for_chart(ticker: str) -> pd.Series | None:
    """Return a boolean series of buy signals for the full history."""
    _df = fetch_ticker_history(ticker, period="5y")  # always 5y for enough feature history
    if _df is None:
        return None
    try:
        from src.features.pipeline import build_features, FEATURE_COLS
        from src.features.market import build_market_features_all
        from config.settings import SECTOR_MAP

        _spy = fetch_ticker_history("SPY", period="5y")
        ticker_map = {ticker: build_features(_df.copy())}
        if _spy is not None:
            ticker_map["SPY"] = build_features(_spy.copy())
        featured_dict = build_market_features_all(ticker_map)
        featured = featured_dict[ticker]
        featured["ticker_encoded"] = ticker_index[ticker]
        featured["sector"]         = SECTOR_MAP.get(ticker, -1)
        featured = featured.dropna(subset=FEATURE_COLS)
        if featured.empty:
            return None
        X = featured[FEATURE_COLS]
        return model.predict(X) if model else None
    except Exception:
        return None

signals = get_signals_for_chart(ticker)

# ── Indicator panel values (most recent row) ──────────────────────────────────
try:
    from src.features.pipeline import build_features, FEATURE_COLS
    from src.features.market import build_market_features_all
    from config.settings import SECTOR_MAP

    spy_df = fetch_ticker_history("SPY", period="5y")
    ticker_map = {ticker: build_features(df_full.copy())}
    if spy_df is not None:
        ticker_map["SPY"] = build_features(spy_df.copy())

    featured_full_dict = build_market_features_all(ticker_map)
    featured_full = featured_full_dict[ticker]
    featured_full["ticker_encoded"] = ticker_index[ticker]
    featured_full["sector"]         = SECTOR_MAP.get(ticker, -1)
    featured_full = featured_full.dropna(subset=FEATURE_COLS)
    last = featured_full.iloc[-1] if not featured_full.empty else None
except Exception as e:
    import traceback; traceback.print_exc()
    last = None

# ── Layout: chart left, indicators right ─────────────────────────────────────
chart_col, info_col = st.columns([3, 1])

with info_col:
    st.markdown('<p class="section-heading">Indicators</p>', unsafe_allow_html=True)
    if last is not None:
        def card(label, value, fmt=".2f", color="#1a1a1a"):
            display = f"{value:{fmt}}" if not np.isnan(float(value)) else "—"
            st.markdown(f"""
            <div class="ind-card">
              <span class="ind-label">{label}</span>
              <span class="ind-value" style="color:{color}">{display}</span>
            </div>""", unsafe_allow_html=True)

        rsi = last.get("rsi", float("nan"))
        rsi_color = "#cc3333" if rsi > 70 else "#2d8a4e" if rsi < 30 else "#1a1a1a"
        card("RSI (14)", rsi, ".1f", rsi_color)
        card("MACD %", last.get("macd_pct", float("nan")), ".4f")
        card("ATR %", last.get("atr_pct", float("nan")), ".3f")
        card("BB %B", last.get("bb_pct_b", float("nan")), ".3f")
        card("Volume Ratio", last.get("volume_ratio", float("nan")), ".2f")
        card("vs SMA20", last.get("close_to_sma20", float("nan")), ".3f")

        st.markdown("<br>", unsafe_allow_html=True)

        # Model confidence on latest bar
        if model and not featured_full.empty:
            X_last = featured_full[FEATURE_COLS].iloc[[-1]]
            prob = model.predict_proba(X_last).iloc[0]
            pred = model.predict(X_last).iloc[0]
            badge_class = "badge-buy" if pred == 1 else "badge-none"
            signal_text = "VOL EXPANDING" if pred == 1 else "CONTRACTING"
            border_color = "rgba(63,185,80,0.4)" if pred == 1 else "rgba(139,148,158,0.2)"
            conf_color   = "#2d8a4e" if pred == 1 else "#1a1a1a"
            st.markdown(f"""
            <div class="conf-card" style="border-color:{border_color}">
              <div class="conf-value" style="color:{conf_color}">{prob:.1%}</div>
              <div class="conf-label">Model Confidence</div>
              <span class="conf-badge {badge_class}">{signal_text}</span>
            </div>""", unsafe_allow_html=True)

    else:
        st.info("Indicators unavailable.")

with chart_col:
    st.markdown(f'<p class="section-heading">{ticker} · Price + Bollinger Bands</p>', unsafe_allow_html=True)
    # ── Build Plotly chart ────────────────────────────────────────────────────
    rows = 3 if show_vol else 2
    row_heights = [0.6, 0.2, 0.2] if show_vol else [0.7, 0.3]
    subplot_titles = [f"{ticker} Price + Bollinger Bands", "RSI", "Volume"] if show_vol else [f"{ticker} Price + Bollinger Bands", "RSI"]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Bollinger bands (compute on df directly for the chart)
    close = df["Close"].squeeze()
    mid   = close.rolling(20).mean()
    std   = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std

    # BB fill
    fig.add_trace(go.Scatter(
        x=df.index, y=upper, line=dict(color="rgba(0,212,255,0.3)", width=1),
        name="BB Upper", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=lower, line=dict(color="rgba(0,212,255,0.3)", width=1),
        fill="tonexty", fillcolor="rgba(0,212,255,0.05)",
        name="BB Lower", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=mid, line=dict(color="rgba(0,212,255,0.5)", width=1, dash="dot"),
        name="BB Mid", showlegend=False,
    ), row=1, col=1)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(),   close=close,
        increasing_line_color="#00ff88", decreasing_line_color="#ff4b4b",
        name=ticker,
    ), row=1, col=1)

    # Buy signal markers — filter to the visible chart period only
    if signals is not None:
        visible_signals = signals[signals.index.isin(df.index)]
        buy_dates  = visible_signals[visible_signals == 1].index
        buy_prices = df.loc[df.index.isin(buy_dates), "Low"].squeeze() * 0.985
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="#00ff88"),
            name="Vol Expanding",
        ), row=1, col=1)

    # RSI — always compute from close, no feature pipeline needed
    try:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rsi_series = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi_series,
            line=dict(color="#7b2fff", width=1.5), name="RSI",
        ), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="#ff4b4b", dash="dash", width=0.8), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="#00ff88", dash="dash", width=0.8), row=2, col=1)
    except Exception:
        pass

    # Volume
    if show_vol:
        vol_colors = ["#00ff88" if c >= o else "#ff4b4b"
                      for c, o in zip(df["Close"].squeeze(), df["Open"].squeeze())]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"].squeeze(),
            marker_color=vol_colors, name="Volume", showlegend=False,
        ), row=3, col=1)

    fig.update_layout(
        height=680,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#aaa", size=12),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=30, t=40, b=20),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#21262d", showgrid=True)
    fig.update_yaxes(gridcolor="#21262d", showgrid=True)

    st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Volatility regime predictions by XGBoost trained on 6 years of daily OHLCV data across 50 S&P 500 tickers.
  &nbsp;·&nbsp; Not financial advice.
</div>""", unsafe_allow_html=True)
