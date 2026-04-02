# VolCast

An end-to-end machine learning web application that predicts **volatility regimes** for 50 S&P 500 tickers — classifying whether a stock's volatility is *expanding* or *contracting* — built with LightGBM, Optuna, and Streamlit.

---

## Overview

Most ML finance projects predict price direction, which is extremely noisy and nearly random. VolCast instead predicts **volatility regimes** — a signal that is far more persistent and actionable. High volatility periods cluster together, and knowing whether vol is expanding or contracting helps inform position sizing, entry timing, and risk management.

---

## Features

- **Interactive dashboard** with a personalized watchlist — add any of the 50 tickers and see live vol regime predictions with model confidence
- **Signal Explorer** — candlestick chart with Bollinger Bands, RSI, volume, and vol-expanding signal markers overlaid on the timeline
- **Live indicator panel** — RSI, MACD %, ATR %, Bollinger %B, Volume Ratio, vs SMA20, and model confidence updated in real time via yfinance
- **Model Performance page** — test-set metrics, 5-fold walk-forward cross-validation results, and top 25 feature importances
- **Animated entrance screen** with a Three.js particle globe

---

## Model

| Metric | Test Set | CV Mean (5-Fold) |
|---|---|---|
| Accuracy | 69.6% | 65.8% |
| Precision | 65.5% | 65.3% |
| Recall | 80.3% | 66.8% |
| F1 Score | 0.721 | 0.659 |
| ROC-AUC | 0.776 | 0.717 |

**Baseline:** ~50% (random). The model significantly outperforms chance on an inherently noisy task.

### Pipeline
- **Data:** 6 years of daily OHLCV data for 50 S&P 500 tickers via `yfinance`
- **Labeling:** Binary regime label — expanding (1) if realized volatility over the next 10 days exceeds a rolling threshold, contracting (0) otherwise
- **Features (100+):** Bollinger Bands, RSI, MACD, ATR, OBV, stochastic oscillator, volume ratios, log returns and lags, price vs SMA20/50/200, relative strength vs SPY and sector ETFs, market regime features
- **Model:** LightGBM classifier with SMOTE for class imbalance
- **Tuning:** Bayesian hyperparameter optimization via Optuna (100 trials, `TimeSeriesSplit` CV to prevent lookahead bias)

---

## Tech Stack

| Category | Tools |
|---|---|
| ML | LightGBM, XGBoost, scikit-learn, imbalanced-learn |
| Hyperparameter tuning | Optuna |
| Data | yfinance, pandas, numpy |
| Visualization | Plotly, Three.js |
| App | Streamlit |
| Other | joblib, pyarrow |

---

## Project Structure

```
bollinger-ml-engine/
├── app/
│   ├── dashboard.py               # Home page + watchlist
│   ├── styles.py                  # Global CSS theme
│   ├── utils.py                   # Shared helpers (model loading, feature inference)
│   └── pages/
│       ├── 01_signal_explorer.py  # Chart + indicators page
│       └── 02_model_performance.py# Metrics + feature importance page
├── src/
│   ├── data/
│   │   ├── fetcher.py             # yfinance data pipeline
│   │   ├── preprocessor.py        # OHLCV cleaning
│   │   └── labeler.py             # Volatility regime labeling
│   ├── features/
│   │   ├── pipeline.py            # Feature assembly + FEATURE_COLS definition
│   │   ├── bollinger.py           # Bollinger Band features
│   │   ├── momentum.py            # RSI, MACD, stochastic
│   │   ├── volatility.py          # ATR, realized vol
│   │   ├── volume.py              # OBV, volume ratio
│   │   ├── price.py               # Returns, SMA ratios
│   │   └── market.py              # SPY/sector relative strength
│   ├── models/
│   │   └── xgboost_model.py       # Model wrapper
│   └── training/
│       ├── trainer.py             # Training loop
│       ├── tuner.py               # Optuna tuning
│       └── evaluator.py           # Metrics + CV evaluation
├── config/
│   └── settings.py                # Tickers, sector map, paths
├── run_pipeline.py                # Full pipeline entry point
├── run_tuner.py                   # Hyperparameter tuning entry point
└── requirements.txt
```

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/qamar213/bollinger-ml-engine.git
cd bollinger-ml-engine

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (fetch data, engineer features, train model)
python run_pipeline.py

# Launch the app
streamlit run app/dashboard.py
```

---

## Author

**Qamar Syed** — Northeastern University
