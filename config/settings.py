from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_RAW_DIR    = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR   = ROOT_DIR / "data" / "processed"
DATA_SPLITS_DIR = ROOT_DIR / "data" / "splits"
MODELS_DIR      = ROOT_DIR / "models"
RESULTS_DIR     = ROOT_DIR / "results"

# Create directories if they don't exist, and fail fast if a file blocks the path.
for d in [DATA_RAW_DIR, DATA_PROC_DIR, DATA_SPLITS_DIR, MODELS_DIR, RESULTS_DIR]:
    if d.exists() and not d.is_dir():
        raise NotADirectoryError(
            f"Expected directory at {d}, but found a file. Remove or rename it and retry."
        )
    d.mkdir(parents=True, exist_ok=True)

# ── Universe of Tickers ──────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V",    "UNH",
    "JNJ",  "WMT",  "XOM",  "PG",  "MA",
    "HD",   "CVX",  "MRK",  "ABBV","PFE",
    "BAC",  "KO",   "PEP",  "AVGO","COST",
    "MCD",  "TMO",  "CSCO", "ACN", "ABT",
    "NKE",  "DHR",  "NEE",  "LIN", "TXN",
    "PM",   "ORCL", "BMY",  "RTX", "AMGN",
    "QCOM", "HON",  "UPS",  "SBUX","GS",
    "BLK",  "CAT",  "IBM",  "GE",  "MMM",
]

# ── Data Settings ────────────────────────────────────────────────────────────
START_DATE      = "2018-01-01"
END_DATE        = "2024-01-01"
PRICE_COL       = "Close"

# ── Labeling Settings ────────────────────────────────────────────────────────
FORWARD_WINDOW  = 5      # days ahead to measure return
BUY_THRESHOLD   = 0.05   # +5% = buy signal (label 1) — ~8% base rate, cleaner signal than 3%
SELL_THRESHOLD  = -0.02  # -2% = sell signal (label 0)

# ── Bollinger Band Settings ──────────────────────────────────────────────────
BB_WINDOW       = 20
BB_STD          = 2.0

# ── Other Indicator Settings ─────────────────────────────────────────────────
RSI_WINDOW      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
ATR_WINDOW      = 14
VOLUME_WINDOW   = 20

# ── Sector Map ───────────────────────────────────────────────────────────────
# Integer-encoded sector for each ticker. XGBoost can use this as a grouping
# signal — tickers in the same sector share macro and earnings cycle behaviour.
SECTOR_MAP = {
    # Technology
    "AAPL": 0, "MSFT": 0, "NVDA": 0, "AVGO": 0, "ORCL": 0,
    "CSCO": 0, "IBM":  0, "TXN":  0, "ACN":  0, "QCOM": 0,
    # Communication Services
    "GOOGL": 1, "META": 1,
    # Consumer Discretionary
    "AMZN": 2, "TSLA": 2, "HD": 2, "MCD": 2, "NKE": 2, "SBUX": 2,
    # Consumer Staples
    "WMT": 3, "PG": 3, "KO": 3, "PEP": 3, "COST": 3,
    # Financials
    "JPM": 4, "V": 4, "MA": 4, "BAC": 4, "GS": 4, "BLK": 4,
    # Health Care
    "UNH": 5, "JNJ": 5, "MRK": 5, "ABBV": 5, "PFE": 5,
    "TMO": 5, "DHR": 5, "ABT": 5, "AMGN": 5, "BMY": 5,
    # Industrials
    "HON": 6, "UPS": 6, "RTX": 6, "CAT": 6, "GE": 6, "MMM": 6,
    # Energy
    "XOM": 7, "CVX": 7,
    # Utilities
    "NEE": 8,
    # Materials
    "LIN": 9,
}

# ── Model Settings ───────────────────────────────────────────────────────────
TEST_SIZE       = 0.2    # 20% held out for final evaluation
N_SPLITS        = 5      # TimeSeriesSplit folds
RANDOM_STATE    = 42

# ── Training Thresholds ──────────────────────────────────────────────────────
MIN_SAMPLES     = 200    # skip tickers with fewer rows than this
