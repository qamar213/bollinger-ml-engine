from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_RAW_DIR    = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR   = ROOT_DIR / "data" / "processed"
DATA_SPLITS_DIR = ROOT_DIR / "data" / "splits"
MODELS_DIR      = ROOT_DIR / "models"
RESULTS_DIR     = ROOT_DIR / "results"

# Create directories if they don't exist
for d in [DATA_RAW_DIR, DATA_PROC_DIR, DATA_SPLITS_DIR, MODELS_DIR, RESULTS_DIR]:
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
BUY_THRESHOLD   = 0.02   # +2% = buy signal (label 1)
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

# ── Model Settings ───────────────────────────────────────────────────────────
TEST_SIZE       = 0.2    # 20% held out for final evaluation
N_SPLITS        = 5      # TimeSeriesSplit folds
RANDOM_STATE    = 42

# ── Training Thresholds ──────────────────────────────────────────────────────
MIN_SAMPLES     = 200    # skip tickers with fewer rows than this