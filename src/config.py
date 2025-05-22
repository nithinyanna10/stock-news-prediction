# src/config.py

from pathlib import Path

# Root paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PRICE_DIR = DATA_DIR / "raw_prices"
RAW_NEWS_DIR = DATA_DIR / "raw_news"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDING_DIR = DATA_DIR / "embeddings"
MODEL_DIR = BASE_DIR / "models"

# Stocks to track
TICKERS = ["AAPL", "TSLA", "JPM", "AMZN", "NVDA"]

# Time window settings (using days now)
PRICE_WINDOW_DAYS = 30        # past 30 trading days
NEWS_WINDOW_DAYS = 30         # news in the past 30 calendar days
PREDICT_FORWARD_DAYS = 5      # future prediction target

# Feature settings
USE_TECHNICAL_INDICATORS = True

# FinBERT embedding cache
EMBEDDING_DIM = 768

# Misc
SEED = 42
