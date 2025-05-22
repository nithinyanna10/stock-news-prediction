# src/data_loader.py

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
from src.config import PRICE_WINDOW_DAYS, RAW_PRICE_DIR, RAW_NEWS_DIR

# --------- Downloader ----------
def download_stock_data(tickers, start_date, end_date, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        data.reset_index(inplace=True)
        data.to_csv(f"{save_dir}/{ticker}.csv", index=False)
        print(f"Saved {ticker} data to {save_dir}/{ticker}.csv")

# --------- Loaders ----------
def load_price_data(ticker: str) -> pd.DataFrame:
    path = RAW_PRICE_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    
    # Lowercase and standardize
    df.rename(columns={
        "Close": "close",
        "Volume": "volume",
        "Open": "open",
        "High": "high",
        "Low": "low"
    }, inplace=True)

    df.set_index("timestamp", inplace=True)
    return df



def load_news_data() -> pd.DataFrame:
    path = Path("data/cleaned_news/news.csv")  # use your updated path
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)  # ✅ remove timezone
    return df


def align_data(price_df: pd.DataFrame, news_df: pd.DataFrame, target_time: pd.Timestamp) -> dict:
    start_price_time = target_time - timedelta(days=PRICE_WINDOW_DAYS)
    start_news_time = target_time - timedelta(days=PRICE_WINDOW_DAYS)

    try:
        price_window = price_df.loc[start_price_time:target_time]
    except KeyError:
        return {"price_window": pd.DataFrame(), "news_window": pd.DataFrame()}

    news_window = news_df[(news_df["timestamp"] >= start_news_time) & 
                          (news_df["timestamp"] <= target_time)]

    return {
        "price_window": price_window,
        "news_window": news_window
    }

def get_training_windows(ticker: str) -> list:
    price_df = load_price_data(ticker)
    news_df = load_news_data()
    news_df = news_df[news_df["ticker"] == ticker]

    timestamps = price_df.index[PRICE_WINDOW_DAYS:]
    data_points = []

    for ts in tqdm(timestamps, desc=f"Aligning data for {ticker}"):
        aligned = align_data(price_df, news_df, ts)
        if not aligned["price_window"].empty and not aligned["news_window"].empty:
            data_points.append((ts, aligned))

    return data_points
