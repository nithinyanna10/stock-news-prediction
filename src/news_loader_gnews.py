# src/news_loader_gnews.py

from gnews import GNews
import pandas as pd
from datetime import datetime
from pathlib import Path
import time

def fetch_gnews_data(tickers, max_articles_per_ticker=100):
    gnews = GNews()
    gnews.max_results = max_articles_per_ticker
    gnews.period = '90d'  # up to 3 months
    gnews.language = 'en'
    gnews.country = 'US'

    all_articles = []

    for ticker in tickers:
        print(f"Fetching news for {ticker}...")
        results = gnews.get_news(ticker)
        for article in results:
            all_articles.append({
                'timestamp': article['published date'],
                'ticker': ticker,
                'headline': article['title']
            })
        time.sleep(1)  # avoid throttling

    df = pd.DataFrame(all_articles)
    df = df.dropna().drop_duplicates().sort_values('timestamp')
    return df

def save_news_csv(df, path="data/raw_news/news.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ News headlines saved to {path}")

if __name__ == "__main__":
    tickers = ["AAPL", "TSLA", "AMZN", "JPM", "NVDA"]
    df_news = fetch_gnews_data(tickers, max_articles_per_ticker=100)
    save_news_csv(df_news)
