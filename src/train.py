# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data_loader import get_training_windows
from src.model import NewsBERTFusionModel
from src.embed_news import embed_and_cache
from src.feature_engineering import add_indicators, price_to_tensor
from src.config import TICKERS  # To loop over all tickers
import pandas as pd


class StockNewsDataset(Dataset):
    def __init__(self, data_points):
        self.data_points = data_points

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        _, aligned = self.data_points[idx]

        # Apply technical indicators and convert to tensor
        price_window = add_indicators(aligned["price_window"])
        price_tensor = price_to_tensor(price_window)

        # FinBERT embeddings from headlines
        news_embeddings = [
            embed_and_cache(row["headline"]) for _, row in aligned["news_window"].iterrows()
        ]
        news_embedding = (
            torch.stack(news_embeddings).mean(dim=0) if news_embeddings else torch.zeros(768)
        )

        # Target return (%)
        close_prices = price_window["close"]
        start_price = close_prices.iloc[0]
        end_price = close_prices.iloc[-1]

        if pd.isna(start_price) or start_price == 0:
            target_return = 0.0
        else:
            target_return = (end_price - start_price) / start_price

        return price_tensor, news_embedding, torch.tensor(target_return, dtype=torch.float32)


def train_model(ticker, epochs=5, batch_size=16, lr=1e-4, device="cpu"):
    print(f"\n⏳ Loading and aligning data for {ticker}")
    data_points = get_training_windows(ticker)

    if not data_points:
        print(f"⚠️  No valid training samples found for {ticker}")
        return

    dataset = StockNewsDataset(data_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NewsBERTFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for price_seq, news_emb, target in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            price_seq, news_emb, target = price_seq.to(device), news_emb.to(device), target.to(device)

            optimizer.zero_grad()
            pred = model(price_seq, news_emb)
            loss = loss_fn(pred, target)
            loss.backward()

            # ✅ Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1} completed. Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), f"models/{ticker}_model.pt")
    print(f"📦 Model saved for {ticker}")


if __name__ == "__main__":
    for ticker in TICKERS:
        train_model(ticker)
