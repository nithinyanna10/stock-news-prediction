# src/evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

from src.data_loader import get_training_windows
from src.feature_engineering import add_indicators, price_to_tensor
from src.embed_news import embed_and_cache
from src.model import NewsBERTFusionModel
from src.config import TICKERS


class EvaluationDataset(Dataset):
    def __init__(self, data_points):
        self.data_points = data_points

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        _, aligned = self.data_points[idx]
        price_window = add_indicators(aligned["price_window"])
        price_tensor = price_to_tensor(price_window)

        news_embeddings = [embed_and_cache(row["headline"]) for _, row in aligned["news_window"].iterrows()]
        news_embedding = torch.stack(news_embeddings).mean(dim=0) if news_embeddings else torch.zeros(768)

        close_prices = price_window["close"]
        start_price = close_prices.iloc[0]
        end_price = close_prices.iloc[-1]

        if pd.isna(start_price) or start_price == 0:
            target_return = 0.0
        else:
            target_return = (end_price - start_price) / start_price

        # 🔍 Debug logs
        if price_tensor.isnan().any():
            print(f"⚠️ NaN in price tensor at index {idx}")
        if news_embedding.isnan().any():
            print(f"⚠️ NaN in news embedding at index {idx}")
        if pd.isna(target_return):
            print(f"⚠️ NaN in target return at index {idx}")

        return price_tensor, news_embedding, torch.tensor(target_return, dtype=torch.float32)


def evaluate_model(ticker, model_path, device="cpu"):
    data_points = get_training_windows(ticker)
    if len(data_points) < 10:
        print(f"⚠️ Not enough data for {ticker} — skipping.")
        return

    split_idx = int(len(data_points) * 0.8)
    val_points = data_points[split_idx:]
    val_dataset = EvaluationDataset(val_points)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = NewsBERTFusionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, targets = []

    with torch.no_grad():
        for idx, (price_seq, news_emb, target) in enumerate(tqdm(val_loader, desc=f"Evaluating {ticker}")):
            price_seq, news_emb = price_seq.to(device), news_emb.to(device)
            output = model(price_seq, news_emb).squeeze().item()
            target_val = target.item()

            # 🔍 Log model predictions and targets
            print(f"[{ticker}][{idx}] ➜ output: {output:.4f}, target: {target_val:.4f}")

            preds.append(output)
            targets.append(target_val)

    if len(preds) == 0:
        print(f"⚠️ No valid predictions for {ticker}")
        return

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    print(f"\n📊 Evaluation for {ticker}:")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    print(f"   R² : {r2:.6f}")


if __name__ == "__main__":
    for ticker in TICKERS:
        model_path = f"models/{ticker}_model.pt"
        try:
            evaluate_model(ticker, model_path)
        except Exception as e:
            print(f"❌ Failed for {ticker}: {e}")
