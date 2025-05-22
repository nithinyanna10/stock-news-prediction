# src/embed_news.py

import os
import hashlib
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.config import EMBEDDING_DIR, EMBEDDING_DIM

# Load FinBERT (make sure it's only loaded once)
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")
model.eval()

EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

def headline_to_filename(headline: str) -> str:
    return hashlib.md5(headline.encode()).hexdigest() + ".pt"

def embed_headline(headline: str) -> torch.Tensor:
    inputs = tokenizer(headline, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)  # shape: [768]
    return output

def embed_and_cache(headline: str) -> torch.Tensor:
    filename = EMBEDDING_DIR / headline_to_filename(headline)
    if filename.exists():
        return torch.load(filename)
    else:
        vec = embed_headline(headline)
        torch.save(vec, filename)
        return vec

def process_news_df(news_df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="Embedding headlines")
    news_df["embedding"] = news_df["headline"].progress_apply(embed_and_cache)
    return news_df

if __name__ == "__main__":
    df = pd.read_csv("data/raw_news/news.csv")
    df = df.dropna(subset=["headline"])
    process_news_df(df)
    print(f"✅ Embeddings cached in {EMBEDDING_DIR}")
