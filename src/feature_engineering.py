import pandas as pd
import ta  # technical analysis library
import torch  # ✅ Add this line

import pandas as pd
import ta
import torch

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ✅ Ensure numeric conversion (especially from Yahoo CSVs)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")

    # ✅ Add technical indicators
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()
    df["macd"] = ta.trend.MACD(close=df["close"]).macd()
    df["volatility"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"]
    ).average_true_range()

    # ✅ Fill missing values safely
    df = df.bfill().ffill()


    return df

def price_to_tensor(price_window: pd.DataFrame, fixed_len=30) -> torch.Tensor:
    features = ["close", "volume", "rsi", "macd", "volatility"]
    df = price_window[features].copy()

    # Padding or truncating
    if len(df) < fixed_len:
        pad_rows = pd.DataFrame(0, index=range(fixed_len - len(df)), columns=features)
        df = pd.concat([pad_rows, df], ignore_index=True)
    elif len(df) > fixed_len:
        df = df.iloc[-fixed_len:]

    return torch.tensor(df.values, dtype=torch.float32)
