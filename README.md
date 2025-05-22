# 📈 Stock News Prediction System

This project builds a deep learning pipeline to predict stock returns by combining:

- 📉 Historical price data using an LSTM
- 📰 Real-time news headlines using FinBERT embeddings
- 🔗 A late fusion model to forecast short-term stock movement

---

## 🔧 Features

- ✅ Loads OHLCV data using `yfinance`
- ✅ Collects and embeds news headlines using FinBERT
- ✅ Aligns price + news into training sequences
- ✅ Trains an LSTM + Dense Fusion model per stock
- ✅ Includes evaluation metrics (MSE, MAE, R²)
- ✅ Ready for real-time prediction and dashboard deployment

---

## 🗂️ Folder Structure

stock-news-predictor/
│
├── data/
│ ├── raw_prices/ # CSVs from yfinance
│ ├── raw_news/ # Raw GNews or NewsAPI headlines
│ ├── cleaned_news/ # Preprocessed news (timestamps fixed)
│ └── embeddings/ # Cached FinBERT embeddings
│
├── models/ # Saved model checkpoints
│
├── src/
│ ├── config.py # Global config and constants
│ ├── data_loader.py # Price & news loader and aligner
│ ├── feature_engineering.py # Technical indicators + tensor conversion
│ ├── embed_news.py # FinBERT headline embedding
│ ├── model.py # Late fusion model definition
│ ├── train.py # Training loop
│ ├── evaluate.py # Metrics on validation set
│ └── predict.py # [Coming soon] real-time return forecasting
│
└── README.md

yaml
Copy
Edit

---

## 🚀 Training

```bash
python -m src.train
Trains one model per stock in TICKERS using aligned price + news data. Outputs saved in models/.

📊 Evaluation
bash
Copy
Edit
python -m src.evaluate
Prints MSE, MAE, and R² scores on the last 20% of aligned sequences per stock.

📡 Prediction & Dashboard
Coming soon:

predict.py: Real-time inference on latest news + prices

app.py: Streamlit dashboard with interactive forecasts

📦 Requirements
bash
Copy
Edit
pip install -r requirements.txt
Includes:

torch

transformers

ta

yfinance

pandas

tqdm

scikit-learn

streamlit (for future dashboard)
