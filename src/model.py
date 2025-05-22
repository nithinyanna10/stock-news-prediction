# src/model.py

import torch
import torch.nn as nn

class PriceLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, 64)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.output(out[:, -1, :])  # Last time step

class NewsBERTFusionModel(nn.Module):
    def __init__(self, price_input_dim=5, lstm_hidden_dim=64, news_dim=768):
        super().__init__()
        self.price_lstm = PriceLSTM(price_input_dim, lstm_hidden_dim)
        self.fc_news = nn.Linear(news_dim, 64)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Regression head
        )

    def forward(self, price_seq, news_embedding):
        price_out = self.price_lstm(price_seq)
        news_out = self.fc_news(news_embedding)
        combined = torch.cat([price_out, news_out], dim=1)
        return self.fusion(combined).squeeze(1)
