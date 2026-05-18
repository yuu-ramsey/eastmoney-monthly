"""Phase 17 LSTM Baseline: LSTM(input=21, hidden=64, 1 layer, dropout=0.2) → FC → (y3, y6)"""
import torch
import torch.nn as nn
import numpy as np

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)  # y3, y6

    def forward(self, x):
        # x: (batch, seq_len=60, features=21)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        out = self.dropout(last_hidden)
        out = self.fc(out)  # (batch, 2)
        return out


class StockDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
