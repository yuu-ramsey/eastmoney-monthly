"""MASTER-inspired: Cross-stock attention with market context.
Key ideas from MASTER (SJTU-DMTai):
- Market context vector guides stock-level predictions
- Cross-asset attention learns relative rankings
- Multi-task heads for 1m/3m/6m returns
"""
import torch, torch.nn as nn, numpy as np

class MarketContextEncoder(nn.Module):
    """Encode market-level information: HS300 avg return, volatility, trend"""
    def __init__(self, input_dim=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 16))
    def forward(self, market_features):
        return self.net(market_features)  # (batch, 16)

class StockEncoder(nn.Module):
    """Encode individual stock features with LSTM + attention"""
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, 2, batch_first=True, dropout=0.2)
        self.attn = nn.Linear(hidden, 1)
    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        context = (w * out).sum(dim=1)  # (batch, hidden)
        return context

class CrossAssetAttention(nn.Module):
    """Multi-head attention across stocks in the same month (cross-section)"""
    def __init__(self, hidden=64, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.1)
        self.ln = nn.LayerNorm(hidden)
    def forward(self, stock_embeddings):
        # stock_embeddings: (batch, hidden) — all stocks in one month
        if stock_embeddings.dim() == 2:
            stock_embeddings = stock_embeddings.unsqueeze(0)  # (1, N_stocks, hidden)
        out, _ = self.attn(stock_embeddings, stock_embeddings, stock_embeddings)
        return self.ln(out + stock_embeddings)

class MASTERBaseline(nn.Module):
    """Simplified MASTER: Market context + Stock LSTM + Cross-attention + Multi-task heads"""
    def __init__(self, stock_input_dim=21, market_input_dim=5, hidden=64, heads=4):
        super().__init__()
        self.market_encoder = MarketContextEncoder(market_input_dim, 32)
        self.stock_encoder = StockEncoder(stock_input_dim, hidden)
        self.cross_attn = CrossAssetAttention(hidden, heads)
        self.market_proj = nn.Linear(16, hidden)  # project market context to hidden dim

        # Multi-task heads: 1m, 3m, 6m
        self.head_1m = nn.Sequential(nn.Linear(hidden*2, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
        self.head_3m = nn.Sequential(nn.Linear(hidden*2, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
        self.head_6m = nn.Sequential(nn.Linear(hidden*2, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, stock_x, market_x):
        """
        stock_x: (batch, seq_len, stock_features)
        market_x: (batch, market_features)
        Only processes one stock at a time — cross-attention added externally.
        """
        stock_emb = self.stock_encoder(stock_x)  # (batch, hidden=64)
        market_emb = self.market_proj(self.market_encoder(market_x))  # (batch, hidden=64)
        combined = torch.cat([stock_emb, market_emb], dim=-1)  # (batch, 128)

        y1 = self.head_1m(combined)
        y3 = self.head_3m(combined)
        y6 = self.head_6m(combined)
        return torch.cat([y1, y3, y6], dim=-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
