"""Phase 17 v2: 3 architecture comparison (LSTM-2, GRU-2, Transformer-2)"""
import torch, torch.nn as nn
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / '.eastmoney-ai' / 'lstm'
MODEL_DIR = DATA_DIR / 'models_v2'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Architecture 1: LSTM-2 + Attention
# ============================================================
class LSTMAttention(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2)  # y3, y6
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (B, 60, H)
        # Scaled dot-product attention over time
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)
        context = (attn_w * lstm_out).sum(dim=1)  # (B, H)
        return self.fc(self.dropout(context))

# ============================================================
# Architecture 2: GRU-2
# ============================================================
class GRUBaseline(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :]))

# ============================================================
# Architecture 3: Transformer-2 Encoder
# ============================================================
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=21, d_model=128, num_layers=2, n_heads=4, dropout=0.3, max_len=60):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]
        encoded = self.encoder(x)
        return self.fc(self.dropout(encoded.mean(dim=1)))  # mean pooling


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResidualLSTM(nn.Module):
    """Configurable depth LSTM with residual + layernorm (for depth >= 4)"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, use_attn=True):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.use_residual = num_layers >= 4
        self.layers = nn.ModuleList()
        self.lns = nn.ModuleList() if self.use_residual else None
        for _ in range(num_layers):
            self.layers.append(nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True))
            if self.use_residual:
                self.lns.append(nn.LayerNorm(hidden_dim))
        self.attn = nn.Linear(hidden_dim, 1) if use_attn else None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x):
        h = self.proj(x)
        for i, lstm in enumerate(self.layers):
            o, _ = lstm(h)
            if self.use_residual:
                h = self.lns[i](o + h)
            else:
                h = o
        if self.attn:
            w = torch.softmax(self.attn(h), dim=1)
            h = (w * h).sum(dim=1)
        else:
            h = h[:, -1, :]
        return self.fc(self.dropout(h))

def create_model(name, input_dim=21):
    if name.startswith('LSTM-'):
        n = int(name.split('-')[1])
        h = {1:128, 2:128, 3:128, 4:96, 5:64, 6:64, 7:128, 8:128, 10:96, 12:64}.get(n, 64)
        d = min(0.2 + n * 0.04, 0.5)
        return ResidualLSTM(input_dim, h, n, d)
    elif name.startswith('GRU-'):
        n = int(name.split('-')[1])
        h = {1:128, 2:128, 3:96, 4:96}.get(n, 96)
        return GRUBaseline(input_dim, h, n, 0.3)
    elif name.startswith('Transformer-'):
        n = int(name.split('-')[1])
        d = min(0.2 + n * 0.03, 0.4)
        return TransformerEncoder(input_dim, 128, n, 4, d)
    else:
        raise ValueError(f"Unknown: {name}")
