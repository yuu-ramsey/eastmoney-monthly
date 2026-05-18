"""Val set evaluation with best checkpoint — StricTest NOT touched"""
import torch, numpy as np, json
from pathlib import Path
from scipy.stats import spearmanr
from model import LSTMBaseline, StockDataset

DATA = Path(__file__).parent.parent.parent / '.eastmoney-ai' / 'lstm'
MODEL_PATH = DATA / 'models' / 'lstm_best.pt'

val = np.load(DATA / 'val.npz')
X_val, y_val = val['X'], val['y']

model = LSTMBaseline(input_dim=21, hidden_dim=64, num_layers=1, dropout=0.2)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

ds = StockDataset(X_val, y_val)
loader = torch.utils.data.DataLoader(ds, batch_size=64)
preds_list, targets_list = [], []
with torch.no_grad():
    for X_b, y_b in loader:
        pred = model(X_b)
        preds_list.append(pred.numpy())
        targets_list.append(y_b.numpy())
preds = np.concatenate(preds_list)
targets = np.concatenate(targets_list)

# IC
ic3, p3 = spearmanr(preds[:, 0], targets[:, 0])
ic6, p6 = spearmanr(preds[:, 1], targets[:, 1])
print(f"IC y3: {ic3:.4f} (p={p3:.6f})")
print(f"IC y6: {ic6:.4f} (p={p6:.6f})")

# Distribution
d3t = targets[:, 0]
d3p = preds[:, 0]
d6t = targets[:, 1]
d6p = preds[:, 1]

print(f"\ny3 true: mean={d3t.mean():.4f} std={d3t.std():.4f} p5={np.percentile(d3t,5):.4f} p50={np.percentile(d3t,50):.4f} p95={np.percentile(d3t,95):.4f}")
print(f"y3 pred: mean={d3p.mean():.4f} std={d3p.std():.4f} p5={np.percentile(d3p,5):.4f} p50={np.percentile(d3p,50):.4f} p95={np.percentile(d3p,95):.4f}")
print(f"y6 true: mean={d6t.mean():.4f} std={d6t.std():.4f} p5={np.percentile(d6t,5):.4f} p50={np.percentile(d6t,50):.4f} p95={np.percentile(d6t,95):.4f}")
print(f"y6 pred: mean={d6p.mean():.4f} std={d6p.std():.4f} p5={np.percentile(d6p,5):.4f} p50={np.percentile(d6p,50):.4f} p95={np.percentile(d6p,95):.4f}")

# Hit rate
hit3 = np.mean(np.sign(preds[:, 0]) == np.sign(targets[:, 0]))
hit6 = np.mean(np.sign(preds[:, 1]) == np.sign(targets[:, 1]))
print(f"\nSign hit rate: y3={hit3:.3f} y6={hit6:.3f}")

# Long-Short
n = len(preds)
cut = int(n * 0.3)

idx3 = np.argsort(preds[:, 0])
long3 = targets[idx3[-cut:], 0].mean()
short3 = targets[idx3[:cut], 0].mean()
spread3 = long3 - short3
ls3 = targets[idx3[-cut:], 0] - targets[idx3[:cut], 0]
sr3 = spread3 / ls3.std() * np.sqrt(12/3)
print(f"\nL-S y3: spread={spread3:.4f} Sharpe(ann)={sr3:.3f}")

idx6 = np.argsort(preds[:, 1])
long6 = targets[idx6[-cut:], 1].mean()
short6 = targets[idx6[:cut], 1].mean()
spread6 = long6 - short6
ls6 = targets[idx6[-cut:], 1] - targets[idx6[:cut], 1]
sr6 = spread6 / ls6.std() * np.sqrt(12/6)
print(f"L-S y6: spread={spread6:.4f} Sharpe(ann)={sr6:.3f}")

# History
with open(DATA / 'train_history.json') as f:
    h = json.load(f)
best_ep = int(np.argmax(h['ic_y3'])) + 1
print(f"\nBest epoch: {best_ep}")
print(f"Train loss: {h['train_loss'][0]:.4f} -> {h['train_loss'][-1]:.4f}")
print(f"Val loss: {h['val_loss'][0]:.4f} -> {h['val_loss'][-1]:.4f}")
print(f"IC y3 peak: {h['ic_y3'][best_ep-1]:.4f}")
print(f"IC y6 peak: {h['ic_y6'][best_ep-1]:.4f}")
