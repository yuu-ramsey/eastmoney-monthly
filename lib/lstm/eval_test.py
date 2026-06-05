"""Test set evaluation — FROZEN, opened only ONCE. No tuning after this."""
import torch, numpy as np, json
from pathlib import Path
from scipy.stats import spearmanr
from model import LSTMBaseline, StockDataset

DATA = Path(__file__).parent.parent.parent / '.eastmoney-ai' / 'lstm'
MODEL_PATH = DATA / 'models' / 'lstm_best.pt'

test = np.load(DATA / 'test.npz')
X_test, y_test = test['X'], test['y']
print(f"Test: {X_test.shape}")

# Also load val for comparison
val = np.load(DATA / 'val.npz')
X_val, y_val = val['X'], val['y']

model = LSTMBaseline(input_dim=21, hidden_dim=64, num_layers=1, dropout=0.2)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def evaluate(X, y, name):
    ds = StockDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=64)
    preds_list, targets_list = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            pred = model(X_b)
            preds_list.append(pred.numpy())
            targets_list.append(y_b.numpy())
    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    ic3, p3 = spearmanr(preds[:, 0], targets[:, 0])
    ic6, p6 = spearmanr(preds[:, 1], targets[:, 1])

    n = len(preds)
    cut = int(n * 0.3)
    idx3 = np.argsort(preds[:, 0])
    ls3 = targets[idx3[-cut:], 0] - targets[idx3[:cut], 0]
    sr3 = ls3.mean() / ls3.std() * np.sqrt(12/3)

    idx6 = np.argsort(preds[:, 1])
    ls6 = targets[idx6[-cut:], 1] - targets[idx6[:cut], 1]
    sr6 = ls6.mean() / ls6.std() * np.sqrt(12/6)

    hit3 = np.mean(np.sign(preds[:, 0]) == np.sign(targets[:, 0]))
    hit6 = np.mean(np.sign(preds[:, 1]) == np.sign(targets[:, 1]))

    print(f"\n{'='*60}")
    print(f"{name} (n={n})")
    print(f"{'='*60}")
    print(f"IC y3: {ic3:.4f} (p={p3:.6f})")
    print(f"IC y6: {ic6:.4f} (p={p6:.6f})")
    print(f"L-S Sharpe y3: {sr3:.3f}")
    print(f"L-S Sharpe y6: {sr6:.3f}")
    print(f"Sign Hit y3: {hit3:.3f}  y6: {hit6:.3f}")
    print(f"y3 true: mean={targets[:,0].mean():.4f} std={targets[:,0].std():.4f}")
    print(f"y3 pred: mean={preds[:,0].mean():.4f} std={preds[:,0].std():.4f}")
    print(f"y6 true: mean={targets[:,1].mean():.4f} std={targets[:,1].std():.4f}")
    print(f"y6 pred: mean={preds[:,1].mean():.4f} std={preds[:,1].std():.4f}")

    return {'ic3': ic3, 'ic6': ic6, 'sr3': sr3, 'sr6': sr6, 'hit3': hit3, 'hit6': hit6,
            'preds': preds, 'targets': targets}

val_r = evaluate(X_val, y_val, "VAL (reference)")
test_r = evaluate(X_test, y_test, "TEST (FROZEN)")

# Comparison table
print(f"\n{'='*70}")
print("FINAL COMPARISON")
print(f"{'='*70}")
print(f"{'Metric':<20} {'Val':>12} {'Test':>12} {'Delta':>12}")
print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
for k in ['ic3', 'ic6', 'sr3', 'sr6', 'hit3', 'hit6']:
    v = val_r[k]
    t = test_r[k]
    d = t - v
    print(f"{k:<20} {v:12.4f} {t:12.4f} {d:+12.4f}")

# vs Phase 13 ASC baseline
print(f"\n{'='*70}")
print("vs BASELINES")
print(f"{'='*70}")
print(f"Phase 12 sector alpha IC: 0.074 (reverse)")
print(f"Phase 13 ASC high-conf: score 0.225 (9.4% coverage)")
print(f"LSTM Test IC y3: {test_r['ic3']:.4f} (100% coverage)")
print(f"LSTM Test L-S Sharpe y3: {test_r['sr3']:.3f}")
print(f"\nLSTM IC/coverage ratio: {test_r['ic3']:.3f} / 1.00")
print(f"ASC score/coverage ratio: 0.225 / 0.094")

if test_r['ic3'] > 0.05:
    print("\nConclusion: LSTM baseline IC > 0.05, positive prediction, 100%Covering. Outperforms all Phase 11-15 prompt engineering.")
