"""Phase 17 LSTM: Training script — 10-epoch dry-run or full training"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time
import sys
import json

from model import LSTMBaseline, StockDataset, count_params

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = Path(__file__).parent.parent.parent / '.eastmoney-ai' / 'lstm'
MODEL_DIR = DATA_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 10


def load_data():
    train = np.load(DATA_DIR / 'train.npz')
    val = np.load(DATA_DIR / 'val.npz')
    return train['X'], train['y'], val['X'], val['y']


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return total_loss / len(loader.dataset), preds, targets


def two_head_loss(pred, target):
    """0.7 * MSE(y3) + 0.3 * MSE(y6) — priority on y3 (stronger IC signal)"""
    mse = nn.MSELoss()
    return 0.7 * mse(pred[:, 0], target[:, 0]) + 0.3 * mse(pred[:, 1], target[:, 1])


def spearman_corr(preds, targets):
    """Compute Spearman correlation for each head"""
    from scipy.stats import spearmanr
    ic3 = spearmanr(preds[:, 0], targets[:, 0])[0] if len(preds) > 10 else 0.0
    ic6 = spearmanr(preds[:, 1], targets[:, 1])[0] if len(preds) > 10 else 0.0
    return ic3, ic6


def main():
    dry_run = '--dry-run' in sys.argv
    max_epochs = 10 if dry_run else 100

    device = torch.device('cpu')
    print(f"Device: {device}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'FULL'} ({max_epochs} epochs)")

    # Load data
    X_train, y_train, X_val, y_val = load_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"y_train mean: {y_train.mean(axis=0)}, y_val mean: {y_val.mean(axis=0)}")

    train_ds = StockDataset(X_train, y_train)
    val_ds = StockDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = LSTMBaseline(input_dim=21, hidden_dim=64, num_layers=1, dropout=0.2).to(device)
    n_params = count_params(model)
    print(f"\nModel params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'ic_y3': [], 'ic_y6': [], 'epoch_time': []}
    best_ic_y3 = -float('inf')
    best_epoch = 0
    no_improve = 0

    print(f"\n{'='*72}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'IC y3':>8} {'IC y6':>8} {'Time':>8} {'Best IC':>8}")
    print(f"{'='*72}")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, two_head_loss, device)
        val_loss, val_preds, val_targets = evaluate(model, val_loader, two_head_loss, device)

        epoch_time = time.time() - t0

        # IC
        ic_y3, ic_y6 = spearman_corr(val_preds, val_targets)

        # Scheduler step (monitor IC y3, higher is better)
        scheduler.step(ic_y3)

        # Log
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['ic_y3'].append(float(ic_y3))
        history['ic_y6'].append(float(ic_y6))
        history['epoch_time'].append(epoch_time)

        marker = ''
        if ic_y3 > best_ic_y3:
            best_ic_y3 = ic_y3
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), MODEL_DIR / 'lstm_best.pt')
            marker = ' *'
        else:
            no_improve += 1

        print(f"{epoch:6d} {train_loss:12.6f} {val_loss:12.6f} {ic_y3:8.4f} {ic_y6:8.4f} {epoch_time:7.1f}s {best_ic_y3:7.4f}{marker}")

        if no_improve >= PATIENCE and epoch > 10:
            print(f"\nEarly stopping at epoch {epoch} (no IC y3 improvement for {PATIENCE} epochs)")
            break

    print(f"\nBest val IC y3: {best_ic_y3:.4f} at epoch {best_epoch} (IC y6: {history['ic_y6'][best_epoch-1]:.4f})")

    # Save history
    with open(DATA_DIR / 'train_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # ---- Dry-run report ----
    if dry_run:
        # Load best model for prediction samples
        model.load_state_dict(torch.load(MODEL_DIR / 'lstm_best.pt'))
        model.eval()
        with torch.no_grad():
            sample_preds, sample_targets = evaluate(model, val_loader, two_head_loss, device)[1:]

        print(f"\n{'='*60}")
        print("DRY-RUN REPORT")
        print(f"{'='*60}")

        # Loss curve summary
        print(f"\nLoss curve (first 3 vs last 3 epochs):")
        for e in [0, 1, 2, -3, -2, -1]:
            ep = history['epoch'][e]
            print(f"  Epoch {ep}: train={history['train_loss'][e]:.6f} val={history['val_loss'][e]:.6f}")

        # Divergence check
        train_last3 = np.mean(history['train_loss'][-3:])
        val_last3 = np.mean(history['val_loss'][-3:])
        gap = val_last3 / max(train_last3, 1e-8) - 1.0
        print(f"\nTrain/Val gap: train_mean(last3)={train_last3:.6f} val_mean(last3)={val_last3:.6f}")
        print(f"Val/Train ratio: {1.0 + gap:.3f}")
        if gap > 0.2:
            print("⚠  Val significantly worse than train — possible overfitting")
        else:
            print("✓  Train/Val gap within normal range")

        # Epoch time
        avg_epoch_time = np.mean(history['epoch_time'])
        print(f"\nAvg epoch time: {avg_epoch_time:.1f}s")
        print(f"Total params: {n_params:,}")

        # Memory
        import os
        try:
            import psutil
            mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            print(f"Peak memory: {mem:.0f} MB")
        except ImportError:
            print("Peak memory: N/A (psutil not installed)")

        # Prediction distribution vs ground truth
        print(f"\n--- Prediction Distribution vs Ground Truth (y3) ---")
        print(f"  y3_true: mean={sample_targets[:, 0].mean():.4f} std={sample_targets[:, 0].std():.4f} "
              f"p5={np.percentile(sample_targets[:, 0], 5):.4f} p50={np.percentile(sample_targets[:, 0], 50):.4f} p95={np.percentile(sample_targets[:, 0], 95):.4f}")
        print(f"  y3_pred: mean={sample_preds[:, 0].mean():.4f} std={sample_preds[:, 0].std():.4f} "
              f"p5={np.percentile(sample_preds[:, 0], 5):.4f} p50={np.percentile(sample_preds[:, 0], 50):.4f} p95={np.percentile(sample_preds[:, 0], 95):.4f}")

        # Prediction samples (first 5 val samples)
        print(f"\n--- Prediction Samples (first 5 val) ---")
        print(f"{'#':>3} {'y3_true':>8} {'y3_pred':>8} {'y6_true':>8} {'y6_pred':>8}")
        n_samples = min(5, len(sample_preds))
        for i in range(n_samples):
            print(f"{i+1:3d} {sample_targets[i, 0]:8.4f} {sample_preds[i, 0]:8.4f} "
                  f"{sample_targets[i, 1]:8.4f} {sample_preds[i, 1]:8.4f}")

        # IC summary
        print(f"\nVal IC (best epoch={best_epoch}): y3={history['ic_y3'][best_epoch-1]:.4f} y6={history['ic_y6'][best_epoch-1]:.4f}")

        print(f"\nDry-run complete. Model saved to {MODEL_DIR / 'lstm_best.pt'}")
    else:
        print(f"\nTraining complete. Model saved to {MODEL_DIR / 'lstm_best.pt'}")


if __name__ == '__main__':
    main()
