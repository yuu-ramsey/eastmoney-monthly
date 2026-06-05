"""Pre-fill experiments table with all overnight tasks (~186 experiments)"""
import sqlite3, json
from pathlib import Path

DB = Path(__file__).parent.parent.parent / '.eastmoney-ai' / 'lstm' / 'overnight.db'
DB.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(str(DB))
conn.execute('''CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY, phase TEXT, priority INTEGER, config_json TEXT,
    status TEXT DEFAULT 'pending', start_time TEXT, end_time TEXT,
    ic_y3 REAL, ic_y6 REAL, val_loss REAL, error TEXT)''')
conn.execute('''CREATE INDEX IF NOT EXISTS idx_status_priority ON experiments(status, priority)''')

experiments = []

def add(phase, priority, config):
    experiments.append((phase, priority, json.dumps(config)))

# ---- P1 Stage 4: Walk-forward ----
for model in ['LSTM-5', 'LSTM-7', 'LSTM-8']:
    for year in range(2020, 2026):
        add('P1_WF', 1, {'task': 'walk_forward', 'model': model, 'retrain_year': year, 'epochs': 80, 'seed': 42})

# ---- P1 Stage 5: Optuna light (grid search, 9 combos each) ----
for model in ['LSTM-5', 'LSTM-7', 'LSTM-8']:
    for lr in [1e-4, 5e-4, 1e-3]:
        for wd in [1e-5, 1e-4, 1e-3]:
            add('P1_Optuna', 1, {'task': 'hyper', 'model': model, 'lr': lr, 'wd': wd, 'epochs': 50, 'seed': 42})

# ---- P2 Long training ----
for model in ['LSTM-5', 'LSTM-7', 'LSTM-8']:
    add('P2_Long', 2, {'task': 'train', 'model': model, 'epochs': 300, 'patience': 30, 'lr': 5e-4, 'seed': 42})

# ---- P2 Multi-target weights ----
for w3, w6 in [(0.5,0.5), (1.0,0.0), (0.7,0.3), (0.3,0.7), (0.6,0.4), (0.4,0.6), (0.8,0.2), (0.2,0.8)]:
    add('P2_LossW', 2, {'task': 'train', 'model': 'LSTM-7', 'epochs': 50, 'loss_w3': w3, 'loss_w6': w6, 'seed': 42})

# ---- P3: Advanced arch (light) ----
for arch in ['Transformer-4', 'GRU-2']:
    add('P3_Arch', 3, {'task': 'train', 'model': arch, 'epochs': 80, 'seed': 42})

# ---- P4: Multi-seed ----
for model in ['LSTM-5', 'LSTM-7', 'LSTM-8']:
    for s in [123, 456, 789]:
        add('P4_Seed', 4, {'task': 'train', 'model': model, 'epochs': 50, 'seed': s})

conn.executemany("INSERT INTO experiments (phase, priority, config_json) VALUES (?, ?, ?)", experiments)
conn.commit()
print(f"Created {len(experiments)} experiments")
print(f"  P1: {sum(1 for e in experiments if e[0].startswith('P1'))}")
print(f"  P2: {sum(1 for e in experiments if e[0].startswith('P2'))}")
print(f"  P3: {sum(1 for e in experiments if e[0].startswith('P3'))}")
print(f"  P4: {sum(1 for e in experiments if e[0].startswith('P4'))}")
conn.close()
