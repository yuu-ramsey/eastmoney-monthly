"""Smoke test: small-scale validation of all modules"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json, tempfile; sys.path.insert(0, 'lib')
import torch, numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
from overnight_core import *

PROJECT = Path('.').resolve()
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
errors = []

def check(desc, ok):
    status = 'OK' if ok else 'FAIL'
    print(f'  {status}: {desc}')
    if not ok: errors.append(desc)

print(f'Device: {DEV}')

print('1. SQLite...'); t0 = time.time()
conn = sqlite3.connect(str(DB))
dc = [r[0] for r in conn.execute('SELECT code FROM daily_klines GROUP BY code HAVING COUNT(*)>=500')][:20]
dd = ','.join('?'*len(dc))
df_d = pd.read_sql_query(f"SELECT code,date,open,close,high,low,volume,turnover_rate FROM daily_klines WHERE code IN ({dd}) AND date>='2020-01-01' ORDER BY code,date", conn, params=dc)
mc = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84')][:20]
mm = ','.join('?'*len(mc))
df_m = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({mm}) AND date>='2015-01' ORDER BY code,date", conn, params=mc)
conn.close()
check('daily data', len(df_d) > 1000)
check('monthly data', len(df_m) > 200)
print(f'    {len(df_d)} daily, {len(df_m)} monthly ({time.time()-t0:.0f}s)')

print('2. Daily LSTM...'); t0 = time.time()
X_d, y_d, dates_d, codes_d = build_daily_seqs(df_d, dc)
pre = dates_d < '2018-01-01'
fm = X_d[pre].reshape(-1, X_d.shape[-1]).mean(0) if pre.sum() > 100 else X_d.reshape(-1, X_d.shape[-1]).mean(0)
fs = X_d[pre].reshape(-1, X_d.shape[-1]).std(0) + 1e-8 if pre.sum() > 100 else X_d.reshape(-1, X_d.shape[-1]).std(0) + 1e-8
X_d = np.clip((X_d - fm)/fs, -5, 5)
mid = len(dates_d) // 2
X_tr = torch.from_numpy(X_d[:mid]).float().to(DEV); y_tr = torch.from_numpy(y_d[:mid]).float().to(DEV)
X_v = torch.from_numpy(X_d[mid:]).float().to(DEV)
model = DailyLSTM(d=8, h=16, n_layers=1, dropout=0.3).to(DEV)
val_ic = train_lstm(model, X_tr, y_tr, X_v, y_d[mid:], DEV, epochs=5, patience=3)
check('daily LSTM trains', not np.isnan(val_ic))
print(f'    {len(X_d)} seqs, val_ic={val_ic:.4f} ({time.time()-t0:.0f}s)')
del model; torch.cuda.empty_cache()

print('3. Monthly features...'); t0 = time.time()
X_m, y_m, dates_m, codes_m = build_monthly_feats(df_m, mc)
mid_m = len(dates_m) // 2
p = train_monthly_ensemble(X_m[:mid_m], y_m[:mid_m], X_m[mid_m:], y_m[mid_m:])
ic = spearmanr(p, y_m[mid_m:])[0]; hit = np.mean((p > 0) == (y_m[mid_m:] > 0))
check('monthly ensemble trains', not np.isnan(ic))
print(f'    {len(X_m)} seqs, IC={ic:.4f} Hit={hit:.3f} ({time.time()-t0:.0f}s)')

print('4. JSON...')
tf = Path(tempfile.gettempdir()) / 'ov_test.jsonl'
save_result(tf, 'test', {'exp_id': 'test', 'ic': np.float32(0.1234),
                          'overfit': np.bool_(False), 'gap': np.float64(0.01)})
saved = json.loads(open(tf).readlines()[-1])
check('numpy float', abs(saved['ic'] - 0.1234) < 0.001)
check('numpy bool', saved['overfit'] == False)
os.remove(tf)

print('5. Overfit...')
is_of, gap = overfit_flag(0.12, 0.05)
check('detects overfit', is_of)
is_of2, _ = overfit_flag(0.06, 0.05)
check('no false positive', not is_of2)

if errors:
    print(f'\nFAILED: {len(errors)}')
    for e in errors: print(f'  - {e}')
    sys.exit(1)
else:
    print('\n=== ALL SMOKE TESTS PASSED ===')
