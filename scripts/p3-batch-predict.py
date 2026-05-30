"""Batch kronos + LSTM on v2 pool. Reuses existing modules."""
import json, sys, os, time, torch, numpy as np, pandas as pd, sqlite3
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

with open(PROJECT/'data'/'frozen-eval-lowpos-v2-baostock.json') as f:
    pool = json.load(f)
pairs = []
seen = set()
for tp in pool['testPoints']:
    if tp['alpha'] is None: continue
    k = f"{tp['stockCode']}|{tp['cutoffDate']}"
    if k not in seen: seen.add(k); pairs.append(tp)
print(f"Pairs: {len(pairs)}")
codes = sorted(set(tp['stockCode'] for tp in pairs))

# ====== Kronos ======
print("\n=== Kronos ===")
k_signals = {}
try:
    from kronos.predictor import KronosPredictor
    from kronos.tokenizer import KronosTokenizer
    from kronos.transformer import Kronos
    from kronos.data_adapter import load_monthly_klines

    tok = KronosTokenizer.from_pretrained(str(PROJECT/'kronos'/'weights'/'tokenizer'))
    model = Kronos.from_pretrained(str(PROJECT/'kronos'/'weights'/'model')).to(DEVICE).eval()
    pred = KronosPredictor(model, tok, device=DEVICE)
    print(f"Kronos loaded: {sum(p.numel() for p in model.parameters()):,} params")

    t0 = time.time()
    for i, tp in enumerate(pairs):
        try:
            df = load_monthly_klines(tp['stockCode'], min_records=12)
            if df is None or len(df) < 12: continue
            df = df[df.index <= tp['cutoffDate']]
            if len(df) < 12: continue
            r = pred.predict(df, pred_len=6, n_samples=10)
            if r and hasattr(r, 'pred_close') and r.pred_close and len(r.pred_close) >= 6:
                close0 = df['close'].iloc[-1]
                if close0 > 0.01:
                    k_signals[f"{tp['stockCode']}|{tp['cutoffDate']}"] = float((r.pred_close[5]-close0)/close0*100)
        except: pass
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(pairs)} ({len(k_signals)} ok, {(time.time()-t0):.0f}s)")
    print(f"  Done: {len(k_signals)} in {(time.time()-t0):.0f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()

# ====== LSTM ======
print("\n=== LSTM ===")
l_signals = {}
try:
    sys.path.insert(0, str(PROJECT/'lib'/'lstm'))
    from model_v2 import create_model

    DB = PROJECT/'.eastmoney-ai'/'db'/'klines-v2.sqlite'
    conn = sqlite3.connect(str(DB))
    d_df = pd.read_sql_query(
        f"SELECT code,date,open,high,low,close,volume FROM daily_klines WHERE code IN ({','.join('?'*len(codes))}) AND date>='2010-01-01' ORDER BY code,date",
        conn, params=codes)
    conn.close()
    print(f"Daily rows: {len(d_df)}")

    LOOKBACK = 252
    mp = PROJECT/'.eastmoney-ai'/'lstm'/'models_v2'/'daily_lstm7.pt'
    lstm = create_model('LSTM-2', 10).to(DEVICE)
    lstm.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=True))
    lstm.eval()
    print("LSTM loaded")

    def rz(s, w=60):
        m = s.rolling(w, min_periods=w).mean()
        std = s.rolling(w, min_periods=w).std()
        return ((s - m) / std).fillna(0).values

    t0 = time.time(); si = 0
    for code, grp in d_df.groupby('code'):
        if code not in set(codes): continue
        grp = grp.sort_values('date').reset_index(drop=True)
        if len(grp) < LOOKBACK + 63: continue
        n = len(grp); dates = grp['date'].tolist()
        c = grp['close'].values.astype(float)
        feats = np.zeros((n, 10), dtype=np.float32)
        s = pd.Series(c)
        feats[:, 0] = rz(s)
        e12 = s.ewm(span=12).mean().values; e26 = s.ewm(span=26).mean().values
        dif = np.nan_to_num(e12 - e26, 0)
        dea = pd.Series(dif).ewm(span=9).mean().values
        feats[:, 5] = dif; feats[:, 6] = dea; feats[:, 7] = (dif - dea) * 2
        feats[:, 9] = np.nan_to_num((c - s.rolling(60).mean().values) / np.maximum(c, 0.01), 0)
        feats = np.nan_to_num(feats, 0.0)

        code_tps = [t for t in pairs if t['stockCode'] == code]
        if not code_tps: continue
        seqs, keys = [], []
        for tp in code_tps:
            ci = -1
            for j, d in enumerate(dates):
                if str(d).startswith(tp['cutoffDate']): ci = j; break
            if ci >= LOOKBACK - 1:
                seqs.append(feats[ci - LOOKBACK + 1:ci + 1])
                keys.append(f"{code}|{tp['cutoffDate']}")
        if seqs:
            with torch.no_grad():
                X = torch.from_numpy(np.array(seqs)).float().to(DEVICE)
                preds = lstm(X).cpu().numpy()[:, 0]
                for k, v in zip(keys, preds): l_signals[k] = float(v)
        si += 1
        if si % 50 == 0: print(f"  {si}/{len(codes)} ({len(l_signals)} preds, {(time.time()-t0):.0f}s)")
    print(f"  Done: {len(l_signals)} in {(time.time()-t0):.0f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()

out = {'kronos': k_signals, 'lstm': l_signals}
with open(PROJECT/'data'/'p3-kronos-lstm-signals.json', 'w') as f:
    json.dump(out, f)
print(f"\nSaved: {len(k_signals)} kronos + {len(l_signals)} lstm signals")
