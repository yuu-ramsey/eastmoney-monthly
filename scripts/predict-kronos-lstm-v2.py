"""Batch kronos + LSTM inference on v2 Baostock pool. Uses GPU."""
import json, sys, os, time, torch, numpy as np, pandas as pd, sqlite3
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

with open(PROJECT / 'data' / 'frozen-eval-lowpos-v2-baostock.json') as f:
    pool = json.load(f)
unique_pairs = {}
for tp in pool['testPoints']:
    if tp['alpha'] is None: continue
    k = f"{tp['stockCode']}|{tp['cutoffDate']}"
    if k not in unique_pairs: unique_pairs[k] = tp
pairs = list(unique_pairs.values())
print(f"Pairs: {len(pairs)}")

# ====== Kronos ======
print("\n=== Kronos ===")
kronos_signals = {}
try:
    from kronos.predictor import KronosPredictor
    from kronos.tokenizer import KronosTokenizer
    from kronos.transformer import Kronos
    from kronos.data_adapter import load_monthly_klines

    tokenizer = KronosTokenizer.from_pretrained(str(PROJECT / 'kronos' / 'weights' / 'tokenizer'))
    model = Kronos.from_pretrained(str(PROJECT / 'kronos' / 'weights' / 'model')).to(DEVICE)
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device=DEVICE)
    print(f"Kronos loaded: {sum(p.numel() for p in model.parameters()):,} params")

    t0 = time.time()
    for i, tp in enumerate(pairs):
        code, cutoff = tp['stockCode'], tp['cutoffDate']
        try:
            df = load_monthly_klines(code, min_records=12)
            if df is None or len(df) < 12: continue
            df = df[df.index <= cutoff]
            if len(df) < 12: continue
            result = predictor.predict(df, pred_len=6, n_samples=10)
            if result and hasattr(result, 'pred_close') and result.pred_close is not None:
                preds = result.pred_close
                if len(preds) >= 6 and df['close'].iloc[-1] > 0.01:
                    pred_ret = (preds[5] - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100
                    kronos_signals[f"{code}|{cutoff}"] = float(pred_ret)
        except: pass
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(pairs)} ({len(kronos_signals)} ok, {(time.time()-t0):.0f}s)")
    print(f"  Done: {len(kronos_signals)} in {(time.time()-t0):.0f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()

# ====== LSTM ======
print("\n=== LSTM ===")
lstm_signals = {}
try:
    sys.path.insert(0, str(PROJECT / 'lib' / 'lstm'))
    from model_v2 import create_model

    codes = list(set(tp['stockCode'] for tp in pairs))
    DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
    conn = sqlite3.connect(str(DB))
    d_df = pd.read_sql_query(
        f"SELECT code, date, open, high, low, close, volume FROM daily_klines WHERE code IN ({','.join('?'*len(codes))}) AND date >= '2010-01-01' ORDER BY code, date",
        conn, params=codes)
    conn.close()

    LOOKBACK = 252
    mp = PROJECT / '.eastmoney-ai' / 'lstm' / 'models_v2' / 'daily_lstm7.pt'
    if not mp.exists(): mp = PROJECT / '.eastmoney-ai' / 'lstm' / 'models_v2' / 'LSTM-2.pt'

    lstm = create_model('LSTM-2', 10).to(DEVICE)
    lstm.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=True))
    lstm.eval()
    print(f"Loaded: {mp.name}")

    def rz(s, w=60):
        m = s.rolling(w, min_periods=w).mean(); std = s.rolling(w, min_periods=w).std()
        return ((s - m) / std).fillna(0).values

    t0 = time.time(); stock_i = 0
    for code, grp in d_df.groupby('code'):
        if code not in set(tp['stockCode'] for tp in pairs): continue
        grp = grp.sort_values('date').reset_index(drop=True)
        if len(grp) < LOOKBACK + 63: continue
        n = len(grp); dates = grp['date'].tolist()
        c = grp['close'].values.astype(float); o = grp['open'].values.astype(float)
        h = grp['high'].values.astype(float); l = grp['low'].values.astype(float)
        v = grp['volume'].values.astype(float)

        feats = np.zeros((n, 21), dtype=np.float32)
        s = pd.Series(c)
        feats[:,0]=rz(s); feats[:,1]=rz(pd.Series(o)); feats[:,2]=rz(pd.Series(h))
        feats[:,3]=rz(pd.Series(l)); feats[:,4]=rz(pd.Series(v))
        e12=s.ewm(span=12).mean().values; e26=s.ewm(span=26).mean().values
        feats[:,5]=np.nan_to_num(e12-e26,0)
        feats[:,6]=np.nan_to_num(pd.Series(feats[:,5]).ewm(span=9).mean().values,0)
        feats[:,7]=(feats[:,5]-feats[:,6])*2; feats[:,8]=np.nan_to_num(50,50)
        feats[:,18]=np.nan_to_num((c-s.rolling(20).mean().values)/np.maximum(c,0.01),0)
        feats[:,19]=np.nan_to_num((c-s.rolling(60).mean().values)/np.maximum(c,0.01),0)
        feats[:,20]=hash(code)%31/31.0
        feats=np.nan_to_num(feats[:,:10].copy(),0.0)

        for tp in [t for t in pairs if t['stockCode']==code]:
            ci=-1
            for j,d in enumerate(dates):
                if str(d).startswith(tp['cutoffDate']): ci=j; break
            if ci<LOOKBACK-1: continue
            seq=feats[ci-LOOKBACK+1:ci+1]
            with torch.no_grad():
                X=torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)
                lstm_signals[f"{code}|{tp['cutoffDate']}"]=float(lstm(X).cpu().numpy()[0][0])
        stock_i+=1
        if stock_i%50==0: print(f"  {stock_i}/{len(codes)} ({len(lstm_signals)} preds, {(time.time()-t0):.0f}s)")
    print(f"  Done: {len(lstm_signals)} in {(time.time()-t0):.0f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()

output = {'kronos': kronos_signals, 'lstm': lstm_signals, 'n_pairs': len(pairs)}
with open(PROJECT / 'data' / 'p3-kronos-lstm-signals.json', 'w') as f:
    json.dump(output, f)
print(f"\nSaved: {len(kronos_signals)} kronos + {len(lstm_signals)} lstm")
