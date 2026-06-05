"""通宵实验核心模块 — import 不触发执行。"""
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn
import numpy as np, pandas as pd, json
from scipy.stats import spearmanr
import lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import Ridge
import pywt

LOOKBACK, BATCH_SIZE = 60, 1024

# ═══ 日线 LSTM ═══
class DailyLSTM(nn.Module):
    def __init__(self, d=8, h=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(d, h, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(self.dropout(o[:, -1, :])).squeeze(-1)

def build_daily_seqs(df, codes, lookback=LOOKBACK):
    Xl, yl, dl, cl = [], [], [], []
    for code in codes:
        g = df[df['code']==code].sort_values('date').reset_index(drop=True)
        if len(g) < lookback + 66: continue
        c = g['close'].values.astype(float); o = g['open'].values.astype(float)
        h = g['high'].values.astype(float); l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float)
        tr = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
        n = len(c); vol_ma20 = pd.Series(v).rolling(20).mean().fillna(1).values
        F = np.zeros((n, 8), dtype=np.float32)
        for i in range(n):
            pc = max(abs(c[i-1]), 0.01) if i >= 1 else c[i]
            F[i,0]=o[i]/max(pc,0.01)-1; F[i,1]=h[i]/max(c[i],0.01)-1
            F[i,2]=l[i]/max(c[i],0.01)-1; F[i,3]=c[i]/max(pc,0.01)-1
            F[i,4]=v[i]/max(vol_ma20[i],1)-1; F[i,5]=tr[i] if not np.isnan(tr[i]) and tr[i]<100 else 0
            F[i,6]=(h[i]-l[i])/max(c[i],0.01); F[i,7]=(c[i]-o[i])/max(o[i],0.01) if o[i]>0 else 0
        F = np.nan_to_num(F, 0.0).astype(np.float32)
        dg = g['date'].values
        for i in range(lookback-1, n-63):
            fwd = (c[i+63]-c[i])/max(c[i],0.01)
            if abs(fwd) > 3: continue
            Xl.append(F[i-lookback+1:i+1]); yl.append(np.clip(fwd, -3, 3))
            dl.append(str(dg[i])[:10]); cl.append(code)
    X = np.array(Xl, dtype=np.float32); y = np.array(yl, dtype=np.float32)
    da = np.array(dl); ca = np.array(cl)
    v = ~np.isnan(X).any(axis=(1,2)) & ~np.isnan(y)
    return X[v], y[v], da[v], ca[v]

def train_lstm(model, X_tr_t, y_tr_t, X_v_t, y_v, device,
               epochs=120, patience=10, lr=0.001, wd=1e-4, batch=BATCH_SIZE):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss(); best_ic, best_st, no_imp = -99, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(X_tr_t), device=device)
        for i in range(0, len(X_tr_t), batch):
            idx = perm[i:i+batch]; loss = loss_fn(model(X_tr_t[idx]), y_tr_t[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            pv = np.concatenate([model(X_v_t[i:i+batch]).cpu().numpy()
                                 for i in range(0, len(X_v_t), batch)])
            ic = spearmanr(pv, y_v)[0]
        if ic > best_ic: best_ic=ic; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
        else: no_imp+=1
        if no_imp >= patience: break
    model.load_state_dict(best_st); return best_ic

def lstm_predict(model, X_te_t, device, batch=BATCH_SIZE):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(X_te_t[i:i+batch]).cpu().numpy()
                               for i in range(0, len(X_te_t), batch)])

# ═══ 月线特征 + 集成 ═══
def wdenoise(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=2)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    return pywt.waverec([coeffs[0]] + [pywt.threshold(c, threshold, mode='soft')
                        for c in coeffs[1:]], 'db4')[:len(signal)]

def build_monthly_feats(df, codes):
    fl, yl, dl, cl = [], [], [], []
    for code in codes:
        g = df[df['code']==code].sort_values('date').reset_index(drop=True)
        if len(g) < 72: continue
        c = g['close'].values.astype(float); o = g['open'].values.astype(float)
        h = g['high'].values.astype(float); l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float)
        tr = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
        n = len(c)
        ma5 = pd.Series(c).rolling(5).mean().values; ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
        dif = e12-e26; dea = pd.Series(dif).ewm(span=9).mean().values; macd_hist = (dif-dea)*2
        delta = np.diff(c, prepend=c[0]); gain=np.where(delta>0,delta,0); loss=np.where(delta<0,-delta,0)
        avg_gain = pd.Series(gain).ewm(alpha=1/14).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/14).mean().values
        rsi14 = np.nan_to_num(100-100/(1+avg_gain/np.maximum(avg_loss,1e-8)), 50)
        bb_std = pd.Series(c).rolling(20).std().values
        bb_pos = np.nan_to_num((c-(ma20-2*bb_std))/np.maximum(4*bb_std,0.01), 0.5)
        trange = np.maximum(h-l, np.abs(h-np.roll(c,1))); atr14 = pd.Series(trange).rolling(14).mean().values
        vol_ma20 = pd.Series(v).rolling(20).mean().fillna(1).values
        tr_ma20 = pd.Series(tr).rolling(20).mean().fillna(0).values
        for i in range(60, n-6):
            fwd = (c[i+6]-c[i])/max(c[i],0.01)
            if abs(fwd) > 5: continue
            f = [c[i]/max(ma5[i],0.01)-1, c[i]/max(ma20[i],0.01)-1, c[i]/max(ma60[i],0.01)-1,
                 dif[i], dea[i], macd_hist[i],
                 np.std(c[max(0,i-5):i+1])/max(np.mean(np.abs(c[max(0,i-5):i+1])),0.01) if i>=1 else 0,
                 atr14[i]/max(c[i],0.01) if c[i]>0 else 0]
            seg = c[max(0,i-60):i+1]; x_r = np.arange(len(seg))
            trend = np.polyfit(x_r,seg,1); detrended = seg-np.polyval(trend,x_r)
            fft_p = np.fft.rfft(detrended); amps = np.abs(fft_p)
            pk = np.argsort(amps[1:])[::-1][:10]+1 if len(amps)>1 else np.array([],dtype=int)
            ff = amps[pk].astype(np.float32) if len(pk)>0 else np.zeros(10,dtype=np.float32)
            f.extend(ff[:10].tolist() if len(ff)>=10 else ff.tolist()+[0]*(10-len(ff)))
            while len(f) < 18: f.append(0)
            f.extend([v[i]/max(vol_ma20[i],1)-1, tr[i]/max(tr_ma20[i],0.01)-1 if tr_ma20[i]>0 else 0,
                      rsi14[i]/100, (c[i]-o[i])/max(o[i],0.01) if o[i]>0 else 0,
                      (h[i]-c[i])/max(c[i],0.01) if c[i]>0 else 0, (l[i]-c[i])/max(c[i],0.01) if c[i]>0 else 0,
                      bb_pos[i],
                      c[i]/max(c[max(0,i-1)],0.01)-1 if i>0 else 0,
                      c[i]/max(c[max(0,i-3)],0.01)-1 if i>=3 else 0,
                      c[i]/max(c[max(0,i-6)],0.01)-1 if i>=6 else 0,
                      c[i]/max(c[max(0,i-12)],0.01)-1 if i>=12 else 0,
                      c[i]/max(c[max(0,i-24)],0.01)-1 if i>=24 else 0,
                      1 if c[i]>ma5[i] else 0, 1 if c[i]>ma20[i] else 0])
            f = np.nan_to_num(np.array(f, dtype=np.float32), 0.0)
            fl.append(f); yl.append(fwd)
            dl.append(str(g['date'].values[i])[:7]); cl.append(code)
    X = np.array(fl, dtype=np.float32); y = np.array(yl, dtype=np.float32)
    da = np.array(dl); ca = np.array(cl)
    v = ~np.isnan(y); X=X[v]; y=y[v]; da=da[v]; ca=ca[v]
    return X, y, da, ca

def train_monthly_ensemble(X_tr, y_tr, X_te, y_te):
    lgb_m = lgb.LGBMRegressor(n_estimators=200, max_depth=6, num_leaves=31,
                               learning_rate=0.05, verbose=-1, random_state=42)
    lgb_m.fit(X_tr, y_tr); p_lgb = lgb_m.predict(X_te)
    xgb_m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                              verbosity=0, random_state=42)
    xgb_m.fit(X_tr, y_tr); p_xgb = xgb_m.predict(X_te)
    ridge_m = Ridge(alpha=1.0); ridge_m.fit(X_tr, y_tr); p_ridge = ridge_m.predict(X_te)
    return (p_lgb + p_xgb + p_ridge) / 3

# ═══ 工具 ═══
def cs_ic_helper(pred, true, dates):
    months = np.unique(dates); ics = []
    for m in months:
        mask = dates == m
        if mask.sum() >= 20: ics.append(spearmanr(pred[mask], true[mask])[0])
    return np.mean(ics) if ics else np.nan, ics

def overfit_flag(val_ic, test_ic, threshold=0.05):
    gap = val_ic - test_ic
    return gap > threshold, round(float(gap), 4)

def save_result(results_path, run_date, r):
    r['run_date'] = run_date
    clean = {}
    for k, v in r.items():
        if isinstance(v, (np.integer,)): clean[k] = int(v)
        elif isinstance(v, (np.floating,)): clean[k] = float(v)
        elif isinstance(v, (np.bool_,)): clean[k] = bool(v)
        elif isinstance(v, np.ndarray): clean[k] = v.tolist()
        else: clean[k] = v
    with open(results_path, 'a') as f:
        f.write(json.dumps(clean, ensure_ascii=False) + '\n')
