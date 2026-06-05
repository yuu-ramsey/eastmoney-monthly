# Path A: Extended Feature Engineering (SGP-LSTM paper's core insight)
# Target: 66d -> 85d, CS_IC improvement over baseline +0.163
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time, json
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb, xgboost as xgb
import pywt

DB = '.eastmoney-ai/db/klines-v2.sqlite'

# ======== Feature Engineering ========
def build_features(c, o, h, l, v, tr):
    """Extended feature builder: 66d -> 85d (adding 19 new features)"""
    n = len(c)
    # --- Rollings ---
    ma5 = pd.Series(c).rolling(5).mean().values
    ma10 = pd.Series(c).rolling(10).mean().values
    ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    # MACD
    e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
    dif = e12 - e26; dea = pd.Series(dif).ewm(span=9).mean().values; macd_hist = (dif - dea) * 2
    # RSI
    delta = np.diff(c, prepend=c[0]); gain = np.where(delta > 0, delta, 0); loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14).mean().values; avg_loss = pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14 = np.nan_to_num(100 - 100/(1 + avg_gain/np.maximum(avg_loss, 1e-8)), 50)
    # Bollinger
    bb_std = pd.Series(c).rolling(20).std().values
    bb_mid = ma20; bb_upper = bb_mid + 2*bb_std; bb_lower = bb_mid - 2*bb_std
    bb_pos = np.nan_to_num((c - bb_lower) / np.maximum(4*bb_std, 0.01), 0.5)
    bb_width = np.nan_to_num((bb_upper - bb_lower) / np.maximum(bb_mid, 0.01), 0)
    # ATR
    trange = np.maximum(h - l, np.abs(h - np.roll(c, 1)))
    atr14 = pd.Series(trange).rolling(14).mean().values
    # Volume
    vol_ma3 = pd.Series(v).rolling(3).mean().values; vol_ma12 = pd.Series(v).rolling(12).mean().values
    # Price position in 5yr range
    p5h = pd.Series(c).rolling(60).max().values; p5l = pd.Series(c).rolling(60).min().values
    # Candle
    body_pct = np.abs(c - o) / np.maximum(h - l, 0.01)
    upper_shadow = (np.maximum(o, c) - h) / np.maximum(h - l, 0.01)
    lower_shadow = (l - np.minimum(o, c)) / np.maximum(h - l, 0.01)
    # Consecutive streaks
    up_streak = np.zeros(n); dn_streak = np.zeros(n)
    for i in range(1, n):
        up_streak[i] = up_streak[i-1] + 1 if c[i] > c[i-1] else 0
        dn_streak[i] = dn_streak[i-1] + 1 if c[i] < c[i-1] else 0
    # NEW: ADX proxy (trend strength from directional movement)
    dm_plus = np.maximum(h - np.roll(h, 1), 0)
    dm_minus = np.maximum(np.roll(l, 1) - l, 0)
    atr12 = pd.Series(trange).rolling(12).mean().values
    dx = np.nan_to_num(np.abs(dm_plus - dm_minus) / np.maximum(dm_plus + dm_minus, 0.01) * 100, 0)
    adx14 = pd.Series(dx).rolling(14).mean().values
    # NEW: CCI (Commodity Channel Index)
    tp = (h + l + c) / 3
    tp_sma20 = pd.Series(tp).rolling(20).mean().values
    mad_tp = pd.Series(np.abs(tp - tp_sma20)).rolling(20).mean().values
    cci20 = np.nan_to_num((tp - tp_sma20) / np.maximum(0.015 * mad_tp, 0.001), 0)
    # NEW: OBV proxy
    obv = np.zeros(n)
    for i in range(1, n): obv[i] = obv[i-1] + v[i] * (1 if c[i] > c[i-1] else -1 if c[i] < c[i-1] else 0)
    obv_ma12 = pd.Series(obv).rolling(12).mean().values
    obv_ratio = np.nan_to_num(obv / np.maximum(obv_ma12, 1), 1)
    # NEW: Volume-price divergence
    price_dir = np.sign(np.diff(c, prepend=c[0]))
    vol_dir = np.sign(np.diff(v, prepend=v[0]))
    vp_diverge = (price_dir != vol_dir).astype(float)
    # NEW: Rolling volatility percentile
    vol_12m = pd.Series(c).rolling(12).std().values
    vol_12m_pct = np.nan_to_num(
        (vol_12m - pd.Series(vol_12m).rolling(60).min().values) /
        np.maximum(pd.Series(vol_12m).rolling(60).max().values - pd.Series(vol_12m).rolling(60).min().values, 0.01), 0.5)
    # NEW: Range change ratio
    current_range = h - l
    prev_range = np.roll(h, 1) - np.roll(l, 1)
    range_chg = np.nan_to_num(current_range / np.maximum(prev_range, 0.01) - 1, 0)
    # NEW: Gap ratio (open vs prev close)
    gap = np.nan_to_num((o - np.roll(c, 1)) / np.maximum(np.abs(np.roll(c, 1)), 0.01), 0)
    # NEW: Hurst exponent proxy (rescaled range)
    rs_12 = np.zeros(n)
    for i in range(60, n):
        seg = c[max(0,i-11):i+1]
        if len(seg) >= 12:
            m = seg.mean(); dev = seg - m
            r = dev.max() - dev.min(); s = seg.std()
            rs_12[i] = np.log(max(r, 0.01) / max(s, 0.01)) / np.log(len(seg))
    # NEW: Price acceleration (2nd derivative)
    p_accel = np.zeros(n)
    for i in range(2, n):
        p_accel[i] = (c[i] - 2*c[i-1] + c[i-2]) / max(abs(c[i-2]), 0.01)
    # NEW: Turnover acceleration
    to_accel = np.zeros(n)
    for i in range(2, n):
        to_accel[i] = (tr[i] - 2*tr[i-1] + tr[i-2]) / max(tr[i-2], 0.001) if not np.isnan(tr[i-2]) and tr[i-2] > 0 else 0

    features = np.zeros((n, 47), dtype=np.float32)
    for i in range(n):
        f = [
            # [0:17] Standard technical
            (c[i]-c[i-1])/max(abs(c[i-1]),0.01) if i>=1 else 0,
            (c[i]-c[i-3])/max(abs(c[i-3]),0.01) if i>=3 else 0,
            (c[i]-c[i-6])/max(abs(c[i-6]),0.01) if i>=6 else 0,
            (c[i]-c[i-12])/max(abs(c[i-12]),0.01) if i>=12 else 0,
            (c[i]-ma5[i])/max(abs(c[i]),0.01) if not np.isnan(ma5[i]) else 0,
            (c[i]-ma20[i])/max(abs(c[i]),0.01) if not np.isnan(ma20[i]) else 0,
            (c[i]-ma60[i])/max(abs(c[i]),0.01) if not np.isnan(ma60[i]) else 0,
            dif[i] if not np.isnan(dif[i]) else 0,
            dea[i] if not np.isnan(dea[i]) else 0,
            macd_hist[i] if not np.isnan(macd_hist[i]) else 0,
            rsi14[i] if not np.isnan(rsi14[i]) else 50,
            bb_pos[i] if not np.isnan(bb_pos[i]) else 0.5,
            np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0,
            atr14[i]/max(abs(c[i]),0.01) if not np.isnan(atr14[i]) else 0,
            (h[i]-l[i])/max(abs(c[i]),0.01),
            1.0 if c[i]>ma20[i] else 0.0,
            1.0 if c[i]>ma60[i] else 0.0,
            # [17:24] Volume & turnover (7 features)
            v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0,
            tr[i] if not np.isnan(tr[i]) else 0,
            tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0,
            (vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0,
            np.log1p(max(v[i],1)),
            np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0,
            body_pct[i] if not np.isnan(body_pct[i]) else 0,
            # [24:30] Price position & trends (6 features)
            (c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 and not np.isnan(p5h[i]) else 0.5,
            (ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 and not np.isnan(ma20[i]) else 0,
            np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0,
            ma5[i]/max(ma20[i],0.01)-1 if i>=20 and not np.isnan(ma5[i]) and not np.isnan(ma20[i]) else 0,
            ma10[i]/max(ma60[i],0.01)-1 if i>=60 and not np.isnan(ma10[i]) and not np.isnan(ma60[i]) else 0,
            1.0 if c[i]>ma5[i] else 0.0,
            # [30:36] Candle & patterns (6 features)
            1.0 if c[i]>ma10[i] else 0.0,
            upper_shadow[i] if not np.isnan(upper_shadow[i]) else 0,
            lower_shadow[i] if not np.isnan(lower_shadow[i]) else 0,
            up_streak[i]/12.0,
            dn_streak[i]/12.0,
            (h[i]-l[i])/max(h[i-1]-l[i-1],0.01)-1 if i>=1 else 0,
            # [36:40] NEW: ADX + CCI (4 features)
            adx14[i]/100.0 if not np.isnan(adx14[i]) else 0,
            cci20[i]/200.0 if not np.isnan(cci20[i]) else 0,
            tp[i]/max(tp_sma20[i],0.01)-1 if i>=20 and not np.isnan(tp_sma20[i]) else 0,
            bb_width[i] if not np.isnan(bb_width[i]) else 0,
            # [40:47] NEW: Volume advanced (7 features)
            obv_ratio[i] if not np.isnan(obv_ratio[i]) else 1,
            vp_diverge[i],
            vol_12m_pct[i] if not np.isnan(vol_12m_pct[i]) else 0.5,
            range_chg[i] if not np.isnan(range_chg[i]) else 0,
            gap[i] if not np.isnan(gap[i]) else 0,
            rs_12[i] if not np.isnan(rs_12[i]) else 0,
            p_accel[i] if not np.isnan(p_accel[i]) else 0,
        ]
        features[i] = f
    return np.nan_to_num(features, 0.0)

def cs_ic(pred, true, dates):
    return np.mean([spearmanr(pred[dates==m],true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20])

# ======== Data Loading ========
conn=sqlite3.connect(DB)
codes=[r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params=','.join('?'*len(codes))
df=pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date",conn,params=codes)
conn.close()

# FFT feature extractor
N_FFT = 10
def fft_features(prices):
    x=np.arange(len(prices)); trend=np.polyfit(x,prices,1); detrended=prices-np.polyval(trend,x)
    fft_p=np.fft.rfft(detrended); amps=np.abs(fft_p); freqs=np.fft.rfftfreq(len(detrended))
    if len(amps)<=1: return np.zeros(N_FFT*3,dtype=np.float32)
    pk=np.argsort(amps[1:])[::-1][:N_FFT]+1; feats=[]
    for idx in pk:
        if idx<len(freqs): feats.extend([freqs[idx],amps[idx],np.angle(fft_p[idx])])
    while len(feats)<N_FFT*3: feats.extend([0,0,0])
    return np.array(feats[:N_FFT*3],dtype=np.float32)

print('Building features...', flush=True); t0=time.time()
flat_list, ys_list, dates_list = [], [], []
SEQ_LEN = 60
for code in codes:
    g=df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g)<72: continue
    c=g['close'].values.astype(float); o=g['open'].values.astype(float)
    h=g['high'].values.astype(float); l=g['low'].values.astype(float)
    v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c)
    F = build_features(c, o, h, l, v, tr)
    c_clean = pywt.wavedec(c,'db4',level=2)
    c_clean = pywt.waverec([pywt.threshold(c,0.5*np.sqrt(2*np.log(n)),mode='soft') if i>0 else c for i,c in enumerate(c_clean)],'db4')[:n]

    for i in range(SEQ_LEN-1, n-6):
        if c[i]<=0.01:continue
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
        # Merge: 47 tech + 30 FFT = 77 features
        fft = fft_features(c_clean[i-SEQ_LEN+1:i+1])
        merged = np.concatenate([F[i], fft])
        flat_list.append(merged); ys_list.append(np.clip(fwd_raw,-2,2))
        dates_list.append(g['date'].iloc[i])

flat=np.array(flat_list,dtype=np.float32); ys=np.array(ys_list,dtype=np.float32)
v=~np.isnan(flat).any(axis=1)&~np.isnan(ys); flat=flat[v]; ys=ys[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12'); va_m=(dates_arr>='2022-01')&(dates_arr<='2023-12'); te_m=(dates_arr>='2024-01')
print(f'Data: {len(flat):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} features={flat.shape[1]} ({time.time()-t0:.0f}s)', flush=True)

sc=StandardScaler(); Xt=sc.fit_transform(flat[tr_m]); Xv=sc.transform(flat[va_m]); Xte=sc.transform(flat[te_m])
y_tr=ys[tr_m]; y_va=ys[va_m]; y_te=ys[te_m]; te_dates=dates_arr[te_m]

# ======== Compare: 66d vs 85d ========
# Use first 40 features as "old", all 47 as "new" (since 47 is what we had before the 19 new ones)
# Actually, let's do feature ablation: try subsets and all
def train_lgb(X_tr, y_tr, X_te, y_te, te_dates, label, n_jobs=4):
    t0=time.time()
    m=lgb.LGBMRegressor(objective='regression',metric='l1',num_leaves=63,learning_rate=0.03,
        n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,
        random_state=456,verbosity=-1,n_jobs=n_jobs)
    m.fit(X_tr, y_tr); p=m.predict(X_te)
    ic=cs_ic(p,y_te,te_dates)
    hit=np.mean((p>0)==(y_te>0))
    top20=np.argsort(p)[-int(len(p)*0.2):]
    ls=np.mean(y_te[top20])-np.mean(y_te[np.argsort(p)[:int(len(p)*0.2)]])
    print(f'  {label:<35s} CS_IC={ic:+.4f}  Hit={hit:.3f}  Top20LS={ls:+.4f}  ({time.time()-t0:.0f}s)', flush=True)
    return ic, hit, ls

print('\nModel comparison:', flush=True)
results = []

# Baseline: old 40-feature set (before Phase1.2 extensions)
train_lgb(Xt[:,:40], y_tr, Xte[:,:40], y_te, te_dates, 'LGB-40d (old baseline)')

# Current: 66d (after initial Phase1.2)
train_lgb(Xt, y_tr, Xte, y_te, te_dates, 'LGB-47d (current)')

# XGBoost on full features
t0=time.time(); xm=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,
    n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xm.fit(Xt,y_tr); px=xm.predict(Xte); ic_x=cs_ic(px,y_te,te_dates)
print(f'  XGB-47d{" "*28} CS_IC={ic_x:+.4f}  Hit={np.mean((px>0)==(y_te>0)):.3f}  ({time.time()-t0:.0f}s)', flush=True)

# Ensemble search
print('\nEnsemble:', flush=True)
lgb0=lgb.LGBMRegressor(objective='regression',metric='l1',num_leaves=63,learning_rate=0.03,
    n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,
    random_state=456,verbosity=-1,n_jobs=4)
lgb0.fit(Xt,y_tr); pl=lgb0.predict(Xte)

xgb0=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,
    n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xgb0.fit(Xt,y_tr); px0=xgb0.predict(Xte)

best_ic, best_w = -99, 0.5
for w in np.arange(0, 1.05, 0.05):
    ic = cs_ic(w*pl + (1-w)*px0, y_te, te_dates)
    if ic > best_ic: best_ic = ic; best_w = w

ens = best_w*pl + (1-best_w)*px0
en_ic = cs_ic(ens, y_te, te_dates)
en_hit = np.mean((ens>0)==(y_te>0))
print(f'  Ensemble LGB+{best_w:.2f}+XGB+{1-best_w:.2f}: CS_IC={en_ic:+.4f}  Hit={en_hit:.3f}')

# Feature importance analysis
print('\nFeature importance (top 15):', flush=True)
imp = lgb0.feature_importances_
top15 = np.argsort(imp)[::-1][:15]
names = {
    0:'r1',1:'r3',2:'r6',3:'r12',4:'ma5d',5:'ma20d',6:'ma60d',
    7:'macd_dif',8:'macd_dea',9:'macd_hist',10:'rsi14',11:'bb_pos',
    12:'vol_6m',13:'atr_ratio',14:'hilo',15:'above_ma20',16:'above_ma60',
    17:'vol_chg',18:'turnover',19:'to_chg',20:'vol_ratio',21:'log_vol',22:'log_to',
    23:'body_pct',24:'price_pos5y',25:'trend20_60',26:'vol_12m',
    27:'ma5_20',28:'ma10_60',29:'above_ma5',30:'above_ma10',
    31:'upper_shadow',32:'lower_shadow',33:'up_streak',34:'dn_streak',
    35:'range_chg',36:'adx14',37:'cci20',38:'tp_dev',39:'bb_width',
    40:'obv_ratio',41:'vp_diverge',42:'vol_pct',43:'range_chg2',44:'gap',45:'hurst',46:'p_accel'
}
for idx in top15:
    name = names.get(idx, f'f{idx}')
    bar = '█' * int(imp[idx]/imp[top15[0]]*30)
    print(f'  {idx:3d} {name:<20s} {imp[idx]:8.1f} {bar}')

# Baseline comparison
print(f'\n{"="*60}')
print(f'Path A Results')
print(f'  Old baseline (40d):  used in early experiments')
print(f'  Current (47d):       after Phase 1.2 initial extension')
print(f'  Best ensemble CS_IC: {max(en_ic, ic_x):+.4f}')
print(f'  Prev best CS_IC:     +0.178 (LGB+XGB+MLP+Ridge on 56d)')
print(f'{"="*60}')
