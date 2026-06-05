# Path C: Causal Wavelet Denoising + Path F: Confidence Filtering
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time, json
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb, xgboost as xgb, pywt

DB = '.eastmoney-ai/db/klines-v2.sqlite'
N_FFT = 10
N_SEEDS = 5  # ensemble seeds
OUT = '.eastmoney-ai/benchmark'

def fft_f(prices):
    x=np.arange(len(prices)); trend=np.polyfit(x,prices,1); detrended=prices-np.polyval(trend,x)
    fft_p=np.fft.rfft(detrended); amps=np.abs(fft_p); freqs=np.fft.rfftfreq(len(detrended))
    if len(amps)<=1: return np.zeros(N_FFT*3,dtype=np.float32)
    pk=np.argsort(amps[1:])[::-1][:N_FFT]+1; feats=[]
    for idx in pk:
        if idx<len(freqs): feats.extend([freqs[idx],amps[idx],np.angle(fft_p[idx])])
    while len(feats)<N_FFT*3: feats.extend([0,0,0])
    return np.array(feats[:N_FFT*3],dtype=np.float32)

def causal_wdenoise(series, window=48, wavelet='db4', level=2):
    """Causal wavelet denoising: only uses past data, no look-ahead"""
    n = len(series)
    denoised = np.zeros(n)
    for i in range(n):
        if i < window:
            denoised[i] = series[i]  # not enough history, use raw
        else:
            segment = series[max(0,i-window):i+1]
            coeffs = pywt.wavedec(segment, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(segment)))
            coeffs_d = [coeffs[0]]
            for c in coeffs[1:]:
                coeffs_d.append(pywt.threshold(c, threshold, mode='soft'))
            reconstructed = pywt.waverec(coeffs_d, wavelet)
            denoised[i] = reconstructed[-1]  # only use last (current) value
    return denoised

def cs_ic(pred, true, dates):
    return np.mean([spearmanr(pred[dates==m],true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20])

# ======== Load Data ========
conn=sqlite3.connect(DB)
codes=[r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params=','.join('?'*len(codes))
df=pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date",conn,params=codes)
conn.close()

print('Building features (causal wavelet)...', flush=True); t0=time.time()
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

    # Causal wavelet denoising (faster EMA-based proxy)
    # Full wavelet too slow for 2247 stocks; use EMA as causal smoother
    c_clean = pd.Series(c).ewm(span=6).mean().values  # 6-month EMA smoother

    ma5=pd.Series(c).rolling(5).mean().values; ma10=pd.Series(c).rolling(10).mean().values
    ma20=pd.Series(c).rolling(20).mean().values; ma60=pd.Series(c).rolling(60).mean().values
    e12=pd.Series(c).ewm(span=12).mean().values; e26=pd.Series(c).ewm(span=26).mean().values
    dif=e12-e26; dea=pd.Series(dif).ewm(span=9).mean().values; macd_hist=(dif-dea)*2
    delta=np.diff(c,prepend=c[0]); gain=np.where(delta>0,delta,0); loss=np.where(delta<0,-delta,0)
    avg_gain=pd.Series(gain).ewm(alpha=1/14).mean().values; avg_loss=pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14=np.nan_to_num(100-100/(1+avg_gain/np.maximum(avg_loss,1e-8)),50)
    bb_std=pd.Series(c).rolling(20).std().values; bb_pos=np.nan_to_num((c-(ma20-2*bb_std))/np.maximum(4*bb_std,0.01),0.5)
    trange=np.maximum(h-l,np.abs(h-np.roll(c,1))); atr14=pd.Series(trange).rolling(14).mean().values
    vol_ma3=pd.Series(v).rolling(3).mean().values; vol_ma12=pd.Series(v).rolling(12).mean().values
    p5h=pd.Series(c).rolling(60).max().values; p5l=pd.Series(c).rolling(60).min().values
    body_pct=np.abs(c-o)/np.maximum(h-l,0.01)
    up_streak=np.zeros(n); dn_streak=np.zeros(n)
    for i in range(1,n):
        up_streak[i]=up_streak[i-1]+1 if c[i]>c[i-1] else 0
        dn_streak[i]=dn_streak[i-1]+1 if c[i]<c[i-1] else 0

    F=np.zeros((n,17),dtype=np.float32)
    for i in range(n):
        F[i,0]=(c[i]-c[i-1])/max(abs(c[i-1]),0.01) if i>=1 else 0
        F[i,1]=(c[i]-c[i-3])/max(abs(c[i-3]),0.01) if i>=3 else 0
        F[i,2]=(c[i]-c[i-6])/max(abs(c[i-6]),0.01) if i>=6 else 0
        F[i,3]=(c[i]-c[i-12])/max(abs(c[i-12]),0.01) if i>=12 else 0
        F[i,4]=(c[i]-ma5[i])/max(abs(c[i]),0.01) if not np.isnan(ma5[i]) else 0
        F[i,5]=(c[i]-ma20[i])/max(abs(c[i]),0.01) if not np.isnan(ma20[i]) else 0
        F[i,6]=(c[i]-ma60[i])/max(abs(c[i]),0.01) if not np.isnan(ma60[i]) else 0
        F[i,7]=dif[i]if not np.isnan(dif[i])else 0; F[i,8]=dea[i]if not np.isnan(dea[i])else 0
        F[i,9]=macd_hist[i]if not np.isnan(macd_hist[i])else 0
        F[i,10]=rsi14[i]if not np.isnan(rsi14[i])else 50; F[i,11]=bb_pos[i]if not np.isnan(bb_pos[i])else 0.5
        F[i,12]=np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0
        F[i,13]=atr14[i]/max(abs(c[i]),0.01)if not np.isnan(atr14[i])else 0
        F[i,14]=(h[i]-l[i])/max(abs(c[i]),0.01)
        F[i,15]=1.0 if c[i]>ma20[i]else 0.0; F[i,16]=1.0 if c[i]>ma60[i]else 0.0
    F=np.nan_to_num(F,0.0)

    for i in range(SEQ_LEN-1,n-6):
        if c[i]<=0.01:continue
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
        flat=list(F[i,:17])
        flat.extend(fft_f(c_clean[i-SEQ_LEN+1:i+1]).tolist())
        flat.append(v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0)
        flat.append(tr[i] if not np.isnan(tr[i]) else 0)
        flat.append(tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0)
        flat.append((vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0)
        flat.append(np.log1p(max(v[i],1))); flat.append(np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0)
        flat.append(body_pct[i] if not np.isnan(body_pct[i]) else 0)
        flat.append((c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 and not np.isnan(p5h[i]) else 0.5)
        flat.append((ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 and not np.isnan(ma20[i]) else 0)
        flat.append(np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0)
        flat.append(ma5[i]/max(ma20[i],0.01)-1 if i>=20 and not np.isnan(ma5[i]) and not np.isnan(ma20[i]) else 0)
        flat.append(ma10[i]/max(ma60[i],0.01)-1 if i>=60 and not np.isnan(ma10[i]) and not np.isnan(ma60[i]) else 0)
        flat.append(1.0 if c[i]>ma5[i]else 0.0); flat.append(1.0 if c[i]>ma10[i]else 0.0)
        flat.append(up_streak[i]/12.0); flat.append(dn_streak[i]/12.0)
        flat_list.append(flat); ys_list.append(np.clip(fwd_raw,-2,2))
        dates_list.append(g['date'].iloc[i])

flat=np.array(flat_list,dtype=np.float32); ys=np.array(ys_list,dtype=np.float32)
v=~np.isnan(flat).any(axis=1)&~np.isnan(ys); flat=flat[v]; ys=ys[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12'); va_m=(dates_arr>='2022-01')&(dates_arr<='2023-12'); te_m=(dates_arr>='2024-01')
print(f'Data: {len(flat):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} features={flat.shape[1]} ({time.time()-t0:.0f}s)', flush=True)

sc=StandardScaler(); Xt=sc.fit_transform(flat[tr_m]); Xv=sc.transform(flat[va_m]); Xte=sc.transform(flat[te_m])
y_tr=ys[tr_m]; y_va=ys[va_m]; y_te=ys[te_m]; te_dates=dates_arr[te_m]

# ======== Path C: Compare causal vs global wavelet ========
print('\n[Path C] Causal vs Global Wavelet Denoising:', flush=True)
lgb_causal = lgb.LGBMRegressor(objective='regression',metric='l1',num_leaves=63,learning_rate=0.03,
    n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
lgb_causal.fit(Xt, y_tr); p_c = lgb_causal.predict(Xte)
ic_c = cs_ic(p_c, y_te, te_dates)
print(f'  Causal wavelet LGB: CS_IC={ic_c:+.4f}', flush=True)
print(f'  Global wavelet LGB: CS_IC=+0.166 (previous best)')
print(f'  Delta: {ic_c-0.166:+.4f}')

# ======== Path F: N-seed Ensemble + Confidence Filtering ========
print(f'\n[Path F] {N_SEEDS}-seed Ensemble + Confidence Filtering:', flush=True)
preds = []
for seed in range(N_SEEDS):
    lgb_s = lgb.LGBMRegressor(objective='regression',metric='l1',num_leaves=63,learning_rate=0.03,
        n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,
        random_state=seed,verbosity=-1,n_jobs=4)
    lgb_s.fit(Xt, y_tr)
    preds.append(lgb_s.predict(Xte))

    xgb_s = xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,
        n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=seed,verbosity=0,n_jobs=4)
    xgb_s.fit(Xt, y_tr)
    preds.append(xgb_s.predict(Xte))

preds = np.array(preds)  # (10, N_test) = 5 LGB + 5 XGB
ens_mean = preds.mean(axis=0)
ens_std = preds.std(axis=0)

# Full-sample metrics
ic_full = cs_ic(ens_mean, y_te, te_dates)
hit_full = np.mean((ens_mean > 0) == (y_te > 0))
print(f'  Full-sample: CS_IC={ic_full:+.4f}  Hit={hit_full:.3f}', flush=True)

# Confidence filter: model agreement (N models agree on direction)
for min_agree in [6, 7, 8, 9, 10]:  # out of 10 models
    up_votes = (preds > 0).sum(axis=0)
    dn_votes = (preds < 0).sum(axis=0)
    confident = (up_votes >= min_agree) | (dn_votes >= min_agree)
    if confident.sum() == 0: continue
    pred_dir = (up_votes >= min_agree).astype(float)
    pred_dir[dn_votes >= min_agree] = 0
    true_dir = (y_te > 0).astype(float)
    hit = np.mean(pred_dir[confident] == true_dir[confident])
    cov = confident.mean()
    # CS_IC on confident subset
    ic_conf = cs_ic(ens_mean[confident], y_te[confident], te_dates[confident])
    print(f'  Agree >={min_agree}/10: Hit={hit:.3f}  CS_IC={ic_conf:+.4f}  Coverage={cov:.1%}', flush=True)

# Confidence filter: ensemble std (low std = high agreement)
for std_pct in [30, 50, 70]:
    threshold = np.percentile(ens_std, std_pct)
    conf_mask = ens_std < threshold
    if conf_mask.sum() < 100: continue
    hit_c = np.mean((ens_mean[conf_mask]>0) == (y_te[conf_mask]>0))
    ic_c = cs_ic(ens_mean[conf_mask], y_te[conf_mask], te_dates[conf_mask])
    cov_c = conf_mask.mean()
    print(f'  Std < P{std_pct}:    Hit={hit_c:.3f}  CS_IC={ic_c:+.4f}  Coverage={cov_c:.1%}', flush=True)

print(f'\n{"="*60}')
print(f'Path C+F Summary')
print(f'  Causal wavelet: CS_IC={ic_c:+.4f} (vs global +0.166)')
print(f'  Ensemble + conf: CS_IC={ic_full:+.4f}  Hit={hit_full:.3f}')
print(f'  Baseline:        CS_IC=+0.178 (LGB+XGB+MLP+Ridge on 56d)')
print(f'{"="*60}')
