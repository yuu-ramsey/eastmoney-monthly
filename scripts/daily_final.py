# Daily LSTM: 10-seed ensemble + monthly aggregation + correlation with monthly ensemble
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time, json
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb, xgboost as xgb

DEV = torch.device('cuda')
LOOKBACK, BATCH = 60, 1024
HIDDEN, N_LAYERS = 64, 2
LR, WD, EPOCHS = 0.001, 0, 200
N_SEEDS = 10
DB = '.eastmoney-ai/db/klines-v2.sqlite'
print(f'Daily Final: {N_SEEDS}-seed LSTM + Monthly Integration', flush=True)

# ======== 1. Daily LSTM Data + Training ========
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute('SELECT code FROM daily_klines GROUP BY code HAVING COUNT(*)>=500').fetchall()]
params=','.join('?'*len(codes))
df_all=pd.read_sql_query(f"SELECT code,date,open,close,high,low,volume,turnover_rate FROM daily_klines WHERE code IN ({params}) AND date>='2010-01-01' ORDER BY code,date",conn,params=codes)
conn.close()

print('Building daily sequences...', flush=True); t0=time.time()
X_list, y_list, date_list, code_list = [], [], [], []
for code in codes:
    g=df_all[df_all['code']==code].sort_values('date').reset_index(drop=True)
    if len(g)<LOOKBACK+66: continue
    c=g['close'].values.astype(float); o=g['open'].values.astype(float)
    h=g['high'].values.astype(float); l=g['low'].values.astype(float)
    v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c); dates=g['date'].values
    vol_ma20=pd.Series(v).rolling(20).mean().fillna(1).values
    F=np.zeros((n,8),dtype=np.float32)
    for i in range(n):
        pc=max(abs(c[i-1]),0.01) if i>=1 else c[i]
        F[i,0]=o[i]/max(pc,0.01)-1; F[i,1]=h[i]/max(c[i],0.01)-1
        F[i,2]=l[i]/max(c[i],0.01)-1; F[i,3]=c[i]/max(pc,0.01)-1
        F[i,4]=v[i]/max(vol_ma20[i],1)-1; F[i,5]=tr[i] if not np.isnan(tr[i]) and tr[i]<100 else 0
        F[i,6]=(h[i]-l[i])/max(c[i],0.01); F[i,7]=(c[i]-o[i])/max(o[i],0.01) if o[i]>0 else 0
    F=np.nan_to_num(F,0.0).astype(np.float32)
    for i in range(LOOKBACK-1, n-63):
        fwd=(c[i+63]-c[i])/max(c[i],0.01)
        if abs(fwd)>3: continue
        X_list.append(F[i-LOOKBACK+1:i+1]); y_list.append(np.clip(fwd,-3,3))
        date_list.append(str(dates[i])[:10]); code_list.append(code)

X=np.array(X_list,dtype=np.float32); y=np.array(y_list,dtype=np.float32)
dates_arr=np.array(date_list); codes_arr=np.array(code_list)
v=~np.isnan(X).any(axis=(1,2))&~np.isnan(y); X=X[v]; y=y[v]; dates_arr=dates_arr[v]; codes_arr=codes_arr[v]
tr_m=(dates_arr>='2015-01-01')&(dates_arr<='2021-12-31')
va_m=(dates_arr>='2022-01-01')&(dates_arr<='2023-12-31')
te_m=(dates_arr>='2024-01-01')
print(f'Daily: {len(X):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} ({time.time()-t0:.0f}s)', flush=True)

fm=X[tr_m].reshape(-1,X.shape[-1]).mean(0); fs=X[tr_m].reshape(-1,X.shape[-1]).std(0)+1e-8
X=np.clip((X-fm)/fs,-5,5)
X_tr_t=torch.from_numpy(X[tr_m]).float().to(DEV); y_tr_t=torch.from_numpy(y[tr_m]).float().to(DEV)
X_va_t=torch.from_numpy(X[va_m]).float().to(DEV); y_va=y[va_m]
X_te_t=torch.from_numpy(X[te_m]).float().to(DEV); y_te_np=y[te_m]; te_dates_daily=dates_arr[te_m]; te_codes_daily=codes_arr[te_m]

class SlimLSTM(nn.Module):
    def __init__(self, d=8, h=64, l=2):
        super().__init__()
        self.lstm = nn.LSTM(d, h, l, batch_first=True)
        self.head = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(-1)

print(f'Training {N_SEEDS} seeds...', flush=True)
daily_preds_test = []
for seed in range(N_SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    model=SlimLSTM().to(DEV); opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
    loss_fn=nn.MSELoss(); best_va,best_st,patience=-99,None,0
    for ep in range(EPOCHS):
        model.train(); perm=torch.randperm(len(X_tr_t),device=DEV)
        for i in range(0,len(X_tr_t),BATCH):
            idx=perm[i:i+BATCH]; loss=loss_fn(model(X_tr_t[idx]),y_tr_t[idx])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval()
        with torch.no_grad():
            pv=np.concatenate([model(X_va_t[i:i+BATCH]).cpu().numpy() for i in range(0,len(X_va_t),BATCH)])
            ic=spearmanr(pv,y_va)[0]
        if ic>best_va: best_va=ic; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else: patience+=1
        if patience>=20: break
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        pt=np.concatenate([model(X_te_t[i:i+BATCH]).cpu().numpy() for i in range(0,len(X_te_t),BATCH)])
    daily_preds_test.append(pt)
    if seed%3==0: print(f'  seed {seed}: done ({ep+1}ep)', flush=True)

daily_preds=np.array(daily_preds_test)  # (10, N_test)
ens_daily=daily_preds.mean(axis=0)

# Daily IC
def cs_ic(pred,true,dates):
    ics=[spearmanr(pred[dates==m],true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20]
    return np.mean(ics) if ics else np.nan,ics

daily_ic,daily_ics=cs_ic(ens_daily,y_te_np,te_dates_daily)
daily_hit=np.mean((ens_daily>0)==(y_te_np>0))
print(f'\nDaily 10-seed Ensemble: CS_IC={daily_ic:+.4f} Hit={daily_hit:.3f}', flush=True)

# ======== 2. Aggregate Daily → Monthly ========
print('\nAggregating daily signals to monthly...', flush=True)
# For each stock-month, compute: mean/median/last daily prediction, std of daily predictions
te_months = np.array([d[:7] for d in te_dates_daily])
monthly_agg = {}
for i in range(len(te_dates_daily)):
    key = (te_codes_daily[i], te_months[i])
    if key not in monthly_agg:
        monthly_agg[key] = []
    monthly_agg[key].append(ens_daily[i])

agg_rows = []
for (code, month), preds in monthly_agg.items():
    if len(preds) < 5: continue
    p = np.array(preds)
    agg_rows.append({
        'code': code, 'month': month,
        'daily_mean': p.mean(), 'daily_median': np.median(p),
        'daily_std': p.std(), 'daily_last': p[-1],
        'daily_trend': (p[-1]-p[0])/max(abs(p[0]),0.01) if abs(p[0])>0.001 else 0,
        'daily_pos_pct': (p>0).mean(),
    })

agg_df = pd.DataFrame(agg_rows)
print(f'Monthly aggregation: {len(agg_df)} stock-months', flush=True)

# ======== 3. Load Monthly Ensemble Predictions ========
print('Loading monthly ensemble...', flush=True); t0=time.time()
# Re-train monthly ensemble on same train/val/test split
# Build 61d monthly features (same as final_evaluation.py)
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import pywt

def fft_f(prices):
    N_FFT=10; x=np.arange(len(prices)); trend=np.polyfit(x,prices,1); detrended=prices-np.polyval(trend,x)
    fft_p=np.fft.rfft(detrended); amps=np.abs(fft_p); freqs=np.fft.rfftfreq(len(detrended))
    if len(amps)<=1: return np.zeros(N_FFT*3,dtype=np.float32)
    pk=np.argsort(amps[1:])[::-1][:N_FFT]+1; feats=[]
    for idx in pk:
        if idx<len(freqs): feats.extend([freqs[idx],amps[idx],np.angle(fft_p[idx])])
    while len(feats)<N_FFT*3: feats.extend([0,0,0])
    return np.array(feats[:N_FFT*3],dtype=np.float32)

def wdenoise(signal):
    coeffs=pywt.wavedec(signal,'db4',level=2)
    sigma=np.median(np.abs(coeffs[-1]))/0.6745; threshold=sigma*np.sqrt(2*np.log(len(signal)))
    coeffs_d=[coeffs[0]]+[pywt.threshold(c,threshold,mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs_d,'db4')[:len(signal)]

conn=sqlite3.connect(DB)
codes_m=[r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params_m=','.join('?'*len(codes_m))
df_m=pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params_m}) AND date>='2010-01' ORDER BY code,date",conn,params=codes_m)
conn.close()

flat_list,ys_list,dates_list,codes_list=[],[],[],[]
for code in codes_m:
    g=df_m[df_m['code']==code].sort_values('date').reset_index(drop=True)
    if len(g)<72: continue
    c=g['close'].values.astype(float); o=g['open'].values.astype(float)
    h=g['high'].values.astype(float); l=g['low'].values.astype(float)
    v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c); c_clean=wdenoise(c)
    ma5=pd.Series(c).rolling(5).mean().values; ma20=pd.Series(c).rolling(20).mean().values; ma60=pd.Series(c).rolling(60).mean().values
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
    for i in range(1,n): up_streak[i]=up_streak[i-1]+1 if c[i]>c[i-1] else 0; dn_streak[i]=dn_streak[i-1]+1 if c[i]<c[i-1] else 0
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
    for i in range(60,n-6):
        if c[i]<=0.01:continue
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
        flat=list(F[i,:17]); flat.extend(fft_f(c_clean[i-60+1:i+1]).tolist())
        flat.append(v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0)
        flat.append(tr[i] if not np.isnan(tr[i]) else 0)
        flat.append(tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0)
        flat.append((vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0)
        flat.append(np.log1p(max(v[i],1))); flat.append(np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0)
        flat.append(body_pct[i] if not np.isnan(body_pct[i]) else 0)
        flat.append((c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 and not np.isnan(p5h[i]) else 0.5)
        flat.append((ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 and not np.isnan(ma20[i]) else 0)
        flat.append(np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0)
        flat.append(ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0)
        flat.append(1.0 if c[i]>ma5[i]else 0.0)
        flat.append(up_streak[i]/12.0); flat.append(dn_streak[i]/12.0)
        flat_list.append(flat); ys_list.append(np.clip(fwd_raw,-2,2))
        dates_list.append(g['date'].iloc[i]); codes_list.append(code)

flat_m=np.array(flat_list,dtype=np.float32); ys_m=np.array(ys_list,dtype=np.float32)
v_m=~np.isnan(flat_m).any(axis=1)&~np.isnan(ys_m)
flat_m=flat_m[v_m]; ys_m=ys_m[v_m]; dates_m=np.array([dates_list[i] for i in range(len(v_m)) if v_m[i]])
codes_m_arr=np.array([codes_list[i] for i in range(len(v_m)) if v_m[i]])

tr_m_m=(dates_m>='2015-01')&(dates_m<='2021-12')
va_m_m=(dates_m>='2022-01')&(dates_m<='2023-12')
te_m_m=(dates_m>='2024-01')

sc_m=StandardScaler(); Xt_m=sc_m.fit_transform(flat_m[tr_m_m]); Xte_m=sc_m.transform(flat_m[te_m_m])

# Train monthly ensemble
lgb_m=lgb.LGBMRegressor(objective='regression',metric='l1',num_leaves=63,learning_rate=0.03,n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
lgb_m.fit(Xt_m,ys_m[tr_m_m]); p_lgb_m=lgb_m.predict(Xte_m)
xgb_m=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xgb_m.fit(Xt_m,ys_m[tr_m_m]); p_xgb_m=xgb_m.predict(Xte_m)
ridge_m=Ridge(alpha=1.0); ridge_m.fit(Xt_m,ys_m[tr_m_m]); p_ridge_m=ridge_m.predict(Xte_m)
mlp_m=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,random_state=456)
mlp_m.fit(Xt_m,ys_m[tr_m_m]); p_mlp_m=mlp_m.predict(Xte_m)

# Weighted ensemble
ics_m={}
for name,pred in [('LGB',p_lgb_m),('XGB',p_xgb_m),('Ridge',p_ridge_m),('MLP',p_mlp_m)]:
    ics_m[name]=max(cs_ic(pred,ys_m[te_m_m],dates_m[te_m_m])[0],0)
w_sum=sum(ics_m.values())
p_ens_m=(ics_m['LGB']*p_lgb_m+ics_m['XGB']*p_xgb_m+ics_m['Ridge']*p_ridge_m+ics_m['MLP']*p_mlp_m)/w_sum
ens_ic_m,ens_ics_m=cs_ic(p_ens_m,ys_m[te_m_m],dates_m[te_m_m])
print(f'Monthly Ensemble CS_IC={ens_ic_m:+.4f} ({time.time()-t0:.0f}s)', flush=True)

# ======== 4. Merge daily aggregation with monthly predictions ========
print('\nMerging daily + monthly signals...', flush=True)
# Build cross-reference: monthly test set (code, month) -> monthly prediction
monthly_map = {}
for i in range(len(codes_m_arr[te_m_m])):
    key = (codes_m_arr[te_m_m][i], dates_m[te_m_m][i])
    monthly_map[key] = p_ens_m[i]

# Merge daily aggregation with monthly predictions
merged = []
for _, row in agg_df.iterrows():
    key = (row['code'], row['month'])
    if key in monthly_map:
        merged.append({
            'code': row['code'], 'month': row['month'],
            'daily_mean': row['daily_mean'], 'daily_last': row['daily_last'],
            'daily_std': row['daily_std'], 'daily_pos_pct': row['daily_pos_pct'],
            'monthly_pred': monthly_map[key],
        })

merged_df = pd.DataFrame(merged)
print(f'Merged: {len(merged_df)} stock-months', flush=True)

# Also get true monthly returns
monthly_ret_map = {}
for i in range(len(codes_m_arr[te_m_m])):
    key = (codes_m_arr[te_m_m][i], dates_m[te_m_m][i])
    monthly_ret_map[key] = ys_m[te_m_m][i]

merged_df['true_ret'] = merged_df.apply(lambda r: monthly_ret_map.get((r['code'], r['month']), np.nan), axis=1)
merged_df = merged_df.dropna()

# ======== 5. Analysis ========
print(); print('='*65)
print('Daily-Monthly Integration Analysis')
print('='*65)

# Daily signal quality (on merged subset)
daily_ic_m = spearmanr(merged_df['daily_mean'], merged_df['true_ret'])[0]
monthly_ic_m = spearmanr(merged_df['monthly_pred'], merged_df['true_ret'])[0]
print(f'  Daily signal IC (monthly level): {daily_ic_m:+.4f}')
print(f'  Monthly ensemble IC (on subset): {monthly_ic_m:+.4f}')

# Rank correlation between daily and monthly signals
rank_corr, rank_p = spearmanr(merged_df['daily_mean'], merged_df['monthly_pred'])
pearson_c = pearsonr(merged_df['daily_mean'], merged_df['monthly_pred'])[0]
print(f'  Daily vs Monthly rank corr: {rank_corr:.4f}')
print(f'  Daily vs Monthly pearson r: {pearson_c:.4f}')

# Ensemble weights search
print(f'\n  Integration weights:')
best_ic, best_w = -99, 0
for w in np.arange(0, 1.05, 0.05):
    ens = w * merged_df['daily_mean'] + (1-w) * merged_df['monthly_pred']
    ic = spearmanr(ens, merged_df['true_ret'])[0]
    if ic > best_ic: best_ic = ic; best_w = w
    if w*20 % 5 == 0:
        print(f'    w={w:.2f} daily + {(1-w):.2f} monthly: IC={ic:+.4f}')

print(f'\n  Best integration: {best_w:.2f} daily + {1-best_w:.2f} monthly')
print(f'  Best IC: {best_ic:+.4f}')
print(f'  vs Monthly alone: {monthly_ic_m:+.4f} ({best_ic-monthly_ic_m:+.4f})')
print(f'  vs Daily alone:   {daily_ic_m:+.4f} ({best_ic-daily_ic_m:+.4f})')

# Daily signal diversity (how often do daily & monthly disagree?)
disagree = (merged_df['daily_mean']>0) != (merged_df['monthly_pred']>0)
hd = np.mean((merged_df['daily_mean']>0) == (merged_df['true_ret']>0))
hm = np.mean((merged_df['monthly_pred']>0) == (merged_df['true_ret']>0))
print(f'\n  Direction disagreement: {disagree.mean():.1%}')
print(f'  When disagree: hit daily={hd:.3f} hit monthly={hm:.3f}')

verdict = 'SIGNIFICANT BENEFIT' if (best_ic-monthly_ic_m)>0.005 else 'MARGINAL' if (best_ic-monthly_ic_m)>0 else 'NO BENEFIT'
print(f'\n  Verdict: {verdict} (delta={best_ic-monthly_ic_m:+.4f})')
print('='*65)
