# Phase 3: Factor-Level Neutralization (not prediction-level)
# Neutralize feature values per cross-section, then retrain + re-evaluate
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb, xgboost as xgb, pywt

DB = '.eastmoney-ai/db/klines-v2.sqlite'; N_FFT = 10

def cs_ic(pred, true, dates):
    ics = [spearmanr(pred[dates==m], true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20]
    ics = np.array(ics); return np.mean(ics), np.mean(ics)/np.std(ics) if np.std(ics)>0 else 0, ics

def fft_f(p):
    x=np.arange(len(p));t=np.polyfit(x,p,1);d=p-np.polyval(t,x)
    fp=np.fft.rfft(d);a=np.abs(fp);fq=np.fft.rfftfreq(len(d))
    if len(a)<=1: return np.zeros(N_FFT*3,dtype=np.float32)
    pk=np.argsort(a[1:])[::-1][:N_FFT]+1;fs=[]
    for i in pk:
        if i<len(fq): fs.extend([fq[i],a[i],np.angle(fp[i])])
    while len(fs)<N_FFT*3: fs.extend([0,0,0])
    return np.array(fs[:N_FFT*3],dtype=np.float32)

def wd(s):
    c=pywt.wavedec(s,'db4',level=2);sigma=np.median(np.abs(c[-1]))/0.6745;th=sigma*np.sqrt(2*np.log(len(s)))
    cd=[c[0]]+[pywt.threshold(cf,th,mode='soft') for cf in c[1:]]; return pywt.waverec(cd,'db4')[:len(s)]

def cross_sectional_neutralize(features, dates, neutralizer_values, neutralizer_type='continuous'):
    """
    Neutralize features against a variable per cross-section.
    features: (N, D) array
    dates: (N,) array
    neutralizer_values: (N,) for continuous, (N,) string for categorical
    Returns: neutralized_features (N, D)
    """
    neutralized = features.copy()
    for m in np.unique(dates):
        mask = dates == m
        if mask.sum() < 50: continue

        if neutralizer_type == 'continuous':
            # Regress each feature on the neutralizer, keep residuals
            X = neutralizer_values[mask].reshape(-1, 1)
            for d in range(features.shape[1]):
                y = features[mask, d]
                lr = LinearRegression()
                lr.fit(X, y)
                neutralized[mask, d] = y - lr.predict(X)

        elif neutralizer_type == 'categorical':
            # Subtract group mean for each category
            groups = neutralizer_values[mask]
            for g in np.unique(groups):
                gmask = mask & (neutralizer_values == g)
                if gmask.sum() >= 3:
                    neutralized[gmask] -= features[gmask].mean(axis=0)

    return neutralized

# ======== Build data ========
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
ind_map = {r[0]:r[1] for r in conn.execute('SELECT stock_code, industry_code FROM stock_industry_mapping')}
params = ','.join('?'*len(codes))
df = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2005-01' ORDER BY code,date", conn, params=codes)
conn.close()
df['month'] = df['date'].str[:7]

print(f'Building features...', flush=True); t0=time.time()
flat_list, ys_list, dates_list, inds_list, size_list = [], [], [], [], []
for code in codes:
    g = df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    c=g['close'].values.astype(float);o=g['open'].values.astype(float);h=g['high'].values.astype(float)
    l=g['low'].values.astype(float);v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c);cc=wd(c);industry=ind_map.get(code,'unknown')
    ma5=pd.Series(c).rolling(5).mean().values;ma20=pd.Series(c).rolling(20).mean().values;ma60=pd.Series(c).rolling(60).mean().values
    e12=pd.Series(c).ewm(span=12).mean().values;e26=pd.Series(c).ewm(span=26).mean().values
    dif=e12-e26;dea=pd.Series(dif).ewm(span=9).mean().values;macd_hist=(dif-dea)*2
    delta=np.diff(c,prepend=c[0]);gain=np.where(delta>0,delta,0);loss=np.where(delta<0,-delta,0)
    avg_gain=pd.Series(gain).ewm(alpha=1/14).mean().values;avg_loss=pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14=np.nan_to_num(100-100/(1+avg_gain/np.maximum(avg_loss,1e-8)),50)
    bb_std=pd.Series(c).rolling(20).std().values;bb_pos=np.nan_to_num((c-(ma20-2*bb_std))/np.maximum(4*bb_std,0.01),0.5)
    trange=np.maximum(h-l,np.abs(h-np.roll(c,1)));atr14=pd.Series(trange).rolling(14).mean().values
    vol_ma3=pd.Series(v).rolling(3).mean().values;vol_ma12=pd.Series(v).rolling(12).mean().values
    p5h=pd.Series(c).rolling(60).max().values;p5l=pd.Series(c).rolling(60).min().values
    body_pct=np.abs(c-o)/np.maximum(h-l,0.01)
    up_streak=np.zeros(n);dn_streak=np.zeros(n)
    for i in range(1,n): up_streak[i]=up_streak[i-1]+1 if c[i]>c[i-1] else 0;dn_streak[i]=dn_streak[i-1]+1 if c[i]<c[i-1] else 0
    for i in range(60,n-6):
        if c[i]<=0.01:continue
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
        flat=[(c[i]-c[i-j])/max(abs(c[i-j]),0.01) if i>=j else 0 for j in [1,3,6,12]]
        for ma in [ma5,ma20,ma60]: flat.append((c[i]-ma[i])/max(abs(c[i]),0.01) if not np.isnan(ma[i]) else 0)
        flat.extend([dif[i]if not np.isnan(dif[i])else 0,dea[i]if not np.isnan(dea[i])else 0,macd_hist[i]if not np.isnan(macd_hist[i])else 0])
        flat.append(rsi14[i]if not np.isnan(rsi14[i])else 50);flat.append(bb_pos[i]if not np.isnan(bb_pos[i])else 0.5)
        flat.append(np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0)
        flat.append(atr14[i]/max(abs(c[i]),0.01)if not np.isnan(atr14[i])else 0);flat.append((h[i]-l[i])/max(abs(c[i]),0.01))
        flat.append(1.0 if c[i]>ma20[i]else 0.0);flat.append(1.0 if c[i]>ma60[i]else 0.0)
        flat.extend(fft_f(cc[i-60+1:i+1]).tolist())
        flat.extend([v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0,tr[i] if not np.isnan(tr[i]) else 0,
            tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0,
            (vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0,np.log1p(max(v[i],1)),np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0])
        flat.extend([body_pct[i] if not np.isnan(body_pct[i]) else 0,(c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 else 0.5,
            (ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 else 0,np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0,
            ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0,1.0 if c[i]>ma5[i]else 0.0,up_streak[i]/12.0,dn_streak[i]/12.0])
        flat_list.append(flat);ys_list.append(np.clip(fwd_raw,-2,2));dates_list.append(g['month'].iloc[i])
        inds_list.append(industry);size_list.append(np.log1p(v[i]))

flat=np.array(flat_list,dtype=np.float32);ys=np.array(ys_list,dtype=np.float32)
v=~np.isnan(flat).any(axis=1)&~np.isnan(ys);flat=flat[v];ys=ys[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
inds_arr=np.array([inds_list[i] for i in range(len(v)) if v[i]])
size_arr=np.array([size_list[i] for i in range(len(v)) if v[i]])
print(f'{len(flat):,} samples, {flat.shape[1]}d ({time.time()-t0:.0f}s)', flush=True)

# Split
tr_m=(dates_arr>='2010-01')&(dates_arr<='2014-12');te_m=(dates_arr>='2015-01')

# Create 4 feature sets
# 0: Original (no neutralization)
# 1: Size-neutralized
# 2: Industry-neutralized
# 3: Both

print('Neutralizing factors...', flush=True); t0=time.time()
# Do neutralization on ALL data (train+test) per cross-section to avoid look-ahead
flat_size = cross_sectional_neutralize(flat.copy(), dates_arr, size_arr, 'continuous')
flat_ind = cross_sectional_neutralize(flat.copy(), dates_arr, inds_arr, 'categorical')
flat_both = cross_sectional_neutralize(flat_size.copy(), dates_arr, inds_arr, 'categorical')
print(f'Neutralization done ({time.time()-t0:.0f}s)', flush=True)

def train_and_eval(X_tr, X_te, label):
    sc=StandardScaler();Xt=sc.fit_transform(X_tr);Xte=sc.transform(X_te)
    lgb_m=lgb.LGBMRegressor(objective='regression',num_leaves=63,learning_rate=0.03,n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
    lgb_m.fit(Xt,y_tr);p_lgb=lgb_m.predict(Xte)
    xgb_m=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
    xgb_m.fit(Xt,y_tr);p_xgb=xgb_m.predict(Xte)
    ridge_m=Ridge(alpha=1.0);ridge_m.fit(Xt,y_tr);p_ridge=ridge_m.predict(Xte)
    mlp_m=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,random_state=456)
    mlp_m.fit(Xt,y_tr);p_mlp=mlp_m.predict(Xte)
    ics={}
    for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge),('MLP',p_mlp)]:
        ic_tmp=np.mean([spearmanr(p[te_dates==m],y_te[te_dates==m])[0] for m in np.unique(te_dates) if (te_dates==m).sum()>=20])
        ics[n]=max(ic_tmp,0)
    w_sum=sum(ics.values())
    p_ens=sum(ics[n]*p for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge),('MLP',p_mlp)])/w_sum
    ic,icir,ics_arr=cs_ic(p_ens,y_te,te_dates)
    return ic,icir,ics_arr,ics

y_tr=ys[tr_m];y_te=ys[te_m];te_dates=dates_arr[te_m]

results = []
all_preds = {}
for feat_set, label in [(flat,'Raw'),(flat_size,'Size Neut'),(flat_ind,'Industry Neut'),(flat_both,'Both Neut')]:
    print(f'  {label}...', flush=True)
    X_tr=feat_set[tr_m];X_te=feat_set[te_m]
    ic,icir,ics_arr,model_ics=train_and_eval(X_tr,X_te,label)
    results.append({'label':label,'IC':ic,'ICIR':icir,'IC>0':np.mean(ics_arr>0),'N_months':len(ics_arr),'models':model_ics})
    # Get the ensemble predictions for later regime breakdown
    sc_tmp=StandardScaler();sc_tmp.fit(X_tr);Xte_t=sc_tmp.transform(X_te)
    all_preds[label] = (Xte_t, model_ics)

# ======== Report ========
print(); print('='*70)
print('PHASE 3: FACTOR-LEVEL NEUTRALIZATION (retrain after neutralize)')
print('='*70)
print(f'  {"Method":<18s} {"IC":>8s} {"ICIR":>8s} {"IC>0":>8s} {"Months":>6s} {"vs Raw IC":>10s}')
print(f'  {"-"*60}')
raw_ic = results[0]['IC']
for r in results:
    delta = r['IC'] - raw_ic
    print(f'  {r["label"]:<18s} {r["IC"]:+8.4f} {r["ICIR"]:+8.3f} {r["IC>0"]:7.0%} {r["N_months"]:>6d} {delta:+10.4f}')

# Annual breakdown
print(f'\n  {"Year":<8s} {"Raw":>8s} {"SizeN":>8s} {"IndN":>8s} {"BothN":>8s}')
for yr in ['2015','2016','2017','2018','2019','2020','2021','2022','2023','2024','2025']:
    mask = np.array([str(d).startswith(yr) for d in te_dates])
    if mask.sum() < 200: continue
    row = f'  {yr:<8s}'
    for r_idx, preds_var in enumerate([p_raw_final, p_size_final, p_ind_final, p_both_final]):
        ic,_,_ = cs_ic(preds_var[mask], y_te[mask], te_dates[mask])
        row += f' {ic:+8.4f}'
    print(row)

print(f'\n  Verdict:')
delta_full = results[-1]['IC'] - raw_ic
if delta_full > -0.02:
    print(f'  PASS - Full neutralization IC change={delta_full:+.4f} (within -0.02 threshold)')
    print(f'  Signal is NOT just size/industry proxy. Independent alpha exists.')
else:
    print(f'  FAIL - IC dropped by {abs(delta_full):.4f}, signal may be size/industry proxy')
print('='*70)
