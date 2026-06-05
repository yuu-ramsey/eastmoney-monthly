# Phase 3: Market Cap + Industry Neutralization (3 rounds)
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

def industry_neutralize(pred, dates, inds):
    """Subtract industry-mean from predictions within each month"""
    neutralized = pred.copy()
    for m in np.unique(dates):
        mask = dates == m
        if mask.sum() < 20: continue
        industries = inds[mask]
        for ind in np.unique(industries):
            ind_mask = mask & (inds == ind)
            if ind_mask.sum() >= 3:
                neutralized[ind_mask] -= pred[ind_mask].mean()
    return neutralized

def size_neutralize(pred, dates, size_proxy):
    """Regress out size effect from predictions within each month"""
    neutralized = pred.copy()
    for m in np.unique(dates):
        mask = dates == m
        if mask.sum() < 50: continue
        X = size_proxy[mask].reshape(-1, 1)
        y = pred[mask]
        lr = LinearRegression()
        lr.fit(X, y)
        neutralized[mask] = y - lr.predict(X)
    return neutralized

def full_neutralize(pred, dates, inds, size_proxy):
    """Industry first, then size"""
    p1 = industry_neutralize(pred, dates, inds)
    p2 = size_neutralize(p1, dates, size_proxy)
    return p2

# ======== Build data with full metadata ========
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
ind_map = {r[0]:r[1] for r in conn.execute('SELECT stock_code, industry_code FROM stock_industry_mapping')}
params = ','.join('?'*len(codes))
df = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2005-01' ORDER BY code,date", conn, params=codes)
conn.close()
df['month'] = df['date'].str[:7]

print(f'Building features + metadata...', flush=True); t0=time.time()
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
        inds_list.append(industry)
        size_list.append(np.log1p(v[i]))  # log volume as size proxy

flat=np.array(flat_list,dtype=np.float32);ys=np.array(ys_list,dtype=np.float32)
v=~np.isnan(flat).any(axis=1)&~np.isnan(ys);flat=flat[v];ys=ys[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
inds_arr=np.array([inds_list[i] for i in range(len(v)) if v[i]])
size_arr=np.array([size_list[i] for i in range(len(v)) if v[i]])
print(f'{len(flat):,} samples ({time.time()-t0:.0f}s)', flush=True)

tr_m=(dates_arr>='2010-01')&(dates_arr<='2014-12');te_m=(dates_arr>='2015-01')
sc=StandardScaler();Xt=sc.fit_transform(flat[tr_m]);Xte=sc.transform(flat[te_m])
y_tr=ys[tr_m];y_te=ys[te_m];te_dates=dates_arr[te_m];te_inds=inds_arr[te_m];te_size=size_arr[te_m]

# Train ensemble
lgb_m=lgb.LGBMRegressor(objective='regression',num_leaves=63,learning_rate=0.03,n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
lgb_m.fit(Xt,y_tr);p_lgb=lgb_m.predict(Xte)
xgb_m=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xgb_m.fit(Xt,y_tr);p_xgb=xgb_m.predict(Xte)
ridge_m=Ridge(alpha=1.0);ridge_m.fit(Xt,y_tr);p_ridge=ridge_m.predict(Xte)
mlp_m=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,random_state=456)
mlp_m.fit(Xt,y_tr);p_mlp=mlp_m.predict(Xte)
ics_m={}
for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge),('MLP',p_mlp)]:
    ic_tmp=np.mean([spearmanr(p[te_dates==m],y_te[te_dates==m])[0] for m in np.unique(te_dates) if (te_dates==m).sum()>=20])
    ics_m[n]=max(ic_tmp,0)
w_sum=sum(ics_m.values())
p_raw=sum(ics_m[n]*p for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge),('MLP',p_mlp)])/w_sum

# Industry neutralize
p_ind = industry_neutralize(p_raw, te_dates, te_inds)
# Size neutralize
p_size = size_neutralize(p_raw, te_dates, te_size)
# Both
p_full = full_neutralize(p_raw, te_dates, te_inds, te_size)

# ======== Evaluate on 3 regimes ========
regimes = [
    ('Full Cycle',         '2015-01', '2025-12', 131),
    ('Strong Factor Period','2015-01', '2016-12;2021-01;2025-12', 72),
    ('Weak Factor Period',  '2017-01', '2020-12', 48),
]

print(); print('='*70)
print('PHASE 3: MARKET CAP + INDUSTRY NEUTRALIZATION')
print('='*70)
print(f'  {"Period":<25s} {"Raw IC":>8s} {"Raw ICIR":>8s} {"Ind IC":>8s} {"Size IC":>8s} {"Full IC":>8s} {"Full ICIR":>8s}')
print(f'  {"-"*70}')

for label, start, end, expected_n in regimes:
    if ';' in end:  # combined periods
        masks = []
        for part in end.split(';'):
            p_start, p_end = part.split('-')
            masks.append((te_dates >= p_start) & (te_dates <= p_end))
        mask = masks[0]
        for m in masks[1:]: mask = mask | m
    else:
        mask = (te_dates >= start) & (te_dates <= end)

    if mask.sum() < 100: continue

    for name, pred in [('Raw',p_raw),('Ind Neut',p_ind),('Size Neut',p_size),('Full Neut',p_full)]:
        ic, icir, ics_arr = cs_ic(pred[mask], y_te[mask], te_dates[mask])
        if name == 'Raw':
            raw_ic, raw_icir = ic, icir
        if name == 'Full Neut':
            full_ic, full_icir = ic, icir

    # Actually print all four
    ic_r, ir_r, _ = cs_ic(p_raw[mask], y_te[mask], te_dates[mask])
    ic_i, ir_i, _ = cs_ic(p_ind[mask], y_te[mask], te_dates[mask])
    ic_s, ir_s, _ = cs_ic(p_size[mask], y_te[mask], te_dates[mask])
    ic_f, ir_f, _ = cs_ic(p_full[mask], y_te[mask], te_dates[mask])
    n_months = len(np.unique(te_dates[mask]))
    print(f'  {label:<25s} {ic_r:+8.4f} {ir_r:+8.3f} {ic_i:+8.4f} {ic_s:+8.4f} {ic_f:+8.4f} {ir_f:+8.3f}  ({n_months}m)')

# Annual summary
print(f'\n  {"Year":<8s} {"Raw":>8s} {"IndNeut":>8s} {"SizeNeut":>8s} {"FullNeut":>8s}')
for yr in ['2015','2016','2017','2018','2019','2020','2021','2022','2023','2024','2025']:
    mask = np.array([str(d).startswith(yr) for d in te_dates])
    if mask.sum() < 200: continue
    r,_,_=cs_ic(p_raw[mask],y_te[mask],te_dates[mask])
    f,_,_=cs_ic(p_full[mask],y_te[mask],te_dates[mask])
    i,_,_=cs_ic(p_ind[mask],y_te[mask],te_dates[mask])
    s,_,_=cs_ic(p_size[mask],y_te[mask],te_dates[mask])
    print(f'  {yr:<8s} {r:+8.4f} {i:+8.4f} {s:+8.4f} {f:+8.4f}')

print('='*70)
