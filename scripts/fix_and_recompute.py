# Step 1-4: Fix date grouping, verify, recompute all metrics
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb, xgboost as xgb, pywt

DB = '.eastmoney-ai/db/klines-v2.sqlite'; N_FFT = 10

def cs_metrics(pred, true, dates):
    """Group by YYYY-MM, compute per-month IC"""
    months = np.unique(dates)
    ics = []
    for m in months:
        mask = dates == m
        if mask.sum() >= 20:  # need at least 20 stocks per month
            ic, _ = spearmanr(pred[mask], true[mask])
            if not np.isnan(ic):
                ics.append(ic)
    ics = np.array(ics)
    ic = np.mean(ics) if len(ics) > 0 else np.nan
    icir = ic / np.std(ics) if len(ics) > 1 and np.std(ics) > 0 else 0
    rank_ic = np.mean([spearmanr(np.argsort(np.argsort(pred[dates==m])), np.argsort(np.argsort(true[dates==m])))[0] for m in months if (dates==m).sum()>=20]) if len(months)>0 else np.nan
    return ic, icir, rank_ic, ics

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

conn = sqlite3.connect(DB)

all_results = {}
for table, label in [('monthly_klines', 'Monthly'), ('weekly_klines', 'Weekly'), ('daily_klines', 'Daily')]:
    print(f'\n{"="*65}')
    print(f'{label}')
    print(f'{"="*65}', flush=True)

    # Load data
    codes = [r[0] for r in conn.execute(f'SELECT code FROM {table} GROUP BY code HAVING COUNT(*)>=84').fetchall()]
    params = ','.join('?'*len(codes))
    df = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM {table} WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date", conn, params=codes)

    # Step 1: Normalize date to YYYY-MM
    df['month'] = df['date'].str[:7]  # Extract YYYY-MM
    n_unique_raw = df['date'].nunique()
    n_unique_month = df['month'].nunique()
    print(f'  Raw dates: {n_unique_raw}, Month groups: {n_unique_month}')
    if n_unique_raw > n_unique_month * 1.5:
        print(f'  WARNING: Date granularity issue detected! {n_unique_raw} dates vs {n_unique_month} months')

    # Step 2: Per-period stock count
    monthly_counts = df.groupby('month')['code'].nunique()
    print(f'  Stock count per month: min={monthly_counts.min()}, max={monthly_counts.max()}, mean={monthly_counts.mean():.0f}, median={monthly_counts.median():.0f}')
    test_counts = monthly_counts[monthly_counts.index >= '2024-01']
    early_counts = monthly_counts[monthly_counts.index < '2015-01']
    print(f'  Pre-2015 mean: {early_counts.mean():.0f}, Post-2024 mean: {test_counts.mean():.0f}')
    print(f'  First 5: {dict(monthly_counts.head(5))}')
    print(f'  Last 5:  {dict(monthly_counts.tail(5))}')

    # Build features with corrected date grouping
    print(f'  Building features...', flush=True); t0 = time.time()
    flat_list, ys_list, dates_list = [], [], []
    for code in codes:
        g = df[df['code']==code].sort_values('date').reset_index(drop=True)
        if len(g) < 72: continue
        c=g['close'].values.astype(float);o=g['open'].values.astype(float);h=g['high'].values.astype(float)
        l=g['low'].values.astype(float);v=g['volume'].values.astype(float)
        tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
        n=len(c);cc=wd(c)
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
            # Use month (YYYY-MM) as the date key
            flat_list.append(flat);ys_list.append(np.clip(fwd_raw,-2,2));dates_list.append(g['month'].iloc[i])

    flat=np.array(flat_list,dtype=np.float32);ys=np.array(ys_list,dtype=np.float32)
    v=~np.isnan(flat).any(axis=1)&~np.isnan(ys);flat=flat[v];ys=ys[v]
    dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])

    # Verify grouping
    test_months = sorted(set(d for d in dates_arr if d >= '2024-01'))
    print(f'  Test months: {len(test_months)} ({test_months[0]} ~ {test_months[-1]})')
    counts = [(m, (dates_arr==m).sum()) for m in test_months[:5]]
    print(f'  First 5 test months: {counts}')

    # Train
    tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12');te_m=(dates_arr>='2024-01')
    sc=StandardScaler();Xt=sc.fit_transform(flat[tr_m]);Xte=sc.transform(flat[te_m])
    y_tr=ys[tr_m];y_te=ys[te_m];te_dates=dates_arr[te_m]

    lgb_m=lgb.LGBMRegressor(objective='regression',num_leaves=63,learning_rate=0.03,n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
    lgb_m.fit(Xt,y_tr);p_lgb=lgb_m.predict(Xte)
    xgb_m=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
    xgb_m.fit(Xt,y_tr);p_xgb=xgb_m.predict(Xte)
    ridge_m=Ridge(alpha=1.0);ridge_m.fit(Xt,y_tr);p_ridge=ridge_m.predict(Xte)
    mlp_m=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,random_state=456)
    mlp_m.fit(Xt,y_tr);p_mlp=mlp_m.predict(Xte)

    ics_model={}
    for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge),('MLP',p_mlp)]:
        ics_model[n]=max(cs_metrics(p,y_te,te_dates)[0],0)
    w_sum=sum(ics_model.values())
    p_ens=sum(ics_model[n]*p for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge),('MLP',p_mlp)])/w_sum

    # Step 3: Recompute all metrics
    ic, icir, rank_ic, ics_arr = cs_metrics(p_ens, y_te, te_dates)
    hit = np.mean((p_ens>0) == (y_te>0))
    n20 = max(1, int(len(p_ens)*0.2))
    ls = np.mean(y_te[np.argsort(p_ens)[-n20:]]) - np.mean(y_te[np.argsort(p_ens)[:n20]])
    ic_pos = np.mean(ics_arr > 0)
    n_test_months = len(np.unique(te_dates))

    all_results[label] = {
        'IC': ic, 'ICIR': icir, 'RankIC': rank_ic, 'Hit': hit, 'Top20LS': ls,
        'IC>0': ic_pos, 'ICstd': np.std(ics_arr), 'months': len(ics_arr),
        'stocks': len(codes), 'samples': len(y_te),
        'per_month_mean': (te_dates>= '2024-01').sum() / n_test_months if n_test_months > 0 else 0
    }

    print(f'\n  CORRECTED Metrics:')
    print(f'  IC={ic:+.4f}  ICIR={icir:+.3f}  RankIC={rank_ic:+.4f}  Hit={hit:.3f}  Top20LS={ls:+.4f}')
    print(f'  IC>0={ic_pos:.0%}  ICstd={np.std(ics_arr):.4f}  N_months={len(ics_arr)}')
    print(f'  Stocks={len(codes)}, Samples={len(y_te):,}', flush=True)

conn.close()

# Final comparison table
print(); print('='*75)
print('CORRECTED vs PREVIOUS IC COMPARISON')
print('='*75)
print(f'  {"Freq":<10s} {"Stocks":>6s} {"PerMonth":>8s} {"IC":>8s} {"ICIR":>8s} {"RankIC":>8s} {"Hit":>8s} {"Top20LS":>8s} {"IC>0":>8s} {"Months":>6s}')
print(f'  {"-"*80}')
for freq in ['Monthly', 'Weekly', 'Daily']:
    r = all_results[freq]
    print(f'  {freq:<10s} {r["stocks"]:>6,} {r["per_month_mean"]:>8.0f} {r["IC"]:+8.4f} {r["ICIR"]:+8.3f} {r["RankIC"]:+8.4f} {r["Hit"]:8.3f} {r["Top20LS"]:+8.4f} {r["IC>0"]:7.0%} {r["months"]:>6d}')

print(f'\n  Previous (inflated) values for reference:')
print(f'  Monthly: IC=+0.1766 ICIR=+2.378')
print(f'  Weekly:  IC=+0.2198 ICIR=+2.699')
print(f'  Daily:   IC=+0.2146')
print(f'  The corrected ICIR should be significantly lower if date grouping was wrong.')
print('='*75)
