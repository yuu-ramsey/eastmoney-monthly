"""LightGBM binary classification on daily features. Much faster than LSTM."""
import json, time, numpy as np, pandas as pd, sqlite3
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import lightgbm as lgb

PROJECT = Path(__file__).resolve().parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
conn = sqlite3.connect(str(DB))
codes = [r[0] for r in conn.execute("SELECT DISTINCT code FROM daily_klines").fetchall()]
np.random.seed(42); sc = np.random.choice(codes, min(2000, len(codes)), replace=False).tolist()
d_df = pd.read_sql_query(f"SELECT code,date,close,open,high,low,volume FROM daily_klines WHERE code IN ({','.join('?'*len(sc))}) AND date>='2010-01-01' ORDER BY code,date", conn, params=sc)
conn.close()
print(f"Daily: {len(d_df)} rows, {d_df['code'].nunique()} codes")

FWD_DAYS = 126
all_feats, all_labels, all_dates = [], [], []

for code, grp in d_df.groupby('code'):
    grp = grp.sort_values('date').reset_index(drop=True)
    if len(grp) < 252 + FWD_DAYS + 60: continue
    n = len(grp); dates = grp['date'].tolist()
    c = grp['close'].values.astype(float); h = grp['high'].values.astype(float)
    l = grp['low'].values.astype(float); v = grp['volume'].values.astype(float)

    for i in range(252, n - FWD_DAYS - 1):
        if c[i] <= 0.01: continue
        past_c = c[max(0,i-252):i]
        if len(past_c) < 60: continue

        ret_1m = (c[i]-c[i-20])/max(c[i-20],0.01) if i>=20 else 0
        ret_3m = (c[i]-c[i-60])/max(c[i-60],0.01) if i>=60 else 0
        ret_6m = (c[i]-c[max(0,i-126)])/max(c[max(0,i-126)],0.01)
        vol_20d = np.std(np.diff(c[max(0,i-20):i+1])/np.maximum(c[max(0,i-20):i],0.01))
        ma20 = np.mean(c[max(0,i-20):i+1]); ma60 = np.mean(c[max(0,i-60):i+1]) if i>=60 else ma20
        ma_pos_20 = (c[i]-ma20)/max(ma20,0.01); ma_pos_60 = (c[i]-ma60)/max(ma60,0.01)
        pos_in_252 = (c[i]-np.min(past_c))/max(np.max(past_c)-np.min(past_c),0.01)
        vol_ratio = np.mean(v[max(0,i-5):i+1])/max(np.mean(v[max(0,i-20):i+1]),1)
        vol_trend = np.mean(v[max(0,i-10):i+1])/max(np.mean(v[max(0,i-60):i+1]),1) if i>=60 else 1
        pos_days_20 = np.mean((np.diff(c[max(0,i-20):i+1])>0).astype(float))
        up_dn = np.sum(np.maximum(np.diff(c[max(0,i-20):i+1]),0))/max(np.sum(np.maximum(-np.diff(c[max(0,i-20):i+1]),0)),0.01)
        e12 = pd.Series(c[max(0,i-26):i+1]).ewm(span=12).mean().values[-1]
        e26 = pd.Series(c[max(0,i-26):i+1]).ewm(span=26).mean().values[-1]
        macd = (e12-e26)/max(c[i],0.01)
        dl = np.diff(c[max(0,i-14):i+1])
        gs = np.sum(dl[dl>0]) if np.any(dl>0) else 0; ls2 = -np.sum(dl[dl<0]) if np.any(dl<0) else 0
        rsi = 100-100/(1+gs/max(ls2,0.01)) if ls2>0 else 50
        hl_range = (h[i]-l[i])/max(c[i],0.01)
        hl_ratio = np.mean((h[max(0,i-20):i+1]-l[max(0,i-20):i+1])/np.maximum(c[max(0,i-20):i+1],0.01))

        feat = [ret_1m,ret_3m,ret_6m,vol_20d,ma_pos_20,ma_pos_60,pos_in_252,vol_ratio,vol_trend,pos_days_20,up_dn,macd,(rsi-50)/50,hl_range,hl_ratio]
        if np.any(np.isnan(feat)): continue

        sigma = np.std(np.diff(c[max(0,i-252):i+1])/np.maximum(c[max(0,i-252):i],0.01))
        if sigma<0.005: sigma=0.02
        upper=c[i]*(1+1.5*sigma); lower=c[i]*(1-1.5*sigma)
        label=None
        for j in range(i+1,min(i+FWD_DAYS+1,n)):
            if h[j]>=upper: label=1; break
            if l[j]<=lower: label=0; break
        if label is None: continue
        all_feats.append(feat); all_labels.append(label); all_dates.append(dates[i])

X=np.array(all_feats,dtype=np.float32); y=np.array(all_labels); da=np.array(all_dates)
print(f"Samples: {len(X)} bull={sum(y==1)} ({100*sum(y==1)/len(y):.0f}%) bear={sum(y==0)}")

tm=(da>='2018-01-01')&(da<='2021-12-31'); vm=(da>='2022-01-01')&(da<='2023-12-31')
X_tr,y_tr=X[tm],y[tm]; X_va,y_va=X[vm],y[vm]
print(f"Train: {len(X_tr)}, Val: {len(X_va)}")

maj_class=np.argmax(np.bincount(y_tr)); maj_ba=balanced_accuracy_score(y_va,np.full_like(y_va,maj_class))
print(f"Always-majority: class={maj_class} ba={maj_ba:.4f}")

td=lgb.Dataset(X_tr,label=y_tr); vd=lgb.Dataset(X_va,label=y_va,reference=td)
params={'objective':'binary','boosting_type':'gbdt','num_leaves':63,'learning_rate':0.05,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'min_data_in_leaf':100,'verbose':-1}
model=lgb.train(params,td,num_boost_round=200,valid_sets=[vd],callbacks=[lgb.early_stopping(10),lgb.log_evaluation(30)])

probs=model.predict(X_va); preds=(probs>0.5).astype(int)
ba=balanced_accuracy_score(y_va,preds); cm=confusion_matrix(y_va,preds)
print(f"\nL1: ba={ba:.4f} vs maj={maj_ba:.4f}\nCM:\n{cm}")
if ba<=maj_ba: print("L1 FAILED"); exit(1)
print("L1 PASSED ✓")

print("\nPredicting 24tp...")
pool=json.load(open(PROJECT/'data'/'frozen-eval-lowpos-v2-24tp.json'))
conn=sqlite3.connect(str(DB)); v2c=list(set(t['stockCode'] for t in pool['testPoints']))
d2=pd.read_sql_query(f"SELECT code,date,close,open,high,low,volume FROM daily_klines WHERE code IN ({','.join('?'*len(v2c))}) AND date>='2008-01-01' ORDER BY code,date",conn,params=v2c); conn.close()
lp={}; t0=time.time()
for code,grp in d2.groupby('code'):
    grp=grp.sort_values('date').reset_index(drop=True)
    if len(grp)<252+FWD_DAYS+60: continue
    n=len(grp); dates=grp['date'].tolist(); c=grp['close'].values.astype(float)
    h=grp['high'].values.astype(float); l=grp['low'].values.astype(float); v=grp['volume'].values.astype(float)
    for tp in [t for t in pool['testPoints'] if t['stockCode']==code and t['alpha'] is not None]:
        ci=-1
        for j,d in enumerate(dates):
            if str(d).startswith(tp['cutoffDate']): ci=j; break
        if ci<252: continue
        past_c=c[max(0,ci-252):ci]
        ret_1m=(c[ci]-c[ci-20])/max(c[ci-20],0.01) if ci>=20 else 0
        ret_3m=(c[ci]-c[ci-60])/max(c[ci-60],0.01) if ci>=60 else 0
        ret_6m=(c[ci]-c[max(0,ci-126)])/max(c[max(0,ci-126)],0.01)
        vol_20d=np.std(np.diff(c[max(0,ci-20):ci+1])/np.maximum(c[max(0,ci-20):ci],0.01))
        ma20=np.mean(c[max(0,ci-20):ci+1]); ma60=np.mean(c[max(0,ci-60):ci+1]) if ci>=60 else ma20
        ma_pos_20=(c[ci]-ma20)/max(ma20,0.01); ma_pos_60=(c[ci]-ma60)/max(ma60,0.01)
        pos_in_252=(c[ci]-np.min(past_c))/max(np.max(past_c)-np.min(past_c),0.01)
        vol_ratio=np.mean(v[max(0,ci-5):ci+1])/max(np.mean(v[max(0,ci-20):ci+1]),1)
        vol_trend=np.mean(v[max(0,ci-10):ci+1])/max(np.mean(v[max(0,ci-60):ci+1]),1) if ci>=60 else 1
        pos_days_20=np.mean((np.diff(c[max(0,ci-20):ci+1])>0).astype(float))
        up_dn=np.sum(np.maximum(np.diff(c[max(0,ci-20):ci+1]),0))/max(np.sum(np.maximum(-np.diff(c[max(0,ci-20):ci+1]),0)),0.01)
        e12=pd.Series(c[max(0,ci-26):ci+1]).ewm(span=12).mean().values[-1]
        e26=pd.Series(c[max(0,ci-26):ci+1]).ewm(span=26).mean().values[-1]
        macd=(e12-e26)/max(c[ci],0.01)
        dl2=np.diff(c[max(0,ci-14):ci+1]); gs2=np.sum(dl2[dl2>0]) if np.any(dl2>0) else 0
        ls3=-np.sum(dl2[dl2<0]) if np.any(dl2<0) else 0
        rsi=100-100/(1+gs2/max(ls3,0.01)) if ls3>0 else 50
        hl_range=(h[ci]-l[ci])/max(c[ci],0.01)
        hl_ratio=np.mean((h[max(0,ci-20):ci+1]-l[max(0,ci-20):ci+1])/np.maximum(c[max(0,ci-20):ci+1],0.01))
        feat=[ret_1m,ret_3m,ret_6m,vol_20d,ma_pos_20,ma_pos_60,pos_in_252,vol_ratio,vol_trend,pos_days_20,up_dn,macd,(rsi-50)/50,hl_range,hl_ratio]
        if np.any(np.isnan(feat)): continue
        prob=model.predict([feat])[0]
        lp[f"{code}|{tp['cutoffDate']}"]=float(prob*2-1)
print(f"LGB: {len(lp)} preds, {(time.time()-t0):.0f}s")
out=json.load(open(PROJECT/'data'/'p3-kronos-lstm-signals.json')); out['lgb']=lp
json.dump(out,open(PROJECT/'data'/'p3-kronos-lstm-signals.json','w'))
print("Saved")
