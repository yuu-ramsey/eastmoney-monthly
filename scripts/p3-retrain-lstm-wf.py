"""Walk-forward LSTM retrain (no leakage) + predict on v2 pool."""
import json, sys, time, torch, numpy as np, pandas as pd, sqlite3
from pathlib import Path
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
conn = sqlite3.connect(str(DB))
codes = [r[0] for r in conn.execute("SELECT DISTINCT code FROM monthly_klines").fetchall()]
print(f"Codes: {len(codes)}")

df = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume FROM monthly_klines WHERE code IN ({','.join('?'*len(codes))}) AND date>='2005-01' ORDER BY code,date", conn, params=codes)
conn.close()
print(f"Rows: {len(df)}")

LOOKBACK, HIDDEN = 60, 64
all_seqs, all_targets, all_dates = [], [], []
for code, grp in df.groupby('code'):
    grp = grp.sort_values('date').reset_index(drop=True)
    if len(grp) < LOOKBACK + 24: continue
    n = len(grp); dates = grp['date'].tolist()
    c = grp['close'].values.astype(float); o = grp['open'].values.astype(float)
    h = grp['high'].values.astype(float); l = grp['low'].values.astype(float)
    v = grp['volume'].values.astype(float)
    feats = np.zeros((n, 10), dtype=np.float32)
    def rz(s, w=60):
        m = pd.Series(s).rolling(w, min_periods=w).mean()
        s2 = pd.Series(s).rolling(w, min_periods=w).std()
        return ((s - m) / s2.clip(1e-8)).fillna(0).values
    feats[:,0]=rz(c); feats[:,1]=rz(o); feats[:,2]=rz(h); feats[:,3]=rz(l); feats[:,4]=rz(v)
    e12=pd.Series(c).ewm(span=12).mean().values; e26=pd.Series(c).ewm(span=26).mean().values
    dif=np.nan_to_num(e12-e26,0); dea=pd.Series(dif).ewm(span=9).mean().values
    feats[:,5]=dif; feats[:,6]=dea; feats[:,7]=(dif-dea)*2
    dl=np.diff(c,prepend=c[0]); ga=pd.Series(np.where(dl>0,dl,0)).ewm(alpha=1/14).mean().values
    lo2=pd.Series(np.where(dl<0,-dl,0)).ewm(alpha=1/14).mean().values
    feats[:,8]=np.nan_to_num(100-100/(1+ga/np.maximum(lo2,1e-8)),50)
    ma60=pd.Series(c).rolling(60).mean().values
    feats[:,9]=np.nan_to_num((c-ma60)/np.maximum(ma60,0.01),0)
    feats=np.nan_to_num(feats,0.0)
    for i in range(LOOKBACK-1,n-12):
        if c[i]<=0.01: continue
        all_seqs.append(feats[i-LOOKBACK+1:i+1])
        all_targets.append(np.clip((c[i+6]-c[i])/max(c[i],0.01),-2,2))
        all_dates.append(dates[i])

X=np.array(all_seqs,dtype=np.float32); y=np.array(all_targets,dtype=np.float32); da=np.array(all_dates)
tm=(da>='2010-01')&(da<='2021-12'); vm=(da>='2022-01')&(da<='2023-12')
X_tr=X[tm]; y_tr=y[tm]; X_va=X[vm]; y_va=y[vm]
print(f"Train:{len(X_tr)} Val:{len(X_va)}")

import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self): super().__init__(); self.l=nn.LSTM(10,HIDDEN,2,batch_first=True,dropout=0.2); self.h=nn.Linear(HIDDEN,1)
    def forward(self,x): o,_=self.l(x); return self.h(o[:,-1,:]).squeeze(-1)

model=LSTM().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
ds_tr=torch.utils.data.TensorDataset(torch.from_numpy(X_tr),torch.from_numpy(y_tr))
ds_va=torch.utils.data.TensorDataset(torch.from_numpy(X_va),torch.from_numpy(y_va))
ld_tr=torch.utils.data.DataLoader(ds_tr,256,shuffle=True,pin_memory=True)
ld_va=torch.utils.data.DataLoader(ds_va,256,pin_memory=True)

best_ic,best_ep,ni=-1,0,0
for ep in range(1,60):
    model.train()
    for Xb,yb in ld_tr: Xb,yb=Xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); loss=nn.MSELoss()(model(Xb),yb); loss.backward(); opt.step()
    model.eval(); preds,targs=[],[]
    with torch.no_grad():
        for Xb,yb in ld_va: preds.append(model(Xb.to(DEVICE)).cpu().numpy()); targs.append(yb.numpy())
    ic=spearmanr(np.concatenate(preds),np.concatenate(targs))[0]
    if ep<=3 or ep%10==0: print(f"Ep{ep}: IC={ic:.4f}")
    if ic>best_ic: best_ic,best_ep,ni=ic,ep,0; torch.save(model.state_dict(),str(PROJECT/'.eastmoney-ai'/'lstm'/'lstm_wf.pt'))
    else: ni+=1
    if ni>=10 and ep>10: break
print(f"Best IC:{best_ic:.4f} ep{best_ep}")

# Predict v2 pool
model.load_state_dict(torch.load(str(PROJECT/'.eastmoney-ai'/'lstm'/'lstm_wf.pt'),map_location=DEVICE)); model.eval()
with open(PROJECT/'data'/'frozen-eval-lowpos-v2-baostock.json') as f: pool=json.load(f)
conn=sqlite3.connect(str(DB)); v2c=list(set(t['stockCode'] for t in pool['testPoints']))
m_df=pd.read_sql_query(f"SELECT code,date,close,open,high,low,volume FROM monthly_klines WHERE code IN ({','.join('?'*len(v2c))}) AND date>='2005-01' ORDER BY code,date",conn,params=v2c); conn.close()

lp={}; t0=time.time()
for code,grp in m_df.groupby('code'):
    grp=grp.sort_values('date').reset_index(drop=True)
    if len(grp)<LOOKBACK+6: continue
    n=len(grp); dates=grp['date'].tolist(); c=grp['close'].values.astype(float)
    o=grp['open'].values.astype(float); h=grp['high'].values.astype(float)
    l=grp['low'].values.astype(float); v=grp['volume'].values.astype(float)
    feats=np.zeros((n,10),dtype=np.float32)
    def rz(s,w=60): m=pd.Series(s).rolling(w,min_periods=w).mean(); s2=pd.Series(s).rolling(w,min_periods=w).std(); return((s-m)/s2.clip(1e-8)).fillna(0).values
    feats[:,0]=rz(c);feats[:,1]=rz(o);feats[:,2]=rz(h);feats[:,3]=rz(l);feats[:,4]=rz(v)
    e12=pd.Series(c).ewm(span=12).mean().values;e26=pd.Series(c).ewm(span=26).mean().values
    dif=np.nan_to_num(e12-e26,0);dea=pd.Series(dif).ewm(span=9).mean().values
    feats[:,5]=dif;feats[:,6]=dea;feats[:,7]=(dif-dea)*2
    ma60=pd.Series(c).rolling(60).mean().values;feats[:,9]=np.nan_to_num((c-ma60)/np.maximum(ma60,0.01),0)
    feats=np.nan_to_num(feats,0.0)
    for tp in [t for t in pool['testPoints'] if t['stockCode']==code and t['alpha'] is not None]:
        ci=-1
        for j,d in enumerate(dates):
            if str(d).startswith(tp['cutoffDate']): ci=j; break
        if ci<LOOKBACK-1: continue
        seq=feats[ci-LOOKBACK+1:ci+1]
        with torch.no_grad(): lp[f"{code}|{tp['cutoffDate']}"]=float(model(torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)).cpu().numpy()[0])
print(f"LSTM-WF:{len(lp)} preds,{(time.time()-t0):.0f}s")

out=json.load(open(PROJECT/'data'/'p3-kronos-lstm-signals.json')); out['lstm_wf']=lp
json.dump(out,open(PROJECT/'data'/'p3-kronos-lstm-signals.json','w'))
print(f"Saved")
