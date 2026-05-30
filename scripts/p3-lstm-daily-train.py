"""LSTM-WF v2: daily klines, walk-forward, predict v2 pool."""
import json, sys, time, torch, numpy as np, pandas as pd, sqlite3
from pathlib import Path
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
conn = sqlite3.connect(str(DB))
codes = [r[0] for r in conn.execute("SELECT DISTINCT code FROM daily_klines").fetchall()]
np.random.seed(42); sample_codes = np.random.choice(codes, min(500, len(codes)), replace=False).tolist()
d_df = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume FROM daily_klines WHERE code IN ({','.join('?'*len(sample_codes))}) AND date>='2010-01-01' ORDER BY code,date", conn, params=sample_codes)
conn.close()
print(f"Daily rows: {len(d_df)} codes:{d_df['code'].nunique()}")

LOOKBACK, HIDDEN, FWD = 60, 64, 126
all_seqs, all_targets, all_dates = [], [], []
for code, grp in d_df.groupby('code'):
    grp = grp.sort_values('date').reset_index(drop=True)
    if len(grp) < LOOKBACK + FWD + 60: continue
    n = len(grp); dates = grp['date'].tolist()
    c = grp['close'].values.astype(float); o = grp['open'].values.astype(float)
    h = grp['high'].values.astype(float); l = grp['low'].values.astype(float)
    v = grp['volume'].values.astype(float)
    feats = np.zeros((n, 10), dtype=np.float32)
    ret = np.zeros(n); ret[1:] = np.clip((c[1:]-c[:-1])/np.maximum(c[:-1],0.01),-0.2,0.2)
    feats[:,0]=ret
    r5 = np.zeros(n); r5[5:] = np.clip((c[5:]-c[:-5])/np.maximum(c[:-5],0.01),-0.5,0.5)
    feats[:,1]=r5
    r20 = np.zeros(n); r20[20:] = np.clip((c[20:]-c[:-20])/np.maximum(c[:-20],0.01),-1,1)
    feats[:,2]=r20
    feats[:,3] = np.clip(pd.Series(ret).rolling(20,min_periods=5).std().fillna(0).values,0,0.1)
    v5 = pd.Series(v).rolling(5).mean().values; v20s = pd.Series(v).rolling(20).mean().values
    feats[:,4] = np.clip(v5/np.maximum(v20s,1),0,5)
    h20 = pd.Series(h).rolling(20,min_periods=5).max().values
    l20 = pd.Series(l).rolling(20,min_periods=5).min().values
    feats[:,5] = np.clip((c-l20)/np.maximum(h20-l20,0.01),0,1)
    feats[:,6] = np.clip((h-l)/np.maximum(c,0.01),0,0.2)
    gap = np.zeros(n); gap[1:] = np.clip((o[1:]-c[:-1])/np.maximum(c[:-1],0.01),-0.1,0.1)
    feats[:,7]=gap
    e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
    feats[:,8] = np.clip(np.nan_to_num((e12-e26)/np.maximum(c,0.01),0),-0.5,0.5)
    gains = pd.Series(np.where(np.diff(c,prepend=c[0])>0,np.diff(c,prepend=c[0]),0)).ewm(alpha=1/14).mean().values
    losses = pd.Series(np.where(np.diff(c,prepend=c[0])<0,-np.diff(c,prepend=c[0]),0)).ewm(alpha=1/14).mean().values
    rsi = np.nan_to_num(100-100/(1+gains/np.maximum(losses,1e-8)),50)
    feats[:,9] = np.clip((rsi-50)/50,-1,1)
    for i in range(LOOKBACK-1, n-FWD):
        if c[i]<=0.01: continue
        y6 = np.clip((c[i+FWD]-c[i])/max(c[i],0.01),-2,2)
        if np.isnan(y6): continue
        all_seqs.append(feats[i-LOOKBACK+1:i+1]); all_targets.append(y6); all_dates.append(dates[i])

X = np.array(all_seqs,dtype=np.float32); y = np.array(all_targets,dtype=np.float32); da = np.array(all_dates)
print(f"Seqs: {len(X)} y:[{y.min():.3f},{y.max():.3f}] std={y.std():.4f}")

tm = (da>='2010-01-01')&(da<='2021-12-31'); vm = (da>='2022-01-01')&(da<='2023-12-31')
X_tr, y_tr = X[tm], y[tm]; X_va, y_va = X[vm], y[vm]
print(f"Train:{len(X_tr)} Val:{len(X_va)}")

import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self, inp=10, hid=64): super().__init__(); self.l=nn.LSTM(inp,hid,2,batch_first=True,dropout=0.2); self.h=nn.Linear(hid,1)
    def forward(self,x): o,_=self.l(x); return self.h(o[:,-1,:]).squeeze(-1)

model = LSTM(10,HIDDEN).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=5,factor=0.5)
ds_tr = torch.utils.data.TensorDataset(torch.from_numpy(X_tr),torch.from_numpy(y_tr))
ds_va = torch.utils.data.TensorDataset(torch.from_numpy(X_va),torch.from_numpy(y_va))
ld_tr = torch.utils.data.DataLoader(ds_tr,256,shuffle=True); ld_va = torch.utils.data.DataLoader(ds_va,256)

best_ic, best_ep, ni = -float('inf'), 0, 0
for ep in range(1,50):
    model.train(); tl=0
    for Xb,yb in ld_tr: Xb,yb=Xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); loss=nn.MSELoss()(model(Xb),yb)
    if not torch.isnan(loss): loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl+=loss.item()*Xb.size(0)
    tl/=len(ds_tr)
    model.eval(); preds,targs,vl=[],[],0
    with torch.no_grad():
        for Xb,yb in ld_va: Xb,yb=Xb.to(DEVICE),yb.to(DEVICE); p=model(Xb); vl+=nn.MSELoss()(p,yb).item()*Xb.size(0); preds.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
    vl/=len(ds_va); preds=np.concatenate(preds); targs=np.concatenate(targs)
    p_std=preds.std(); ic=spearmanr(preds,targs)[0] if p_std>1e-8 else -1; ic=ic if not np.isnan(ic) else -1
    sched.step(ic)
    if ep<=3 or ep%10==0: print(f"Ep{ep:3d}: TL={tl:.4f} VL={vl:.4f} IC={ic:.4f} p_std={p_std:.6f}")
    if ic>best_ic: best_ic,best_ep,ni=ic,ep,0; torch.save(model.state_dict(),str(PROJECT/'.eastmoney-ai'/'lstm'/'lstm_wf_v2.pt'))
    else: ni+=1
    if ni>=10 and ep>10: break
print(f"Best IC={best_ic:.4f} ep{best_ep}")

# Predict v2
model.load_state_dict(torch.load(str(PROJECT/'.eastmoney-ai'/'lstm'/'lstm_wf_v2.pt'),map_location=DEVICE)); model.eval()
with open(PROJECT/'data'/'frozen-eval-lowpos-v2-baostock.json') as f: pool=json.load(f)
conn=sqlite3.connect(str(DB)); v2c=list(set(t['stockCode'] for t in pool['testPoints']))
d2=pd.read_sql_query(f"SELECT code,date,close,open,high,low,volume FROM daily_klines WHERE code IN ({','.join('?'*len(v2c))}) AND date>='2008-01-01' ORDER BY code,date",conn,params=v2c); conn.close()
lp={}; t0=time.time()
for code,grp in d2.groupby('code'):
    grp=grp.sort_values('date').reset_index(drop=True)
    if len(grp)<LOOKBACK+FWD+60: continue
    n=len(grp); dates=grp['date'].tolist(); c=grp['close'].values.astype(float)
    o=grp['open'].values.astype(float); h=grp['high'].values.astype(float)
    l=grp['low'].values.astype(float); v=grp['volume'].values.astype(float)
    feats=np.zeros((n,10),dtype=np.float32)
    ret=np.zeros(n); ret[1:]=np.clip((c[1:]-c[:-1])/np.maximum(c[:-1],0.01),-0.2,0.2); feats[:,0]=ret
    r5=np.zeros(n); r5[5:]=np.clip((c[5:]-c[:-5])/np.maximum(c[:-5],0.01),-0.5,0.5); feats[:,1]=r5
    r20=np.zeros(n); r20[20:]=np.clip((c[20:]-c[:-20])/np.maximum(c[:-20],0.01),-1,1); feats[:,2]=r20
    feats[:,3]=np.clip(pd.Series(ret).rolling(20,min_periods=5).std().fillna(0).values,0,0.1)
    feats[:,4]=np.clip(pd.Series(v).rolling(5).mean().values/np.maximum(pd.Series(v).rolling(20).mean().values,1),0,5)
    h20=pd.Series(h).rolling(20,min_periods=5).max().values; l20=pd.Series(l).rolling(20,min_periods=5).min().values
    feats[:,5]=np.clip((c-l20)/np.maximum(h20-l20,0.01),0,1)
    feats[:,6]=np.clip((h-l)/np.maximum(c,0.01),0,0.2)
    gap=np.zeros(n); gap[1:]=np.clip((o[1:]-c[:-1])/np.maximum(c[:-1],0.01),-0.1,0.1); feats[:,7]=gap
    e12=pd.Series(c).ewm(span=12).mean().values; e26=pd.Series(c).ewm(span=26).mean().values
    feats[:,8]=np.clip(np.nan_to_num((e12-e26)/np.maximum(c,0.01),0),-0.5,0.5)
    ga2=pd.Series(np.where(np.diff(c,prepend=c[0])>0,np.diff(c,prepend=c[0]),0)).ewm(alpha=1/14).mean().values
    lo2=pd.Series(np.where(np.diff(c,prepend=c[0])<0,-np.diff(c,prepend=c[0]),0)).ewm(alpha=1/14).mean().values
    feats[:,9]=np.clip((np.nan_to_num(100-100/(1+ga2/np.maximum(lo2,1e-8)),50)-50)/50,-1,1)
    feats=np.nan_to_num(feats,0.0)
    for tp in [t for t in pool['testPoints'] if t['stockCode']==code and t['alpha'] is not None]:
        ci=-1
        for j,d in enumerate(dates):
            if str(d).startswith(tp['cutoffDate']): ci=j; break
        if ci<LOOKBACK-1: continue
        seq=feats[ci-LOOKBACK+1:ci+1]
        with torch.no_grad(): lp[f"{code}|{tp['cutoffDate']}"]=float(model(torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)).cpu().numpy()[0])
print(f"LSTM-WF-v2:{len(lp)} preds,{(time.time()-t0):.0f}s")
out=json.load(open(PROJECT/'data'/'p3-kronos-lstm-signals.json')); out['lstm_wf_v2']=lp
json.dump(out,open(PROJECT/'data'/'p3-kronos-lstm-signals.json','w'))
print("Saved")
