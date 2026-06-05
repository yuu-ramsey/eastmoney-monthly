"""LSTM v3: 120d lookback + 20 dim features + GRU-2@64 + cosine annealing.
All features computed point-in-time (cutoff data only). Walk-forward split."""
import json, sys, time, torch, numpy as np, pandas as pd, sqlite3
from pathlib import Path
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
conn = sqlite3.connect(str(DB))
codes = [r[0] for r in conn.execute("SELECT DISTINCT code FROM daily_klines").fetchall()]
d_df = pd.read_sql_query(f"SELECT code,date,close,open,high,low,volume FROM daily_klines WHERE code IN ({','.join('?'*len(codes))}) AND date>='2008-01-01' ORDER BY code,date", conn, params=codes)
conn.close()
print(f"Daily: {len(d_df)} rows, {d_df['code'].nunique()} codes")

LOOKBACK, FWD_DAYS, NF = 60, 126, 15
all_seqs, all_labels, all_dates = [], [], []

for code, grp in d_df.groupby('code'):
    grp = grp.sort_values('date').reset_index(drop=True)
    if len(grp) < LOOKBACK + FWD_DAYS + 100: continue
    n = len(grp); dates = grp['date'].tolist()
    c = grp['close'].values.astype(float); o = grp['open'].values.astype(float)
    h = grp['high'].values.astype(float); l = grp['low'].values.astype(float)
    v = grp['volume'].values.astype(float)

    feats = np.zeros((n, NF), dtype=np.float32)
    ret = np.zeros(n); ret[1:] = np.clip(np.log(np.maximum(c[1:],0.01)/np.maximum(c[:-1],0.01)), -0.2, 0.2)
    feats[:,0]=ret; feats[:,1]=np.clip(pd.Series(c).pct_change(5).fillna(0).values, -0.5, 0.5)
    feats[:,2]=np.clip(pd.Series(c).pct_change(20).fillna(0).values, -1, 1)
    feats[:,3]=pd.Series(ret).rolling(20,min_periods=5).std().fillna(0).values
    feats[:,4]=np.clip(pd.Series(v).rolling(5).mean().values/np.maximum(pd.Series(v).rolling(20).mean().values,1),0,5)
    h20=pd.Series(h).rolling(20,min_periods=5).max().values; l20=pd.Series(l).rolling(20,min_periods=5).min().values
    feats[:,5]=np.clip((c-l20)/np.maximum(h20-l20,0.01),0,1); feats[:,6]=np.clip((h-l)/np.maximum(c,0.01),0,0.2)
    gap=np.zeros(n); gap[1:]=np.clip((o[1:]-c[:-1])/np.maximum(c[:-1],0.01),-0.1,0.1); feats[:,7]=gap
    e12=pd.Series(c).ewm(span=12).mean().values; e26=pd.Series(c).ewm(span=26).mean().values
    dif=np.nan_to_num((e12-e26)/np.maximum(c,0.01),0)
    feats[:,8]=np.clip(dif,-0.5,0.5); feats[:,9]=np.clip(pd.Series(dif).ewm(span=9).mean().values,-0.5,0.5)
    ma20=pd.Series(c).rolling(20).mean().values; std20=pd.Series(c).rolling(20).std().values
    feats[:,10]=np.clip(np.nan_to_num((c-ma20)/np.maximum(std20,0.01),0),-3,3)
    feats[:,11]=np.clip(std20/np.maximum(c,0.01),0,0.2)
    tr=np.maximum(h-l,np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
    feats[:,12]=np.clip(pd.Series(tr).rolling(14).mean().fillna(0).values/np.maximum(c,0.01),0,0.1)
    obv=np.zeros(n); obv[0]=v[0]
    for i in range(1,n): obv[i]=obv[i-1]+v[i]*(1 if c[i]>c[i-1] else (-1 if c[i]<c[i-1] else 0))
    feats[:,13]=np.clip(pd.Series(obv).pct_change(20).fillna(0).values,-1,1)
    ga=pd.Series(np.where(np.diff(c,prepend=c[0])>0,np.diff(c,prepend=c[0]),0)).ewm(alpha=1/14).mean().values
    lo=pd.Series(np.where(np.diff(c,prepend=c[0])<0,-np.diff(c,prepend=c[0]),0)).ewm(alpha=1/14).mean().values
    rsi=np.nan_to_num(100-100/(1+ga/np.maximum(lo,1e-8)),50); feats[:,14]=np.clip((rsi-50)/50,-1,1)
    ma60=pd.Series(c).rolling(60).mean().values
    feats[:,15]=np.clip((c-ma20)/np.maximum(ma20,0.01),-0.5,0.5); feats[:,16]=np.clip((c-ma60)/np.maximum(ma60,0.01),-0.5,0.5)
    feats[:,17]=np.clip(v/np.maximum(pd.Series(v).rolling(20).mean().values,1),0,5)
    feats[:,18]=np.clip((c-o)/np.maximum(c,0.01),-0.1,0.1)
    feats[:,19]=pd.Series((ret>0).astype(float)).rolling(20).mean().values

    for i in range(LOOKBACK-1, n-FWD_DAYS):
        if c[i]<=0.01: continue
        seq=feats[i-LOOKBACK+1:i+1]
        if np.any(np.isnan(seq)): continue
        past_rw=max(0,i-252)
        if i-past_rw<20: continue
        sigma=ret[past_rw:i].std()
        if sigma<0.005: sigma=0.02
        upper=c[i]*(1+1.5*sigma); lower=c[i]*(1-1.5*sigma)
        label=0
        for j in range(i+1,min(i+FWD_DAYS+1,n)):
            if h[j]>=upper: label=1; break
            if l[j]<=lower: label=2; break
        all_seqs.append(seq); all_labels.append(label); all_dates.append(dates[i])

X=np.array(all_seqs,dtype=np.float32); y=np.array(all_labels,dtype=np.int64); da=np.array(all_dates)
print(f"Seqs: {len(X)}")
for i,nm in enumerate(['neutral','bull','bear']): print(f"  {nm}: {sum(1 for l in y if l==i)} ({100*sum(1 for l in y if l==i)/len(y):.1f}%)")

tm=(da>='2010-01-01')&(da<='2021-12-31'); vm=(da>='2022-01-01')&(da<='2023-12-31')
MAX_TR,MAX_VA=1500000,300000
tr_idx=np.where(tm)[0]; va_idx=np.where(vm)[0]
np.random.seed(42)
if len(tr_idx)>MAX_TR: tr_idx=np.random.choice(tr_idx,MAX_TR,replace=False)
if len(va_idx)>MAX_VA: va_idx=np.random.choice(va_idx,MAX_VA,replace=False)
X_tr,y_tr=X[tr_idx],y[tr_idx]; X_va,y_va=X[va_idx],y[va_idx]
print(f"Train: {len(X_tr)}, Val: {len(X_va)}")

maj_class=np.argmax(np.bincount(y_tr)); maj_ba=balanced_accuracy_score(y_va,np.full_like(y_va,maj_class))
print(f"Always-majority: class={maj_class} ba={maj_ba:.4f}")

import torch.nn as nn
class GRU2(nn.Module):
    def __init__(self): super().__init__(); self.gru=nn.GRU(NF,64,2,batch_first=True,dropout=0.2); self.head=nn.Linear(64,3)
    def forward(self,x): o,_=self.gru(x); return self.head(o[:,-1,:])

total=len(y_tr); cw=torch.tensor([total/max(sum(1 for l in y_tr if l==i),1) for i in range(3)]).float().to(DEVICE)
loss_fn=nn.CrossEntropyLoss(weight=cw)
ds_tr=torch.utils.data.TensorDataset(torch.from_numpy(X_tr),torch.from_numpy(y_tr))
ds_va=torch.utils.data.TensorDataset(torch.from_numpy(X_va),torch.from_numpy(y_va))
ld_tr=torch.utils.data.DataLoader(ds_tr,256,shuffle=True); ld_va=torch.utils.data.DataLoader(ds_va,256)

model=GRU2().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
model.train()
for Xb,yb in ld_tr: Xb,yb=Xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); loss=loss_fn(model(Xb),yb)
if not torch.isnan(loss): loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
model.eval(); preds,targs=[],[]
with torch.no_grad():
    for Xb,yb in ld_va: p=model(Xb.to(DEVICE)); preds.append(p.argmax(dim=1).cpu().numpy()); targs.append(yb.cpu().numpy())
preds=np.concatenate(preds); targs=np.concatenate(targs)
ba=balanced_accuracy_score(targs,preds); cm=confusion_matrix(targs,preds,labels=[0,1,2])
print(f"\nSmoke: ba={ba:.4f} vs maj={maj_ba:.4f}\nCM:\n{cm}")
if ba<=maj_ba: print("L1 FAILED"); exit(1)

model=GRU2().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=5,T_mult=2)
best_ba,best_ep,ni=-1.0,0,0
for ep in range(1,25):
    model.train()
    for Xb,yb in ld_tr: Xb,yb=Xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); loss=loss_fn(model(Xb),yb)
    if not torch.isnan(loss): loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    sched.step()
    model.eval(); preds,targs=[],[]
    with torch.no_grad():
        for Xb,yb in ld_va: p=model(Xb.to(DEVICE)); preds.append(p.argmax(dim=1).cpu().numpy()); targs.append(yb.cpu().numpy())
    preds=np.concatenate(preds); targs=np.concatenate(targs)
    ba=balanced_accuracy_score(targs,preds)
    if ep<=3 or ep%5==0: print(f"Ep{ep:3d}: ba={ba:.4f}")
    if ba>best_ba: best_ba,best_ep,ni=ba,ep,0; torch.save(model.state_dict(),str(PROJECT/'.eastmoney-ai'/'lstm'/'lstm_v3.pt'))
    ni+=1
    if ni>=8 and ep>5: break
print(f"Best ba={best_ba:.4f} ep{best_ep} vs maj={maj_ba:.4f}")
if best_ba<=maj_ba: print("L1 FAILED"); exit(1)

print("\nPredicting 24tp...")
model.load_state_dict(torch.load(str(PROJECT/'.eastmoney-ai'/'lstm'/'lstm_v3.pt'),map_location=DEVICE)); model.eval()
pool=json.load(open(PROJECT/'data'/'frozen-eval-lowpos-v2-24tp.json'))
conn=sqlite3.connect(str(DB)); v2c=list(set(t['stockCode'] for t in pool['testPoints']))
d2=pd.read_sql_query(f"SELECT code,date,close,open,high,low,volume FROM daily_klines WHERE code IN ({','.join('?'*len(v2c))}) AND date>='2008-01-01' ORDER BY code,date",conn,params=v2c); conn.close()
lp={}; t0=time.time()
for code,grp in d2.groupby('code'):
    grp=grp.sort_values('date').reset_index(drop=True)
    if len(grp)<LOOKBACK+FWD_DAYS+100: continue
    n=len(grp); dates=grp['date'].tolist(); c=grp['close'].values.astype(float)
    o=grp['open'].values.astype(float); h=grp['high'].values.astype(float)
    l=grp['low'].values.astype(float); v=grp['volume'].values.astype(float)
    feats=np.zeros((n,NF),dtype=np.float32)
    ret=np.zeros(n); ret[1:]=np.clip(np.log(np.maximum(c[1:],0.01)/np.maximum(c[:-1],0.01)),-0.2,0.2)
    feats[:,0]=ret; feats[:,1]=np.clip(pd.Series(c).pct_change(5).fillna(0).values,-0.5,0.5)
    feats[:,2]=np.clip(pd.Series(c).pct_change(20).fillna(0).values,-1,1)
    feats[:,3]=pd.Series(ret).rolling(20,min_periods=5).std().fillna(0).values
    feats[:,4]=np.clip(pd.Series(v).rolling(5).mean().values/np.maximum(pd.Series(v).rolling(20).mean().values,1),0,5)
    h20=pd.Series(h).rolling(20,min_periods=5).max().values; l20=pd.Series(l).rolling(20,min_periods=5).min().values
    feats[:,5]=np.clip((c-l20)/np.maximum(h20-l20,0.01),0,1); feats[:,6]=np.clip((h-l)/np.maximum(c,0.01),0,0.2)
    gap=np.zeros(n); gap[1:]=np.clip((o[1:]-c[:-1])/np.maximum(c[:-1],0.01),-0.1,0.1); feats[:,7]=gap
    e12=pd.Series(c).ewm(span=12).mean().values; e26=pd.Series(c).ewm(span=26).mean().values
    dif=np.nan_to_num((e12-e26)/np.maximum(c,0.01),0)
    feats[:,8]=np.clip(dif,-0.5,0.5); feats[:,9]=np.clip(pd.Series(dif).ewm(span=9).mean().values,-0.5,0.5)
    ma20=pd.Series(c).rolling(20).mean().values; std20=pd.Series(c).rolling(20).std().values
    feats[:,10]=np.clip(np.nan_to_num((c-ma20)/np.maximum(std20,0.01),0),-3,3)
    feats[:,11]=np.clip(std20/np.maximum(c,0.01),0,0.2)
    tr=np.maximum(h-l,np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
    feats[:,12]=np.clip(pd.Series(tr).rolling(14).mean().fillna(0).values/np.maximum(c,0.01),0,0.1)
    obv=np.zeros(n); obv[0]=v[0]
    for i in range(1,n): obv[i]=obv[i-1]+v[i]*(1 if c[i]>c[i-1] else (-1 if c[i]<c[i-1] else 0))
    feats[:,13]=np.clip(pd.Series(obv).pct_change(20).fillna(0).values,-1,1)
    ga=pd.Series(np.where(np.diff(c,prepend=c[0])>0,np.diff(c,prepend=c[0]),0)).ewm(alpha=1/14).mean().values
    lo=pd.Series(np.where(np.diff(c,prepend=c[0])<0,-np.diff(c,prepend=c[0]),0)).ewm(alpha=1/14).mean().values
    rsi=np.nan_to_num(100-100/(1+ga/np.maximum(lo,1e-8)),50); feats[:,14]=np.clip((rsi-50)/50,-1,1)
    ma60=pd.Series(c).rolling(60).mean().values
    feats[:,15]=np.clip((c-ma20)/np.maximum(ma20,0.01),-0.5,0.5); feats[:,16]=np.clip((c-ma60)/np.maximum(ma60,0.01),-0.5,0.5)
    feats[:,17]=np.clip(v/np.maximum(pd.Series(v).rolling(20).mean().values,1),0,5)
    feats[:,18]=np.clip((c-o)/np.maximum(c,0.01),-0.1,0.1)
    feats[:,19]=pd.Series((ret>0).astype(float)).rolling(20).mean().values; feats=np.nan_to_num(feats,0.0)
    for tp in [t for t in pool['testPoints'] if t['stockCode']==code and t['alpha'] is not None]:
        ci=-1
        for j,d in enumerate(dates):
            if str(d).startswith(tp['cutoffDate']): ci=j; break
        if ci<LOOKBACK-1: continue
        seq=feats[ci-LOOKBACK+1:ci+1]
        with torch.no_grad():
            Xt=torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)
            probs=torch.softmax(model(Xt),dim=1).cpu().numpy()[0]
            lp[f"{code}|{tp['cutoffDate']}"]=float(probs[1]-probs[2])
print(f"LSTM-v3: {len(lp)} preds, {(time.time()-t0):.0f}s")
out=json.load(open(PROJECT/'data'/'p3-kronos-lstm-signals.json')); out['lstm_v3']=lp
json.dump(out,open(PROJECT/'data'/'p3-kronos-lstm-signals.json','w'))
print("Saved")
