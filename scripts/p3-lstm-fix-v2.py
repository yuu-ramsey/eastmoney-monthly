"""LSTM-fix-v2: Triple-Barrier classification + GRU baseline on daily klines.
σ = per-stock past-12m daily return std, point-in-time.
Two-level gating: balanced acc > always-majority → v2 pool spread CI."""
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
np.random.seed(42)
sample_codes = np.random.choice(codes, min(500, len(codes)), replace=False).tolist()
d_df = pd.read_sql_query(
    f"SELECT code,date,close,open,high,low,volume FROM daily_klines WHERE code IN ({','.join('?'*len(sample_codes))}) AND date>='2010-01-01' ORDER BY code,date",
    conn, params=sample_codes)
conn.close()
print(f"Daily: {len(d_df)} rows, {d_df['code'].nunique()} codes")

LOOKBACK, FWD_DAYS = 60, 126  # 60d lookback, ~6m forward
all_seqs, all_labels, all_dates = [], [], []
label_counts = [0, 0, 0]

for code, grp in d_df.groupby('code'):
    grp = grp.sort_values('date').reset_index(drop=True)
    if len(grp) < LOOKBACK + FWD_DAYS + 60: continue
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
    feats[:,4] = np.clip(pd.Series(v).rolling(5).mean().values/np.maximum(pd.Series(v).rolling(20).mean().values,1),0,5)
    h20 = pd.Series(h).rolling(20,min_periods=5).max().values
    l20 = pd.Series(l).rolling(20,min_periods=5).min().values
    feats[:,5] = np.clip((c-l20)/np.maximum(h20-l20,0.01),0,1)
    feats[:,6] = np.clip((h-l)/np.maximum(c,0.01),0,0.2)
    gap = np.zeros(n); gap[1:] = np.clip((o[1:]-c[:-1])/np.maximum(c[:-1],0.01),-0.1,0.1)
    feats[:,7]=gap
    e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
    feats[:,8] = np.clip(np.nan_to_num((e12-e26)/np.maximum(c,0.01),0),-0.5,0.5)
    ga = pd.Series(np.where(np.diff(c,prepend=c[0])>0,np.diff(c,prepend=c[0]),0)).ewm(alpha=1/14).mean().values
    lo = pd.Series(np.where(np.diff(c,prepend=c[0])<0,-np.diff(c,prepend=c[0]),0)).ewm(alpha=1/14).mean().values
    feats[:,9] = np.clip((np.nan_to_num(100-100/(1+ga/np.maximum(lo,1e-8)),50)-50)/50,-1,1)

    for i in range(LOOKBACK-1, n-FWD_DAYS):
        if c[i] <= 0.01: continue
        seq = feats[i-LOOKBACK+1:i+1]
        if np.any(np.isnan(seq)): continue
        # Triple-Barrier: σ = past 12m daily return std, point-in-time
        past_ret_window = max(0, i-252)
        if i-past_ret_window < 20: continue
        sigma = ret[past_ret_window:i].std()
        if sigma < 0.005: sigma = 0.02
        upper = c[i]*(1+1.5*sigma); lower = c[i]*(1-1.5*sigma)
        label = 0  # neutral
        for j in range(i+1, min(i+FWD_DAYS+1, n)):
            if h[j] >= upper: label=1; break
            if l[j] <= lower: label=2; break
        all_seqs.append(seq); all_labels.append(label); all_dates.append(dates[i])
        label_counts[label] += 1

X = np.array(all_seqs,dtype=np.float32); y = np.array(all_labels,dtype=np.int64); da = np.array(all_dates)
print(f"\nSeqs: {len(X)}")
for i, name in enumerate(['neutral','bull','bear']):
    print(f"  {name}: {label_counts[i]} ({100*label_counts[i]/len(y):.1f}%)")

tm = (da>='2010-01-01')&(da<='2021-12-31'); vm = (da>='2022-01-01')&(da<='2023-12-31')
X_tr, y_tr = X[tm], y[tm]; X_va, y_va = X[vm], y[vm]
print(f"Train: {len(X_tr)}, Val: {len(X_va)}")

maj_class = np.argmax(np.bincount(y_tr))
maj_preds = np.full_like(y_va, maj_class)
maj_ba = balanced_accuracy_score(y_va, maj_preds)
maj_f1 = f1_score(y_va, maj_preds, average='macro')
print(f"Always-majority: class={maj_class} ba={maj_ba:.4f} f1={maj_f1:.4f}")

import torch.nn as nn
class GRU(nn.Module):
    def __init__(self, inp=10, hid=32, n_class=3):
        super().__init__()
        self.gru = nn.GRU(inp, hid, 1, batch_first=True)
        self.head = nn.Linear(hid, n_class)
    def forward(self, x): o,_ = self.gru(x); return self.head(o[:,-1,:])

print("\n=== Smoke test (1 epoch) ===")
model = GRU(10,32,3).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
total = sum(label_counts)
cw = torch.tensor([total/max(label_counts[0],1), total/max(label_counts[1],1), total/max(label_counts[2],1)]).float().to(DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=cw)
ds_tr = torch.utils.data.TensorDataset(torch.from_numpy(X_tr),torch.from_numpy(y_tr))
ds_va = torch.utils.data.TensorDataset(torch.from_numpy(X_va),torch.from_numpy(y_va))
ld_tr = torch.utils.data.DataLoader(ds_tr,256,shuffle=True)
ld_va = torch.utils.data.DataLoader(ds_va,256)

model.train(); tl=0
for Xb,yb in ld_tr:
    Xb,yb = Xb.to(DEVICE),yb.to(DEVICE)
    opt.zero_grad()
    loss = loss_fn(model(Xb),yb)
    if not torch.isnan(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        tl += loss.item()*Xb.size(0)
tl /= len(ds_tr)

model.eval(); preds,targs=[],[]
with torch.no_grad():
    for Xb,yb in ld_va:
        p = model(Xb.to(DEVICE))
        preds.append(p.argmax(dim=1).cpu().numpy())
        targs.append(yb.cpu().numpy())
preds=np.concatenate(preds); targs=np.concatenate(targs)
ba = balanced_accuracy_score(targs,preds); f1 = f1_score(targs,preds,average='macro')
cm = confusion_matrix(targs,preds,labels=[0,1,2])
print(f"Smoke TL={tl:.4f} ba={ba:.4f} f1={f1:.4f} vs maj(ba={maj_ba:.4f},f1={maj_f1:.4f})")
print(f"Confusion:\n{cm}")

if ba > maj_ba:
    print("\n=== Level 1 PASSED — full training ===")
    model = GRU(10,32,3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=5,factor=0.5)
    best_ba,best_ep,ni=-1.0,0,0
    for ep in range(1,30):
        model.train(); tl=0
        for Xb,yb in ld_tr:
            Xb,yb=Xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
            loss=loss_fn(model(Xb),yb)
            if not torch.isnan(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                tl+=loss.item()*Xb.size(0)
        tl/=len(ds_tr)
        model.eval(); preds,targs=[],[]
        with torch.no_grad():
            for Xb,yb in ld_va:
                p=model(Xb.to(DEVICE)); preds.append(p.argmax(dim=1).cpu().numpy()); targs.append(yb.cpu().numpy())
        preds=np.concatenate(preds); targs=np.concatenate(targs)
        ba=balanced_accuracy_score(targs,preds); sched.step(ba)
        if ep<=3 or ep%10==0: print(f"Ep{ep:3d}: TL={tl:.4f} ba={ba:.4f}")
        if ba>best_ba: best_ba,best_ep,ni=ba,ep,0; torch.save(model.state_dict(),str(PROJECT/'.eastmoney-ai'/'lstm'/'gru_wf.pt'))
        else: ni+=1
        if ni>=8 and ep>5: break
    print(f"Best ba={best_ba:.4f} ep{best_ep} vs maj={maj_ba:.4f}")
    if best_ba <= maj_ba: print("\n=== Level 1 FAILED: ba <= always-majority ==="), exit(1)

    # v2 pool
    model.load_state_dict(torch.load(str(PROJECT/'.eastmoney-ai'/'lstm'/'gru_wf.pt'),map_location=DEVICE)); model.eval()
    with open(PROJECT/'data'/'frozen-eval-lowpos-v2-baostock.json') as f: pool=json.load(f)
    conn=sqlite3.connect(str(DB)); v2c=list(set(t['stockCode'] for t in pool['testPoints']))
    d2=pd.read_sql_query(f"SELECT code,date,close,open,high,low,volume FROM daily_klines WHERE code IN ({','.join('?'*len(v2c))}) AND date>='2008-01-01' ORDER BY code,date",conn,params=v2c); conn.close()
    lp={}; t0=time.time()
    for code,grp in d2.groupby('code'):
        grp=grp.sort_values('date').reset_index(drop=True)
        if len(grp)<LOOKBACK+FWD_DAYS+60: continue
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
        feats[:,5]=np.clip((c-l20)/np.maximum(h20-l20,0.01),0,1); feats[:,6]=np.clip((h-l)/np.maximum(c,0.01),0,0.2)
        gap=np.zeros(n); gap[1:]=np.clip((o[1:]-c[:-1])/np.maximum(c[:-1],0.01),-0.1,0.1); feats[:,7]=gap
        e12=pd.Series(c).ewm(span=12).mean().values; e26=pd.Series(c).ewm(span=26).mean().values
        feats[:,8]=np.clip(np.nan_to_num((e12-e26)/np.maximum(c,0.01),0),-0.5,0.5)
        ga=pd.Series(np.where(np.diff(c,prepend=c[0])>0,np.diff(c,prepend=c[0]),0)).ewm(alpha=1/14).mean().values
        lo2=pd.Series(np.where(np.diff(c,prepend=c[0])<0,-np.diff(c,prepend=c[0]),0)).ewm(alpha=1/14).mean().values
        feats[:,9]=np.clip((np.nan_to_num(100-100/(1+ga/np.maximum(lo2,1e-8)),50)-50)/50,-1,1); feats=np.nan_to_num(feats,0.0)
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
    print(f"GRU-WF: {len(lp)} preds, {(time.time()-t0):.0f}s")
    out=json.load(open(PROJECT/'data'/'p3-kronos-lstm-signals.json')); out['gru_wf']=lp
    json.dump(out,open(PROJECT/'data'/'p3-kronos-lstm-signals.json','w'))
    print(f"Saved: {len(lp)}")
else:
    print("\n=== Level 1 FAILED: smoke test ba <= always-majority. Task not learnable. ===")
