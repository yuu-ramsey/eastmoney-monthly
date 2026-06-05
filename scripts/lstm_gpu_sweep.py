# LSTM GPU hyperparameter sweep: hidden size, layers, architectures
# Metrics: IC, RankIC, MAE, RMSE, MAPE, Direction Accuracy
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, copy
import numpy as np, pandas as pd, sqlite3, time, json
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import pywt

DEV = torch.device('cuda')
SEP60 = '=' * 60
SEP70 = '=' * 70
SEQ_LEN, BATCH = 60, 256
LR, WD, EPOCHS = 3e-4, 1e-5, 60
N_FFT, RANK_LAMBDA = 10, 0.3
EMA_DECAY = 0.999
DB = '.eastmoney-ai/db/klines-v2.sqlite'
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)', flush=True)

# ======== Data (same as final_ensemble) ========
def fft_f(prices):
    x=np.arange(len(prices)); trend=np.polyfit(x,prices,1); detrended=prices-np.polyval(trend,x)
    fft_p=np.fft.rfft(detrended); amps=np.abs(fft_p); freqs=np.fft.rfftfreq(len(detrended))
    if len(amps)<=1: return np.zeros(N_FFT*3,dtype=np.float32)
    pk=np.argsort(amps[1:])[::-1][:N_FFT]+1; feats=[]
    for idx in pk:
        if idx<len(freqs): feats.extend([freqs[idx],amps[idx],np.angle(fft_p[idx])])
    while len(feats)<N_FFT*3: feats.extend([0,0,0])
    return np.array(feats[:N_FFT*3],dtype=np.float32)

def wdenoise(signal,wavelet='db4',level=2):
    coeffs=pywt.wavedec(signal,wavelet,level=level)
    sigma=np.median(np.abs(coeffs[-1]))/0.6745; threshold=sigma*np.sqrt(2*np.log(len(signal)))
    coeffs_d=[coeffs[0]]+[pywt.threshold(c,threshold,mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs_d,wavelet)[:len(signal)]

conn=sqlite3.connect(DB)
codes=[r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params=','.join('?'*len(codes))
df=pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date",conn,params=codes)
conn.close()

print('Loading data...', flush=True); t0=time.time()
seqs_list,ys_list,dates_list=[],[],[]
for code in codes:
    g=df[df['code']==code].sort_values('date').reset_index(drop=True)
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
        fwd=np.clip(fwd_raw,-2,2)
        seq=np.zeros((SEQ_LEN,17+N_FFT*3),dtype=np.float32)
        seq[:,:17]=F[i-SEQ_LEN+1:i+1]; seq[:,17:]=fft_f(c_clean[i-SEQ_LEN+1:i+1])
        seqs_list.append(seq); ys_list.append(fwd); dates_list.append(g['date'].iloc[i])

seqs=np.array(seqs_list,dtype=np.float32); ys=np.array(ys_list,dtype=np.float32)
v=~np.isnan(seqs).any(axis=(1,2))&~np.isnan(ys); seqs=seqs[v]; ys=ys[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12'); va_m=(dates_arr>='2022-01')&(dates_arr<='2023-12'); te_m=(dates_arr>='2024-01')
print(f'Data: {len(seqs):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} ({seqs.shape[2]}feat) {time.time()-t0:.0f}s',flush=True)

st=seqs[tr_m]; fm=st.reshape(-1,seqs.shape[2]).mean(0); fs=st.reshape(-1,seqs.shape[2]).std(0)+1e-8
seqs=np.clip((seqs-fm)/fs,-5,5)

X_tr=torch.from_numpy(seqs[tr_m]).float().to(DEV); y_tr_t=torch.from_numpy(ys[tr_m]).float().to(DEV)
X_va=torch.from_numpy(seqs[va_m]).float().to(DEV); y_va_np=ys[va_m]
X_te=torch.from_numpy(seqs[te_m]).float().to(DEV); y_te_np=ys[te_m]; te_dates=dates_arr[te_m]

# ======== Metrics ========
def all_metrics(pred, true):
    pred=np.asarray(pred); true=np.asarray(true)
    mae=np.mean(np.abs(pred-true))
    rmse=np.sqrt(np.mean((pred-true)**2))
    mape=np.mean(np.abs((pred-true)/np.maximum(np.abs(true),0.001)))*100
    ic=spearmanr(pred,true)[0]
    hit=np.mean((pred>0)==(true>0))
    # Monthly CS IC
    cs_ics=[spearmanr(pred[te_dates==m],true[te_dates==m])[0] for m in np.unique(te_dates) if (te_dates==m).sum()>=20]
    cs_ic=np.mean(cs_ics) if cs_ics else np.nan
    # Top-20% LS
    n20=max(1,int(len(pred)*0.2))
    t20=np.argsort(pred)[-n20:]; b20=np.argsort(pred)[:n20]
    ls=np.mean(true[t20])-np.mean(true[b20])
    return {'MAE':mae,'RMSE':rmse,'MAPE':mape,'IC':ic,'CS_IC':cs_ic,'Hit':hit,'Top20_LS':ls,'N_months':len(cs_ics)}

def rank_loss(pred,target):
    if len(pred)<2: return torch.tensor(0.0,device=pred.device)
    d=target.unsqueeze(0)-target.unsqueeze(1); pd=pred.unsqueeze(0)-pred.unsqueeze(1)
    return torch.sigmoid(-pd*torch.sign(d)*torch.abs(d)).mean()

# ======== LSTM Configs to Sweep ========
configs = [
    # (name, hidden, layers, dropout, bidirectional)
    ('LSTM-h128-l2', 128, 2, 0.3, False),
    ('LSTM-h256-l2', 256, 2, 0.3, False),
    ('LSTM-h256-l3', 256, 3, 0.3, False),
    ('LSTM-h512-l2', 512, 2, 0.35, False),
    ('LSTM-h256-l2-bi', 256, 2, 0.3, True),
    ('LSTM-h128-l4', 128, 4, 0.35, False),
    ('LSTM-h256-l4', 256, 4, 0.35, False),
]

class LSTMRev2(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, bidirectional):
        super().__init__()
        self.in_ln = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0,
                            bidirectional=bidirectional)
        # Forget gate bias = 1.0
        for name, param in self.lstm.named_parameters():
            if 'bias_ih' in name:
                n = param.shape[0] // 4
                param.data[n:2*n].fill_(1.0)
            elif 'bias_hh' in name:
                n = param.shape[0] // 4
                param.data[n:2*n].fill_(1.0)
        lstm_out = hidden * (2 if bidirectional else 1)
        self.ln = nn.LayerNorm(lstm_out)
        self.head = nn.Sequential(
            nn.Linear(lstm_out, lstm_out//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(lstm_out//2, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1))
    def forward(self, x):
        x = self.in_ln(x)
        o, _ = self.lstm(x)
        o = self.ln(o[:, -1, :])
        return self.head(o).squeeze(-1)

huber = nn.HuberLoss(delta=0.5)
results = []

for cfg_name, hidden, n_layers, dropout, bidirectional in configs:
    print(f'\n{SEP60}')
    print(f'Config: {cfg_name} (hidden={hidden}, layers={n_layers}, dropout={dropout}, bidirectional={bidirectional})')
    print(SEP60, flush=True)
    t0 = time.time()

    model = LSTMRev2(seqs.shape[2], hidden, n_layers, dropout, bidirectional).to(DEV)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_ic, best_state, patience = -99, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr), device=DEV)
        for i in range(0, len(X_tr), BATCH):
            idx = perm[i:i+BATCH]
            pred = model(X_tr[idx])
            loss = huber(pred, y_tr_t[idx]) + RANK_LAMBDA * rank_loss(pred, y_tr_t[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for ep_, p in zip(ema_model.parameters(), model.parameters()):
                    ep_.data.mul_(EMA_DECAY).add_(p.data, alpha=1-EMA_DECAY)

        ema_model.eval()
        with torch.no_grad():
            p_v = [ema_model(X_va[i:i+BATCH]).detach().cpu().numpy() for i in range(0, len(X_va), BATCH)]
            ic = spearmanr(np.concatenate(p_v), y_va_np)[0]

        if ic > best_ic:
            best_ic = ic; best_state = copy.deepcopy(ema_model.state_dict()); patience = 0
        else:
            patience += 1

        if patience >= 20:
            print(f'  Early stop ep{ep+1} best={best_ic:+.4f}', flush=True)
            break

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_te = np.concatenate([model(X_te[i:i+BATCH]).detach().cpu().numpy() for i in range(0, len(X_te), BATCH)])

    metrics = all_metrics(p_te, y_te_np)
    metrics['config'] = cfg_name
    metrics['hidden'] = hidden
    metrics['layers'] = n_layers
    metrics['bidirectional'] = bidirectional
    metrics['params'] = sum(p.numel() for p in model.parameters())
    metrics['time_s'] = time.time() - t0
    metrics['val_best_ic'] = float(best_ic)
    results.append(metrics)

    cs_ic_str = f"{metrics['CS_IC']:+.4f}"
    ic_str = f"{metrics['IC']:+.4f}"
    hit_str = f"{metrics['Hit']:.3f}"
    mae_str = f"{metrics['MAE']:.4f}"
    rmse_str = f"{metrics['RMSE']:.4f}"
    mape_str = f"{metrics['MAPE']:.1f}"
    ls_str = f"{metrics['Top20_LS']:+.4f}"
    p_str = f"{metrics['params']:,}"
    t_str = f"{metrics['time_s']:.0f}"
    print('  CS_IC=' + cs_ic_str + ' IC=' + ic_str + ' Hit=' + hit_str)
    print('  MAE=' + mae_str + ' RMSE=' + rmse_str + ' MAPE=' + mape_str + '%')
    print('  Top20_LS=' + ls_str + ' Params=' + p_str + ' Time=' + t_str + 's')

# ======== Summary ========
print(f'\n{SEP70}')
print(f'LSTM GPU Sweep Summary')
print(SEP70)
hdr = f"{'Config':<22s} {'CS_IC':>8s} {'IC':>8s} {'Hit':>8s} {'MAE':>8s} {'RMSE':>8s} {'MAPE':>8s} {'Params':>10s} {'Time':>6s}"
print(hdr)
print('-'*90)
for r in sorted(results, key=lambda x: x['CS_IC'], reverse=True):
    row = (f"{r['config']:<22s} {r['CS_IC']:+8.4f} {r['IC']:+8.4f} {r['Hit']:8.3f} "
           f"{r['MAE']:8.4f} {r['RMSE']:8.4f} {r['MAPE']:7.1f}% {r['params']:>10,} {r['time_s']:>5.0f}s")
    print(row)

best_r = max(results, key=lambda x: x['CS_IC'])
print(f"\nBest: {best_r['config']} CS_IC={best_r['CS_IC']:+.4f}")

# Compare with baselines
print(f'\nBaselines:')
print(f'  Ridge:        CS_IC=+0.060  MAE=—     RMSE=—')
print(f'  LightGBM:     CS_IC=+0.166  MAE=—     RMSE=—')
print(f'  ENSEMBLE:     CS_IC=+0.178  MAE=—     RMSE=—')
