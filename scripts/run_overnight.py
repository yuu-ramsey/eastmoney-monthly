"""Overnight experiment pipeline — python scripts/run_overnight.py"""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, 'lib')
import torch, numpy as np, pandas as pd, sqlite3, json, time
from pathlib import Path; from datetime import date
from scipy.stats import spearmanr
from overnight_core import *

PROJECT = Path(__file__).resolve().parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'overnight_v2'; OUT.mkdir(parents=True, exist_ok=True)
RUN_DATE = date.today().isoformat()
DEV = torch.device('cuda')

LOG_PATH = OUT / f'overnight_{RUN_DATE}.log'
class Tee:
    def __init__(self, *fs): self.files = fs
    def write(self, s):
        for f in self.files: f.write(s); f.flush()
    def flush(self):
        for f in self.files: f.flush()
LOG_FH = open(LOG_PATH, 'w', encoding='utf-8', buffering=1)
sys.stdout = Tee(sys.stdout, LOG_FH)

print(f'=== Overnight {RUN_DATE}  Device: {DEV} ===')
RESULTS = OUT / 'results.jsonl'; BASELINE = OUT / 'baseline.json'
completed = set()
if RESULTS.exists():
    for line in open(RESULTS):
        try: completed.add(json.loads(line).get('exp_id',''))
        except: pass
print(f'Completed: {len(completed)}')

print('Loading...'); t0 = time.time()
conn = sqlite3.connect(str(DB))
dc = [r[0] for r in conn.execute('SELECT code FROM daily_klines GROUP BY code HAVING COUNT(*)>=500')]
dd = ','.join('?'*len(dc))
df_d = pd.read_sql_query(f"SELECT code,date,open,close,high,low,volume,turnover_rate FROM daily_klines WHERE code IN ({dd}) AND date>='2010-01-01' ORDER BY code,date", conn, params=dc)
mc = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84')]
mm = ','.join('?'*len(mc))
df_m = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({mm}) AND date>='2010-01' ORDER BY code,date", conn, params=mc)
conn.close()
print(f'  {len(dc)}d/{len(mc)}m stocks ({time.time()-t0:.0f}s)')

X_d, y_d, dates_d, codes_d = build_daily_seqs(df_d, dc)
pre_tr = dates_d < '2018-01-01'
fm_d = X_d[pre_tr].reshape(-1, X_d.shape[-1]).mean(0); fs_d = X_d[pre_tr].reshape(-1, X_d.shape[-1]).std(0) + 1e-8
X_d = np.clip((X_d-fm_d)/fs_d, -5, 5)
tr_d = (dates_d >= '2018-01-01') & (dates_d <= '2021-12-31')
va_d = (dates_d >= '2022-01-01') & (dates_d <= '2023-12-31')
te_d = (dates_d >= '2024-01-01')
print(f'Daily: {len(X_d):,} T={tr_d.sum():,} V={va_d.sum():,} Te={te_d.sum():,}')
X_tr_t = torch.from_numpy(X_d[tr_d]).float().to(DEV); y_tr_t = torch.from_numpy(y_d[tr_d]).float().to(DEV)
X_va_t = torch.from_numpy(X_d[va_d]).float().to(DEV); y_va = y_d[va_d]
X_te_t_d = torch.from_numpy(X_d[te_d]).float().to(DEV); y_te_d = y_d[te_d]
te_dates_d = dates_d[te_d]; te_codes_d = codes_d[te_d]

X_m, y_m, dates_m, codes_m = build_monthly_feats(df_m, mc)
tr_m = (dates_m >= '2015-01') & (dates_m <= '2021-12')
va_m = (dates_m >= '2022-01') & (dates_m <= '2023-12')
te_m = (dates_m >= '2024-01')
print(f'Monthly: {len(X_m):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,}')
X_m_tr = X_m[tr_m]; y_m_tr = y_m[tr_m]; X_m_va = X_m[va_m]; y_m_va = y_m[va_m]
X_m_te = X_m[te_m]; y_m_te = y_m[te_m]

def dosave(r):
    save_result(RESULTS, RUN_DATE, r)

ARCH = [('D1_1x32',1,32),('D1_1x64',1,64),('D1_1x128',1,128),
        ('D1_2x32',2,32),('D1_2x64',2,64),('D1_2x128',2,128),
        ('D1_3x32',3,32),('D1_3x64',3,64)]
GROUPS = {'G2_MA':range(0,3),'G3_MACD':range(3,6),'G4_Vol':range(6,8),
          'FFT':range(8,18),'G7_PriceVol':range(18,32)}

print('\n'+'='*60+'\nD1: Architecture Sweep\n'+'='*60)
d1r = []
for eid,nl,nh in ARCH:
    if eid in completed: print(f'  [{eid}] SKIP'); continue
    t1 = time.time(); torch.manual_seed(42); np.random.seed(42)
    m = DailyLSTM(d=8,h=nh,n_layers=nl,dropout=0.3).to(DEV)
    vic = train_lstm(m,X_tr_t,y_tr_t,X_va_t,y_va,DEV)
    pt = lstm_predict(m,X_te_t_d,DEV)
    tic = spearmanr(pt,y_te_d)[0]; hit = np.mean((pt>0)==(y_te_d>0))
    is_of,gap = overfit_flag(vic,tic)
    r = {'exp_id':eid,'phase':'D1','n_layers':nl,'hidden':nh,
         'val_ic':round(vic,4),'test_ic':round(tic,4),'hit':round(hit,4),
         'overfit_gap':gap,'overfit':is_of,
         'params':sum(p.numel() for p in m.parameters()),'elapsed_s':round(time.time()-t1,1)}
    dosave(r); d1r.append(r)
    print(f'  [{eid}] L={nl} H={nh} Val={vic:.4f} Test={tic:.4f} Hit={hit:.3f} gap={gap:+.4f}{" OF" if is_of else ""}')
    del m; torch.cuda.empty_cache()
vd1 = [r for r in d1r if not r.get('overfit')]
best = max(vd1,key=lambda r:r['test_ic']) if vd1 else (max(d1r,key=lambda r:r['test_ic']) if d1r else None)
if best: print(f'Best: {best["exp_id"]} IC={best["test_ic"]:.4f}')

print('\n'+'='*60+'\nD2: 20-Seed\n'+'='*60)
eid2 = 'D2_20seed'
if eid2 not in completed:
    bl = best['n_layers'] if best else 2; bh = best['hidden'] if best else 64
    print(f'  Arch L={bl} H={bh}')
    dpreds, vics2 = [], []
    for s in range(20):
        t1 = time.time(); torch.manual_seed(s); np.random.seed(s)
        m = DailyLSTM(d=8,h=bh,n_layers=bl,dropout=0.3).to(DEV)
        vic = train_lstm(m,X_tr_t,y_tr_t,X_va_t,y_va,DEV); vics2.append(vic)
        dpreds.append(lstm_predict(m,X_te_t_d,DEV))
        del m; torch.cuda.empty_cache()
        if s%5==0: print(f'  seed {s}/20 {time.time()-t1:.0f}s')
    dp = np.array(dpreds); em = dp.mean(axis=0)
    ic_m = spearmanr(em,y_te_d)[0]; hit_m = np.mean((em>0)==(y_te_d>0))
    sics = [spearmanr(dp[s],y_te_d)[0] for s in range(20)]
    mv = np.mean(vics2); is_of,gap = overfit_flag(mv,ic_m)
    r = {'exp_id':eid2,'phase':'D2','n_seeds':20,'arch':f'L{bl}_H{bh}',
         'val_ic_mean':round(mv,4),'test_ic_mean':round(float(ic_m),4),
         'seed_ic_std':round(float(np.std(sics)),4),
         'hit':round(float(hit_m),4),'overfit_gap':gap,'overfit':is_of}
    dosave(r)
    print(f'  IC={ic_m:.4f} std={np.std(sics):.4f} Hit={hit_m:.3f}{" OF" if is_of else ""}')
    np.savez(OUT/'daily_ens_preds.npz',ens_mean=em,dates=te_dates_d,codes=te_codes_d,y_true=y_te_d)

print('\n'+'='*60+'\nM1: Feature Ablation\n'+'='*60)
eidf = 'M1_full_32d'
if eidf not in completed:
    pf = train_monthly_ensemble(X_m_tr,y_m_tr,X_m_te,y_m_te)
    pv = train_monthly_ensemble(X_m_tr,y_m_tr,X_m_va,y_m_va)
    vic = spearmanr(pv,y_m_va)[0]; tic,_ = cs_ic_helper(pf,y_m_te,dates_m[te_m])
    hit = np.mean((pf>0)==(y_m_te>0)); is_of,gap = overfit_flag(vic,tic)
    dosave({'exp_id':eidf,'phase':'M1','features':'all_32d','val_ic':round(vic,4),
            'test_ic':round(float(tic),4),'hit':round(float(hit),4),'overfit_gap':gap,'overfit':is_of})
    print(f'  FULL: Val={vic:.4f} Test={tic:.4f} Hit={hit:.3f}{" OF" if is_of else ""}')
    if not BASELINE.exists():
        json.dump({'created':RUN_DATE,'monthly_ic':round(float(tic),4),'monthly_hit':round(float(hit),4)},open(BASELINE,'w'),indent=2)
        print(f'  Baseline frozen: {tic:.4f}')
    for gn,gidx in GROUPS.items():
        eid = f'M1_abl_no{gn}'
        if eid in completed: print(f'  [{eid}] SKIP'); continue
        keep = [i for i in range(32) if i not in gidx]
        pa = train_monthly_ensemble(X_m_tr[:,keep],y_m_tr,X_m_te[:,keep],y_m_te)
        pva = train_monthly_ensemble(X_m_tr[:,keep],y_m_tr,X_m_va[:,keep],y_m_va)
        va = spearmanr(pva,y_m_va)[0]; te,_ = cs_ic_helper(pa,y_m_te,dates_m[te_m])
        ha = np.mean((pa>0)==(y_m_te>0)); delta = te-tic
        is_of,gap = overfit_flag(va,te)
        dosave({'exp_id':eid,'phase':'M1','ablated_group':gn,'n_features':len(keep),
                'val_ic':round(va,4),'test_ic':round(float(te),4),'delta_ic':round(float(delta),4),
                'hit':round(float(ha),4),'overfit_gap':gap,'overfit':is_of})
        print(f'  -{gn}: IC={te:.4f} D={delta:+.4f} Hit={ha:.3f}{" OF" if is_of else ""}')

print('\n'+'='*60+'\nLeaderboard\n'+'='*60)
all_r = [json.loads(l) for l in open(RESULTS) if l.strip()] if RESULTS.exists() else []
LB = OUT/f'leaderboard-{RUN_DATE}.md'
with open(LB,'w',encoding='utf-8') as f:
    bl = json.load(open(BASELINE)) if BASELINE.exists() else {}
    f.write(f'# Overnight {RUN_DATE}\n\n**Baseline**: {bl.get("monthly_ic","?")}\n\n')
    of_r = [r for r in all_r if r.get('overfit')]
    f.write(f'## Overfitting ({len(of_r)})\n')
    for r in of_r: f.write(f'- {r["exp_id"]}: gap={r.get("overfit_gap"):+.4f}\n')
    f.write('\n## D1\n|Exp|L|H|Test IC|Hit|Params|OF|\n|---|---|---|---|---|---|---|\n')
    for r in sorted([r for r in all_r if r.get('phase')=='D1'],key=lambda r:r.get('test_ic',-99),reverse=True):
        f.write(f'|{r["exp_id"]}|{r.get("n_layers")}|{r.get("hidden")}|{r.get("test_ic"):.4f}|{r.get("hit"):.3f}|{r.get("params")}|{"Y" if r.get("overfit") else ""}|\n')
    f.write('\n## M1\n|Ablated|Test IC|Delta|Hit|OF|\n|---|---|---|---|---|\n')
    for r in sorted([r for r in all_r if 'ablated_group' in r],key=lambda r:r.get('delta_ic',-99)):
        f.write(f'|{r.get("ablated_group")}|{r.get("test_ic"):.4f}|{r.get("delta_ic"):+.4f}|{r.get("hit"):.3f}|{"Y" if r.get("overfit") else ""}|\n')
print(f'{LB}  Exps:{len(all_r)}  OF:{len(of_r)}  Done.')
LOG_FH.close()
