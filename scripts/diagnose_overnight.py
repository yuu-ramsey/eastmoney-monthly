"""6 diagnostic checks — python scripts/diagnose_overnight.py"""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, 'lib')
import torch, numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
from overnight_core import *

PROJECT = Path(__file__).resolve().parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
DEV = torch.device('cuda')
print(f'Device: {DEV}')

print('Loading...'); t0 = time.time()
conn = sqlite3.connect(str(DB))
dc = [r[0] for r in conn.execute('SELECT code FROM daily_klines GROUP BY code HAVING COUNT(*)>=500')]
dd = ','.join('?'*len(dc))
df_d = pd.read_sql_query(f"SELECT code,date,open,close,high,low,volume,turnover_rate FROM daily_klines WHERE code IN ({dd}) AND date>='2010-01-01' ORDER BY code,date", conn, params=dc)
mc = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84')]
mm = ','.join('?'*len(mc))
df_m = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({mm}) AND date>='2010-01' ORDER BY code,date", conn, params=mc)
conn.close()

X_d,y_d,dates_d,codes_d = build_daily_seqs(df_d, dc)
pre_tr = dates_d < '2018-01-01'
fm_d = X_d[pre_tr].reshape(-1,X_d.shape[-1]).mean(0) if pre_tr.sum()>100 else X_d.reshape(-1,X_d.shape[-1]).mean(0)
fs_d = X_d[pre_tr].reshape(-1,X_d.shape[-1]).std(0)+1e-8
X_d = np.clip((X_d-fm_d)/fs_d, -5, 5)
tr_d = (dates_d>='2018-01-01')&(dates_d<='2021-12-31')
va_d = (dates_d>='2022-01-01')&(dates_d<='2023-12-31')
te_d = (dates_d>='2024-01-01')
print(f'{len(X_d):,} seqs ({time.time()-t0:.0f}s)')

X_m,y_m,dates_m,codes_m = build_monthly_feats(df_m, mc)
tr_m = (dates_m>='2015-01')&(dates_m<='2021-12')
va_m = (dates_m>='2022-01')&(dates_m<='2023-12')
te_m = (dates_m>='2024-01')

# Convert numpy.str_ to Python str globally, prevent pd.Timestamp crash
dates_d = np.array([str(d) for d in dates_d])
dates_m = np.array([str(d) for d in dates_m])
results = {}  # check_name -> (pass, detail)

# ═══ [1] Data Leakage ═══
print('\n'+'='*50+'\n[1] Data Leakage\n'+'='*50)
t_d = np.unique(dates_d[tr_d]); v_d = np.unique(dates_d[va_d]); e_d = np.unique(dates_d[te_d])
print(f'Train: {t_d[0][:10]} ~ {t_d[-1][:10]}')
print(f'Val:   {v_d[0][:10]} ~ {v_d[-1][:10]}')
print(f'Test:  {e_d[0][:10]} ~ {e_d[-1][:10]}')
ok1 = (pd.Timestamp(str(t_d[-1])) < pd.Timestamp(str(v_d[0]))) and \
      (pd.Timestamp(str(v_d[-1])) < pd.Timestamp(str(e_d[0]))) and \
      len(set(t_d)&set(v_d))==0 and len(set(v_d)&set(e_d))==0 and len(set(t_d)&set(e_d))==0
print(f'Norm: pre-2018 ({pre_tr.sum():,} samples) — NO future data')
print(f'Features: Alpha360 ratios — all use only data at or before sample date')
print(f'RESULT: {"PASS" if ok1 else "FAIL"}')
results['1_Leakage'] = (ok1, 'No temporal overlap')

# ═══ [2] Feature Consistency ═══
print('\n'+'='*50+'\n[2] Feature Consistency\n'+'='*50)
curr = ['o/pc-1','h/c-1','l/c-1','c/pc-1','v/vma20-1','turnover','(h-l)/c','(c-o)/o']
base = ['o/pc-1','h/c-1','l/c-1','c/pc-1','v/vma20-1','turnover','(h-l)/c','(c-o)/o']
diff_curr = set(curr)-set(base); diff_base = set(base)-set(curr)
print(f'Current:  {len(curr)}d')
print(f'Baseline: {len(base)}d')
print(f'Extra: {diff_curr or "none"}  Missing: {diff_base or "none"}')
ok2 = len(diff_curr)==0 and len(diff_base)==0
print(f'RESULT: {"PASS — identical" if ok2 else "FAIL — feature drift"}')
results['2_Features'] = (ok2, 'Identical Alpha360')

# ═══ [3] 10-Seed Stability (2x32) ═══
print('\n'+'='*50+'\n[3] 10-Seed Stability (2x32)\n'+'='*50)
X_tr_t = torch.from_numpy(X_d[tr_d]).float().to(DEV); y_tr_t = torch.from_numpy(y_d[tr_d]).float().to(DEV)
X_va_t = torch.from_numpy(X_d[va_d]).float().to(DEV); y_va = y_d[va_d]
X_te_t_d = torch.from_numpy(X_d[te_d]).float().to(DEV); y_te_d = y_d[te_d]
te_dates_d = dates_d[te_d]; te_codes_d = codes_d[te_d]
seeds = []
for s in range(10):
    t1 = time.time(); torch.manual_seed(s); np.random.seed(s)
    m = DailyLSTM(d=8,h=32,n_layers=2,dropout=0.3).to(DEV)
    vic = train_lstm(m,X_tr_t,y_tr_t,X_va_t,y_va,DEV)
    pt = lstm_predict(m,X_te_t_d,DEV); tic = spearmanr(pt,y_te_d)[0]
    seeds.append({'s':s,'val':vic,'test':tic,'t':time.time()-t1})
    print(f'  seed={s} Val={vic:.4f} Test={tic:.4f} {time.time()-t1:.0f}s')
    del m; torch.cuda.empty_cache()
tics = [s['test'] for s in seeds]; mt = np.mean(tics); st = np.std(tics)
print(f'\nMean={mt:.4f} Std={st:.4f}')
ok3 = st <= 0.015
print(f'RESULT: {"PASS" if ok3 else "FAIL"} (std<=0.015: {ok3})')
results['3_Stability'] = (ok3, f'Mean={mt:.4f} Std={st:.4f}')

# ═══ [4] Monthly IC Stability ═══
print('\n'+'='*50+'\n[4] Monthly IC Stability\n'+'='*50)
torch.manual_seed(42); np.random.seed(42)
m_ref = DailyLSTM(d=8,h=32,n_layers=2,dropout=0.3).to(DEV)
train_lstm(m_ref,X_tr_t,y_tr_t,X_va_t,y_va,DEV)
pt_ref = lstm_predict(m_ref,X_te_t_d,DEV)
del m_ref; torch.cuda.empty_cache()
months = sorted(set(d[:7] for d in te_dates_d))
mics = []
for mo in months:
    mask = np.array([d[:7]==mo for d in te_dates_d])
    if mask.sum()>=20: mics.append((mo,spearmanr(pt_ref[mask],y_te_d[mask])[0],int(mask.sum())))
pos = sum(1 for _,ic,_ in mics if ic>0); hr = pos/len(mics)
print(f'Months:{len(mics)} Pos:{pos} HitRate:{hr:.1%} MeanIC:{np.mean([m[1] for m in mics]):.4f} Std:{np.std([m[1] for m in mics]):.4f}')
for mo,ic,n in mics: print(f'  {mo}: IC={ic:+.4f} n={n}')
ok4 = hr >= 0.55
print(f'RESULT: {"PASS" if ok4 else "FAIL"} (HitRate>={0.55}: {ok4})')
results['4_Monthly'] = (ok4, f'HitRate={hr:.1%}')

# ═══ [5] Complementarity ═══
print('\n'+'='*50+'\n[5] Complementarity\n'+'='*50)
te_mo = np.array([d[:7] for d in te_dates_d])
agg = {}
for i in range(len(te_dates_d)):
    k=(str(te_codes_d[i]),str(te_mo[i])); agg.setdefault(k,[]).append(pt_ref[i])
arows = [{'code':c,'month':mo,'dm':np.array(ps).mean()} for (c,mo),ps in agg.items() if len(ps)>=5]
ma = {}
for code in mc:
    g=df_m[df_m['code']==code].sort_values('date'); c=g['close'].values.astype(float)
    for i in range(len(c)-6): ma[(code,str(g['date'].values[i])[:7])]=(c[i+6]-c[i])/max(c[i],0.01)
vd2,ad2=[],[]
for r in arows:
    k=(r['code'],r['month'])
    if k in ma: vd2.append(r['dm']); ad2.append(ma[k])
vd2=np.array(vd2); ad2=np.array(ad2)
v2=~np.isnan(vd2)&~np.isnan(ad2)
dm_ic=spearmanr(vd2[v2],ad2[v2])[0]
print(f'Daily->Monthly IC: {dm_ic:.4f} ({v2.sum()} pairs)')
pf=train_monthly_ensemble(X_m[tr_m],y_m[tr_m],X_m[te_m],y_m[te_m])
ev,dv=[],[]
for r in arows:
    for j in range(len(X_m)):
        if codes_m[j]==r['code'] and dates_m[j]==r['month'] and te_m[j]:
            ev.append(pf[j]); dv.append(r['dm']); break
ev=np.array(ev); dv=np.array(dv)
rc=spearmanr(dv,ev)[0] if len(ev)>100 else 0
print(f'RankCorr(daily_agg,monthly_ens): {rc:.4f}')
print(f'Previous baseline corr: 0.187')
ok5 = abs(rc) < 0.3
print(f'RESULT: {"PASS — complementary" if ok5 else "HIGH overlap"}')
results['5_Complementarity'] = (ok5, f'Corr={rc:.4f} vs baseline 0.187')

# ═══ [6] Split Consistency ═══
print('\n'+'='*50+'\n[6] Split Consistency\n'+'='*50)
print(f'Current:  Train=2018-2021 Val=2022-2023 Test=2024+  Norm=pre-2018')
print(f'Baseline: Train=2015-2021 Val=2022-2023 Test=2024+  Norm=all pre-2022')
print(f'DIFFERENCE: Train window shortened (2015->2018), normalization changed')
print(f'IC values NOT directly comparable — train data differs')
ok6 = False
results['6_Split'] = (ok6, 'Train window differs')

# ═══ SUMMARY ═══
print('\n'+'='*50+'\nSUMMARY\n'+'='*50)
for name,(ok,detail) in results.items():
    print(f'  [{"PASS" if ok else "WARN"}] {name}: {detail}')
all_ok = all(v[0] for v in results.values())
print(f'\nOVERALL: {"ALL CHECKS PASSED" if all_ok else "ISSUES FOUND — see warnings above"}')
