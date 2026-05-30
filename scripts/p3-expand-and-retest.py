"""C1/C2: Expand pool 12→24tp, recompute GRU/reversal baselines."""
import json, numpy as np
from pathlib import Path
PROJECT = Path(__file__).resolve().parent.parent

cache = json.load(open(PROJECT/'data'/'baostock-klines-cache.json'))
print(f"Cache: {len(cache)} codes")

TP24 = ['2018-03','2018-06','2018-09','2018-12','2019-03','2019-06','2019-09','2020-03','2020-06','2020-09','2021-03','2021-06','2021-09','2021-12','2022-03','2022-06','2022-09','2022-10','2023-03','2023-06','2023-09','2024-02','2024-06','2024-10']
TRAIN24 = {'2018-03','2018-06','2018-09','2018-12','2019-03','2019-06','2019-09','2020-09','2021-03','2021-06','2022-03','2022-06'}

# Build pool
tps = []; tp_id = 0
for t in TP24:
    candidates = []
    for code, klines in cache.items():
        if not isinstance(klines, list) or len(klines) < 60: continue
        ci = -1
        for j, row in enumerate(klines):
            if isinstance(row, list) and row[0] == t: ci = j; break
        if ci < 60: continue
        c = klines[ci][1];
        if not isinstance(c, (int,float)) or c <= 0.01: continue
        past = [klines[j][1] for j in range(max(0,ci-12), ci) if isinstance(klines[j], list) and klines[j][1] > 0.001]
        if len(past) < 12: continue
        mn, mx = min(past), max(past)
        rp = (c - mn) / (mx - mn) if mx > mn else 0.5
        ma_cl = [klines[j][1] for j in range(max(0,ci-60), ci+1) if isinstance(klines[j], list) and klines[j][1] > 0.001]
        if len(ma_cl) < 60: continue
        ma60 = sum(ma_cl) / len(ma_cl)
        if rp <= 0.20 and c < ma60:
            fwd = [klines[j][1] for j in range(ci+1, min(len(klines), ci+7)) if isinstance(klines[j], list) and klines[j][1] > 0.001]
            alpha = (fwd[5] - c) / c * 100 if len(fwd) >= 6 else None
            num_code = code.split('.')[-1] if '.' in code else code
            candidates.append((num_code, c, rp, ma60, alpha))
    np.random.seed(42)
    if len(candidates) > 140:
        candidates = [candidates[i] for i in np.random.choice(len(candidates), 140, replace=False)]
    for code, close, rp, ma60, alpha in candidates:
        tps.append({'stockCode': code, 'cutoffDate': t, 'alpha': alpha, 'isTrain': t in TRAIN24})
        tp_id += 1
    print(f"  {t}: {len(candidates)}")

n_all = len(tps)
n_train = sum(1 for t in tps if t['isTrain'])
n_test = n_all - n_train
print(f"Pool: {n_all} pairs, Train: {n_train}, Test: {n_test}")

# Save pool
pool = {'version': 'lowpos-v2-24tp', 'testPoints': tps, 'config': {'nTimepoints': 24}}
with open(PROJECT/'data'/'frozen-eval-lowpos-v2-24tp.json', 'w') as f:
    json.dump(pool, f)
print("Pool saved.")

# ====== Reversal baseline on expanded pool ======
print("\n=== Reversal ===")
ap = []
for tp in tps:
    if tp['alpha'] is None: continue
    code = tp['stockCode']
    full_key = None
    for k in cache:
        if k.endswith('.' + code): full_key = k; break
    kl = cache.get(code) or (cache.get(full_key) if full_key else None)
    if not kl or len(kl) < 60: continue
    ci = -1
    for j, row in enumerate(kl):
        if isinstance(row, list) and row[0] == tp['cutoffDate']: ci = j; break
    if ci < 14: continue
    c = kl[ci][1] if isinstance(kl[ci], list) else 0
    if c <= 0.01: continue
    rev1m = (c - kl[ci-1][1]) / kl[ci-1][1] if ci >= 1 and kl[ci-1][1] > 0.01 else 0
    rev3m = (c - kl[ci-3][1]) / kl[ci-3][1] if ci >= 3 and kl[ci-3][1] > 0.01 else 0
    ms = 0; mn2 = 0
    for i in range(ci, max(-1, ci-60), -1):
        if isinstance(kl[i], list) and kl[i][1] > 0.01: ms += kl[i][1]; mn2 += 1
    d60 = (c - ms/mn2) / (ms/mn2) if mn2 >= 60 else 0
    ap.append({'alpha': tp['alpha'], 'rev1m': rev1m, 'rev3m': rev3m, 'd60': d60, 'isTrain': tp['isTrain']})

for key in ['rev1m', 'rev3m', 'd60']:
    vals = [p[key] for p in ap if not np.isnan(p[key])]
    m = sum(vals) / len(vals)
    s = np.sqrt(sum((v-m)**2 for v in vals) / len(vals))
    for p in ap: p[key+'_z'] = (p[key] - m) / max(s, 1e-8)
for p in ap: p['revZ'] = (p['rev1m_z'] + p['rev3m_z'] + p['d60_z']) / 3

def wins(arr):
    s = sorted(arr)
    lo, hi = s[max(0, len(s)//100-1)], s[min(len(s)-1, len(s)*99//100)]
    w = [lo if x < lo else (hi if x > hi else x) for x in arr]
    return sum(w) / len(w) if w else 0

def block_ci(pairs, z_key, ascending=True):
    blocks = {}
    for p in pairs:
        bk = f"{p.get('code','x')}|{p.get('cutoff','y')}"
        if bk not in blocks: blocks[bk] = []
        blocks[bk].append(p)
    blist = list(blocks.values())
    if not blist: return 0, 0, 0, 0
    vals = []
    for _ in range(5000):
        sample = []
        for _ in range(len(blist)):
            sample.extend(blist[np.random.randint(0, len(blist))])
        s = sorted(sample, key=lambda x: x[z_key], reverse=not ascending)
        n20 = len(s) // 5
        bulls = [p['alpha'] for p in s[:n20]]
        bears = [p['alpha'] for p in s[-n20:]]
        if bulls and bears: vals.append(wins(bulls) - wins(bears))
    if not vals: return 0, 0, 0, 0
    sv = sorted(vals)
    return sv[len(sv)//40], sv[len(sv)*39//40], sum(vals)/len(vals), len(blist)

rev_lo, rev_hi, rev_mean, rev_nb = block_ci(ap, 'revZ', True)
rev_test = [p for p in ap if not p['isTrain']]
rev_tlo, rev_thi, rev_tm, rev_tb = block_ci(rev_test, 'revZ', True)
print(f"Full: CI[{rev_lo:.1f},{rev_hi:.1f}] nBlocks={rev_nb}")
print(f"Test: CI[{rev_tlo:.1f},{rev_thi:.1f}] nBlocks={rev_tb}")

# ====== GRU ======
print("\n=== GRU ===")
old_sig = json.load(open(PROJECT/'data'/'p3-kronos-lstm-signals.json'))
gru_preds = old_sig.get('gru_wf', {})

gru_ap = []
for tp in tps:
    if tp['alpha'] is None: continue
    k = f"{tp['stockCode']}|{tp['cutoffDate']}"
    v = gru_preds.get(k)
    if v is None or np.isnan(v): continue
    gru_ap.append({'alpha': tp['alpha'], 'sv': v, 'code': tp['stockCode'], 'cutoff': tp['cutoffDate'], 'isTrain': tp['isTrain']})

print(f"GRU pairs: {len(gru_ap)}")
if gru_ap:
    gru_lo, gru_hi, gru_mean, gru_nb = block_ci(gru_ap, 'sv', False)
    gru_test = [p for p in gru_ap if not p['isTrain']]
    gru_tlo, gru_thi, gru_tm, gru_tb = block_ci(gru_test, 'sv', False)
    print(f"Full: CI[{gru_lo:.1f},{gru_hi:.1f}] nBlocks={gru_nb}")
    print(f"Test: CI[{gru_tlo:.1f},{gru_thi:.1f}] nBlocks={gru_tb}")
    print(f"\n=== VERDICT ===")
    if gru_tlo > 0: print("GRU test CI > 0 → GATE PASSED ✓")
    elif gru_thi < 0: print("GRU test CI < 0 → FAILED")
    else: print("GRU test CI includes 0 → FINAL: boundary signal, not verifiable")
