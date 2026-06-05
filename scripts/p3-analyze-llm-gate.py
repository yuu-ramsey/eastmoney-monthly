"""分析 LLM 24tp 门控结果"""
import json, numpy as np
from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / '.eastmoney-ai' / 'eval' / 'runs' / 'p3-llm-gate-24tp-20260604-232113.jsonl'

rows = []
for line in open(RESULTS, encoding='utf-8'):
    line = line.strip()
    if not line: continue
    rows.append(json.loads(line))

print(f"Total: {len(rows)}")

# 信号分布
sigs = Counter(r['signal'] for r in rows)
print(f"\nSignal distribution:")
for s, n in sigs.most_common():
    print(f"  {s}: {n} ({n/len(rows)*100:.1f}%)")

# 成本
total_cost = sum(r.get('cost', 0) for r in rows)
print(f"\nTotal cost: ¥{total_cost:.2f}")

# 排除 parse_failed 和 api_error
valid = [r for r in rows if r['signal'] not in ('parse_failed', 'api_error')]
print(f"Valid: {len(valid)}/{len(rows)}")

# ── 分桶分析 ──
def winsorize(arr, pct=1):
    s = sorted(arr)
    lo = s[max(0, len(s) // (100//pct) - 1)]
    hi = s[min(len(s) - 1, len(s) * (100-pct) // 100)]
    return [lo if x < lo else (hi if x > hi else x) for x in arr]

def wins_mean(arr):
    w = winsorize(arr)
    return sum(w) / len(w) if w else 0

signal_map = {'strong_bull': 2, 'bull': 1, 'neutral': 0, 'bear': -1, 'strong_bear': -2}

# 分桶
buckets = {'strong_bull': [], 'bull': [], 'neutral': [], 'bear': [], 'strong_bear': []}
for r in valid:
    buckets[r['signal']].append(r['alpha'])

print(f"\n=== Alpha by Signal ===")
for sig in ['strong_bull', 'bull', 'neutral', 'bear', 'strong_bear']:
    alphas = buckets[sig]
    if alphas:
        print(f"  {sig}: n={len(alphas)} alpha={wins_mean(alphas):+.1f}% raw={sum(alphas)/len(alphas):+.1f}%")

# Spread: bullish - bearish
bullish = [r for r in valid if r['signal'] in ('strong_bull', 'bull')]
bearish = [r for r in valid if r['signal'] in ('strong_bear', 'bear')]
bull_alpha = wins_mean([r['alpha'] for r in bullish]) if bullish else 0
bear_alpha = wins_mean([r['alpha'] for r in bearish]) if bearish else 0
spread = bull_alpha - bear_alpha
print(f"\n  bullish alpha={bull_alpha:+.1f}%, bearish alpha={bear_alpha:+.1f}%, spread={spread:+.1f}%")
print(f"  bullish n={len(bullish)}, bearish n={len(bearish)}")

# ── Block CI ──
def block_ci(rows_list, ascending=True):
    blocks = {}
    for r in rows_list:
        bk = f"{r['stockCode']}|{r['cutoffDate']}"
        blocks.setdefault(bk, []).append(r)
    blist = list(blocks.values())
    if len(blist) < 5:
        return 0, 0, 0, len(blist)
    vals = []
    for _ in range(5000):
        sample = []
        for _ in range(len(blist)):
            sample.extend(blist[np.random.randint(0, len(blist))])
        s = sorted(sample, key=lambda r: signal_map.get(r['signal'], 0), reverse=True)
        n20 = max(1, len(s) // 5)
        bulls = [r['alpha'] for r in s[:n20]]
        bears = [r['alpha'] for r in s[-n20:]]
        if bulls and bears:
            vals.append(wins_mean(bulls) - wins_mean(bears))
    if not vals:
        return 0, 0, 0, len(blist)
    sv = sorted(vals)
    return sv[len(sv) // 40], sv[len(sv) * 39 // 40], sum(vals) / len(vals), len(blist)

lo, hi, mean, nb = block_ci(valid, ascending=True)
print(f"\n=== LLM 24tp Block CI ===")
print(f"  CI [{lo:+.1f}, {hi:+.1f}] mean={mean:+.1f}% nBlocks={nb}")

# 方向命中率
correct = sum(1 for r in valid if signal_map.get(r['signal'], 0) != 0
              and np.sign(signal_map.get(r['signal'], 0)) == np.sign(r['alpha']))
total_nz = sum(1 for r in valid if signal_map.get(r['signal'], 0) != 0 and r['alpha'] != 0)
print(f"\n  Direction hit: {correct}/{total_nz} = {correct/total_nz*100:.1f}%" if total_nz else "")

# 门控判定
print(f"\n=== VERDICT ===")
if lo > 0:
    print(f"  LLM 24tp CI [{lo:+.1f}, {hi:+.1f}] > 0 -> GATE PASSED")
elif hi < 0:
    print(f"  LLM 24tp CI [{lo:+.1f}, {hi:+.1f}] < 0 -> FAILED")
else:
    print(f"  LLM 24tp CI [{lo:+.1f}, {hi:+.1f}] includes 0 -> boundary, not verifiable")
