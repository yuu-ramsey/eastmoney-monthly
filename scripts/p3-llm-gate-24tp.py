"""LLM 24tp re-gating — minimal prompt, pure LLM technical analysis capability.
Same methodology as Phase C 12tp, switched to 24tp pool.
DeepSeek only, resumable from checkpoint.
Usage: python scripts/p3-llm-gate-24tp.py [--sample N] [--dry]
  --sample N   Only run N pairs (default: full test set)
  --dry        Do not call API, verify data readiness
"""
import json, os, sys, time, re
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).resolve().parent.parent

# ── Load data ──
pool = json.load(open(PROJECT / 'data' / 'frozen-eval-lowpos-v2-24tp.json'))
kcache = json.load(open(PROJECT / 'data' / 'baostock-klines-cache.json'))

TRAIN = {'2018-03','2018-06','2018-09','2018-12','2019-03','2019-06','2019-09',
         '2020-09','2021-03','2021-06','2022-03','2022-06'}

# Deduplicate + filter
unique = {}
for tp in pool['testPoints']:
    if tp.get('alpha') is None: continue
    k = f"{tp['stockCode']}|{tp['cutoffDate']}"
    if k not in unique:
        unique[k] = tp

pairs = list(unique.values())
test_pairs = [p for p in pairs if p['cutoffDate'] not in TRAIN]
print(f"Pool: {len(pairs)} pairs (Train: {len(pairs)-len(test_pairs)}, Test: {len(test_pairs)})")

# CLI
sample_n = None
dry_run = False
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == '--sample' and i+1 < len(args):
        sample_n = int(args[i+1]); i += 2
    elif args[i] == '--dry':
        dry_run = True; i += 1
    else:
        i += 1

if sample_n and sample_n < len(test_pairs):
    import random
    random.seed(42)
    test_pairs = random.sample(test_pairs, sample_n)
    print(f"Sampled: {len(test_pairs)} pairs")

# ── Build minimal prompt ──
def get_klines(code, cutoff_date):
    full_key = None
    for k in kcache:
        if k.endswith('.' + code):
            full_key = k; break
    kl = kcache.get(code) or (kcache.get(full_key) if full_key else None)
    if not kl or len(kl) < 60:
        return None
    ci = -1
    for j, row in enumerate(kl):
        if isinstance(row, list) and row[0] == cutoff_date:
            ci = j; break
    if ci < 12:
        return None
    rows = []
    for j in range(max(0, ci-12), ci+1):
        close = f"{kl[j][1]:.2f}"
        rows.append(f"{kl[j][0]} {close}")
    return 'Date Close\n' + '\n'.join(rows)

def build_prompt(tp):
    kl = get_klines(tp['stockCode'], tp['cutoffDate'])
    if not kl:
        return None
    return (
        f"你是A股技术分析师。以下是{tp['stockCode']}近12+个月月线(前复权):\n"
        f"{kl}\n\n"
        f"该股处于低位——过去12月底部20%且<MA60。判断未来6个月方向。\n\n"
        f'输出JSON:\n```json\n{{"signal":"strong_bull|bull|neutral|bear|strong_bear"}}\n```\n'
        f"signal必五选一。"
    )

# ── Verify data readiness ──
ready = sum(1 for tp in test_pairs if build_prompt(tp))
print(f"Ready: {ready}/{len(test_pairs)} pairs have kline data")

if dry_run:
    print("Dry run done.")
    sys.exit(0)

# ── API ──
env_path = PROJECT / '.env'
env = {}
if env_path.exists():
    for line in open(env_path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        if '=' in line:
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip()
API_KEY = env.get('DEEPSEEK_API_KEY')
if not API_KEY:
    print("ERROR: DEEPSEEK_API_KEY not found in .env")
    sys.exit(1)

import urllib.request
import urllib.error

def call_llm(prompt):
    data = json.dumps({
        'model': 'deepseek-chat',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 4000,
        'temperature': 0.0,
    }).encode('utf-8')
    req = urllib.request.Request(
        'https://api.deepseek.com/chat/completions',
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}',
        }
    )
    resp = urllib.request.urlopen(req, timeout=60)
    d = json.loads(resp.read())
    return {
        'text': d['choices'][0]['message']['content'],
        'usage': {
            'input': d['usage']['prompt_tokens'],
            'output': d['usage']['completion_tokens'],
        }
    }

def retry_fn(fn, n=3):
    last_err = None
    delays = [1, 4, 16]
    for attempt in range(n + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < n:
                time.sleep(delays[attempt])
    raise last_err

def parse_signal(text):
    try:
        m = re.search(r'```json\s*([\s\S]*?)```', text)
        if m:
            obj = json.loads(m.group(1).strip())
            sig = obj.get('signal', 'parse_failed')
            if sig in ('strong_bull', 'bull', 'neutral', 'bear', 'strong_bear'):
                return sig
    except Exception:
        pass
    return 'parse_failed'

# ── Checkpoint / resume ──
RUNS_DIR = PROJECT / '.eastmoney-ai' / 'eval' / 'runs'
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = f"p3-llm-gate-24tp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
OUT_PATH = RUNS_DIR / f"{RUN_ID}.jsonl"

completed = set()
if OUT_PATH.exists():
    for line in open(OUT_PATH, encoding='utf-8'):
        line = line.strip()
        if not line: continue
        try:
            r = json.loads(line)
            completed.add(f"{r['stockCode']}|{r['cutoffDate']}")
        except Exception:
            pass

print(f"Completed: {len(completed)}")
pending = [(tp, build_prompt(tp)) for tp in test_pairs
           if f"{tp['stockCode']}|{tp['cutoffDate']}" not in completed
           and build_prompt(tp) is not None]
print(f"Pending: {len(pending)}")

if not pending:
    print("All done.")
    sys.exit(0)

total_cost = 0
ok = len(completed)
fail = 0
start = time.time()

for i in range(0, len(pending), 2):
    batch = pending[i:i+2]
    for tp, prompt in batch:
        try:
            result = retry_fn(lambda: call_llm(prompt))
            signal = parse_signal(result['text'])
            cost = result['usage']['input'] / 1e6 * 1.0 + result['usage']['output'] / 1e6 * 4.0
            row = {
                'stockCode': tp['stockCode'],
                'cutoffDate': tp['cutoffDate'],
                'signal': signal,
                'alpha': tp['alpha'],
                'isTrain': tp.get('isTrain', False),
                'cost': cost,
            }
            total_cost += cost
            ok += 1
        except Exception as e:
            print(f"  FAIL {tp['stockCode']}|{tp['cutoffDate']}: {e}")
            fail += 1
            row = {
                'stockCode': tp['stockCode'],
                'cutoffDate': tp['cutoffDate'],
                'signal': 'api_error',
                'alpha': tp['alpha'],
                'isTrain': tp.get('isTrain', False),
                'cost': 0,
                'error': str(e)[:200],
            }
        with open(OUT_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    done = ok + fail
    if done % 40 == 0 or done >= len(pending) + len(completed):
        elapsed = (time.time() - start) / 60
        print(f"  {done}/{len(pending)+len(completed)} ok={ok} fail={fail} ¥{total_cost:.2f} {elapsed:.1f}min")

    if i + 2 < len(pending):
        time.sleep(0.3)

print(f"\nDone: ¥{total_cost:.2f} → {OUT_PATH}")
