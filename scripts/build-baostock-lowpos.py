"""Pull Baostock monthly klines for existing codes + delisted, build low-pos pool."""
import baostock as bs
import json, os, random, sqlite3
from datetime import datetime

TIMEPOINTS = ['2018-06','2018-12','2019-06','2020-03','2020-09','2021-06',
              '2021-12','2022-06','2022-10','2023-06','2024-02','2024-10']
N_MONTHS, MA_PERIOD, HORIZON, MAX_PER_TP = 12, 60, 6, 140
X_PCT = 0.20

print("=== Baostock low-pos pool (targeted) ===\n")
bs.login()

# Get existing codes from our DB + known delisted
codes = set()
try:
    db = sqlite3.connect(os.path.join('.eastmoney-ai','db','klines-v2.sqlite'))
    for r in db.execute("SELECT DISTINCT code FROM monthly_klines"):
        c = r[0]
        codes.add(f"sh.{c}" if c.startswith('6') else f"sz.{c}")
    db.close()
except: pass

DELISTED = ['sz.000033','sz.000511','sz.002070','sz.000018','sh.600401','sh.600432',
            'sz.000003','sz.000015','sz.000047','sz.000062','sh.600069','sh.600074',
            'sh.600077','sh.600087','sh.600091','sh.600145','sh.600149','sh.600175',
            'sh.600179','sh.600186','sh.600202','sh.600209','sh.600212','sh.600215',
            'sh.600234','sh.600240','sh.600242','sh.600247','sh.600265','sh.600275',
            'sh.600281','sh.600289','sh.600301','sh.600306','sh.600311','sh.600317',
            'sh.600401','sh.600408','sh.600421','sh.600423','sh.600432','sh.600539']
codes.update(DELISTED)
codes = list(codes)
print(f"Codes: {len(codes)} (from DB: {len(codes)-len(DELISTED)}, delisted add: {len(DELISTED)})")

# Cache: code -> [(ym, close), ...]
cache = {}
def get_monthly(code):
    if code in cache: return cache[code]
    try:
        rs = bs.query_history_k_data_plus(code, 'date,close', start_date='2010-01-01', end_date='2026-05-31', frequency='m', adjustflag='2')
        rows = []
        while (rs.error_code == '0') and rs.next():
            d = rs.get_row_data()
            c = float(d[1]) if d[1] and d[1] != '' else None
            if c and c > 0.001: rows.append((d[0][:7], c))
        cache[code] = rows
        return rows
    except:
        cache[code] = []
        return []

print("Pulling monthly data...")
pulled = 0
for code in codes:
    if code not in cache:
        get_monthly(code)
        pulled += 1
        if pulled % 500 == 0: print(f"  {pulled}/{len(codes)}...")
print(f"Pulled {pulled} new, cache size: {len(cache)}")
# Save klines cache for later baseline computation
with open('data/baostock-klines-cache.json', 'w') as f:
    json.dump(cache, f, ensure_ascii=False)
print("Klines cache saved to data/baostock-klines-cache.json")

# Build pool
results = {'perTimepoint': {}, 'testPoints': []}
tp_id = 0
random.seed(42)
db_codes = set()

print("\nBuilding pool...")
for t in TIMEPOINTS:
    candidates = []
    for code in codes:
        data = get_monthly(code)
        if len(data) < MA_PERIOD: continue
        ci = -1
        for j, (d, _) in enumerate(data):
            if d == t: ci = j; break
        if ci < MA_PERIOD: continue
        close = data[ci][1]
        past = [data[j][1] for j in range(max(0,ci-N_MONTHS),ci) if data[j][1]>0.001]
        if len(past) < N_MONTHS: continue
        rp = (close-min(past))/(max(past)-min(past)) if max(past)>min(past) else 0.5
        ma_cl = [data[j][1] for j in range(max(0,ci-MA_PERIOD),ci+1) if data[j][1]>0.001]
        if len(ma_cl) < MA_PERIOD: continue
        ma60 = sum(ma_cl)/len(ma_cl)
        if rp <= X_PCT and close < ma60:
            fwd = [data[j][1] for j in range(ci+1,min(len(data),ci+HORIZON+1)) if data[j][1]>0.001]
            alpha = (fwd[HORIZON-1]-close)/close*100 if len(fwd)>=HORIZON else None
            candidates.append((code,close,rp,ma60,alpha))

    if len(candidates) > MAX_PER_TP:
        candidates = random.sample(candidates, MAX_PER_TP)
    results['perTimepoint'][t] = {'n':len(candidates)}
    for code,close,rp,ma60,alpha in candidates:
        results['testPoints'].append({'id':f'bs_{tp_id}','stockCode':code.split('.')[-1],
            'baostockCode':code,'cutoffDate':t,
            'closeAtCutoff':round(close,2),'rangePosition':round(rp,4),
            'ma60':round(ma60,2),'alpha':round(alpha,2) if alpha else None})
        tp_id += 1
    print(f"  {t}: {len(candidates)}")

bs.logout()

results['version']='lowpos-v2-baostock'
results['createdAt']=datetime.now().isoformat()
results['source']='baostock_single_source'
results['config']={'nMonths':N_MONTHS,'xPct':X_PCT,'maPeriod':MA_PERIOD,'horizonMonths':HORIZON,'maxPerTimepoint':MAX_PER_TP}
results['summary']={'total':len(results['testPoints']),'uniqueCodes':len(set(tp['stockCode'] for tp in results['testPoints']))}

os.makedirs('data',exist_ok=True)
with open('data/frozen-eval-lowpos-v2-baostock.json','w') as f:
    json.dump(results,f,ensure_ascii=False)
print(f"\nSaved: {len(results['testPoints'])} testPoints")
