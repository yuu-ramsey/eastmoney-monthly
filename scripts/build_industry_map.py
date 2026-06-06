"""Build complete stock→industry mapping from Shenwan XLS + existing mappings"""
import json
import pandas as pd

# 1. Load existing mapping (industry names for 403 stocks)
with open('data/industry-map.json', 'r', encoding='utf-8') as f:
    old_map = json.load(f)
stock_to_name = old_map.get('stockToIndustry', {})
print(f'Existing mapping: {len(stock_to_name)} stocks')
# Count industries
from collections import Counter
name_counts = Counter(stock_to_name.values())
for name, n in name_counts.most_common():
    print(f'  {name}: {n}')

# 2. Load Shenwan XLS
df = pd.read_excel('data/SwClass2021_stock.xls')
df['stock_code'] = df['股票代码'].astype(str).str.zfill(6)
df['industry_code'] = df['行业代码'].astype(str)
df['inclusion_date'] = pd.to_datetime(df['计入日期'])
print(f'\nXLS: {len(df)} rows, {df["stock_code"].nunique()} unique stocks')
print(f'Unique L3 industry codes: {df["industry_code"].nunique()}')

# 3. Get latest industry per stock
latest = df.sort_values('inclusion_date').groupby('stock_code').last().reset_index()
latest_map = dict(zip(latest['stock_code'], latest['industry_code']))
print(f'Latest industry per stock: {len(latest_map)}')

# 4. Build L3_code → industry_name from known mapping
code_to_name: dict[str, str] = {}  # L3_code → name
conflicts: dict[str, set[str]] = {}  # L3_code → {multiple names}

for stock, name in stock_to_name.items():
    if stock in latest_map:
        l3 = latest_map[stock]
        if l3 in code_to_name:
            if code_to_name[l3] != name:
                if l3 not in conflicts:
                    conflicts[l3] = {code_to_name[l3]}
                conflicts[l3].add(name)
        else:
            code_to_name[l3] = name

print(f'\nL3 code→name mapping: {len(code_to_name)} codes')
print(f'Conflicts: {len(conflicts)} codes')
for l3, names in list(conflicts.items())[:10]:
    print(f'  {l3}: {names}')

# 5. Resolve conflicts: use most frequent name
l3_counter: dict[str, Counter] = {}
for stock, name in stock_to_name.items():
    if stock in latest_map:
        l3 = latest_map[stock]
        if l3 not in l3_counter:
            l3_counter[l3] = Counter()
        l3_counter[l3][name] += 1

code_to_name_final: dict[str, str] = {}
for l3, counter in l3_counter.items():
    code_to_name_final[l3] = counter.most_common(1)[0][0]
print(f'\nFinal L3 code→name mapping: {len(code_to_name_final)} codes')

# 6. Apply to all stocks
new_stock_to_industry: dict[str, str] = {}
unmapped: list[str] = []
for stock, l3 in latest_map.items():
    if l3 in code_to_name_final:
        new_stock_to_industry[stock] = code_to_name_final[l3]
    else:
        unmapped.append(l3)

print(f'\nMapping result: {len(new_stock_to_industry)}/{len(latest_map)} stocks ({100*len(new_stock_to_industry)/len(latest_map):.1f}%)')
print(f'Unmapped L3 codes: {len(set(unmapped))}')

# Stock count per industry
final_counts = Counter(new_stock_to_industry.values())
print(f'\nStocks per industry:')
for name, n in final_counts.most_common():
    print(f'  {name}: {n}')

# 7. For unmapped L3 codes, try L2 or L1 level inference
if unmapped:
    # Build L2→name mapping (first 4 digits of L3)
    l2_to_name: dict[str, Counter] = {}
    for l3, name in code_to_name_final.items():
        l2 = l3[:4]
        if l2 not in l2_to_name:
            l2_to_name[l2] = Counter()
        l2_to_name[l2][name] += 1

    l2_final = {l2: counter.most_common(1)[0][0] for l2, counter in l2_to_name.items()}

    # Build L1→name mapping (first 2 digits of L3)
    l1_to_name: dict[str, Counter] = {}
    for l3, name in code_to_name_final.items():
        l1 = l3[:2]
        if l1 not in l1_to_name:
            l1_to_name[l1] = Counter()
        l1_to_name[l1][name] += 1

    l1_final = {l1: counter.most_common(1)[0][0] for l1, counter in l1_to_name.items()}
    print(f'\nL1 mapping: {l1_final}')

    # Apply L2/L1 inference
    extra_mapped = 0
    for stock, l3 in latest_map.items():
        if stock in new_stock_to_industry:
            continue
        l2 = l3[:4]
        l1 = l3[:2]
        if l2 in l2_final:
            new_stock_to_industry[stock] = l2_final[l2]
            extra_mapped += 1
        elif l1 in l1_final:
            new_stock_to_industry[stock] = l1_final[l1]
            extra_mapped += 1

    print(f'\nL2/L1 inference additions: {extra_mapped}')
    print(f'Final mapping: {len(new_stock_to_industry)}/{len(latest_map)} ({100*len(new_stock_to_industry)/len(latest_map):.1f}%)')

# 8. Save
final_counts = Counter(new_stock_to_industry.values())
output = {
    'description': 'Shenwan level-1 industry classification for A-shares, from swsresearch.com XLS',
    'source': 'https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls',
    'fetchDate': '2026-05-24',
    'version': '2026-05-v3',
    'method': 'Cross-referenced 403 known stocks with XLS internal L3 codes, extended via L2/L1 fallback',
    'stockCount': len(new_stock_to_industry),
    'industryCount': len(final_counts),
    'industries': [{'name': name, 'stockCount': n} for name, n in final_counts.most_common()],
    'stockToIndustry': new_stock_to_industry
}

out_path = 'data/industry-map.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f'\nSaved to {out_path}')
print(f'Total stocks: {len(new_stock_to_industry)}')
print(f'Industries: {len(final_counts)}')
