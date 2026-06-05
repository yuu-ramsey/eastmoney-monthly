"""Get Shenwan constituents via index_component_sw, cross-reference XLS internal codes to build mapping"""
import ssl, urllib3
urllib3.disable_warnings()
import requests
import json
import pandas as pd
import time
import re
from collections import Counter

# 31 Shenwan L1 industries
SW_INDUSTRIES = [
    ('801010', 'Agriculture, Forestry, Animal Husbandry & Fishery'), ('801030', 'Basic Chemicals'), ('801040', 'Steel'),
    ('801050', 'Non-ferrous Metals'), ('801080', 'Electronics'), ('801880', 'Automobiles'),
    ('801110', 'Home Appliances'), ('801120', 'Food & Beverage'), ('801130', '纺织服饰'),
    ('801140', 'Light Manufacturing'), ('801150', 'Pharmaceutical & Biotech'), ('801160', 'Public Utilities'),
    ('801170', 'Transportation'), ('801180', 'Real Estate'), ('801200', 'Commercial Retail'),
    ('801210', '社会服务'), ('801780', 'Banking'), ('801790', 'Non-bank Financials'),
    ('801230', '综合'), ('801710', 'Building Materials'), ('801720', 'Building Decoration'),
    ('801730', 'Electrical Equipment'), ('801890', 'Mechanical Equipment'), ('801740', 'Defense & Military'),
    ('801750', 'Computers'), ('801760', 'Media'), ('801770', 'Communications'),
    ('801950', '煤炭'), ('801960', '石油石化'), ('801970', '环保'),
    ('801980', '美容护理'),
]

# Load XLS
df = pd.read_excel('data/SwClass2021_stock.xls')
df['stock_code'] = df['股票代码'].astype(str).str.zfill(6)
df['industry_code'] = df['行业代码'].astype(str)
df['inclusion_date'] = pd.to_datetime(df['计入日期'])

# Latest industry per stock
latest = df.sort_values('inclusion_date').groupby('stock_code').last().reset_index()
xls_map = dict(zip(latest['stock_code'], latest['industry_code']))
print(f'XLS stocks: {len(xls_map)}')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

def fetch_index_stocks(index_code: str) -> list[str]:
    """获取申万某指数的成分股列表"""
    url = f'https://www.swsresearch.com/institute_sw/allIndex/releasedIndex/releasedetail?code={index_code}'
    r = requests.get(url, headers=HEADERS, timeout=30, verify=False)
    if r.status_code != 200:
        print(f'    HTTP {r.status_code}')
        return []
    html = r.text
    # 找成分股代码: 6位数字.SH或.SZ
    stocks = []
    for m in re.finditer(r'(\d{6})\.(?:SH|SZ)', html):
        stocks.append(m.group(1))
    return list(set(stocks))  # 去重

# 对每个行业获取成分股
l3_to_industry: dict[str, Counter] = {}  # L3_code → Counter of industry names
industry_stock_counts: dict[str, int] = {}

for ind_code, ind_name in SW_INDUSTRIES:
    try:
        stocks = fetch_index_stocks(ind_code)
        mapped = 0
        for sc in stocks:
            if sc in xls_map:
                l3 = xls_map[sc]
                if l3 not in l3_to_industry:
                    l3_to_industry[l3] = Counter()
                l3_to_industry[l3][ind_name] += 1
                mapped += 1
        industry_stock_counts[ind_name] = len(stocks)
        print(f'  {ind_name} ({ind_code}): {len(stocks)} stocks, {mapped} in XLS')
    except Exception as e:
        print(f'  {ind_name} ({ind_code}): ERROR - {e}')
    time.sleep(2)

# 解析映射
l3_final: dict[str, str] = {}
conflicts = 0
for l3, counter in l3_to_industry.items():
    top_name = counter.most_common(1)[0][0]
    l3_final[l3] = top_name
    if len(counter) > 1:
        conflicts += 1

print(f'\nL3→行业映射: {len(l3_final)} codes, {conflicts} conflicts')

# 未映射的L3代码尝试用L2推断
l2_to_name: dict[str, Counter] = {}
for l3, name in l3_final.items():
    l2 = l3[:4]
    if l2 not in l2_to_name:
        l2_to_name[l2] = Counter()
    l2_to_name[l2][name] += 1

l2_final = {l2: counter.most_common(1)[0][0] for l2, counter in l2_to_name.items()}
l1_to_name: dict[str, Counter] = {}
for l3, name in l3_final.items():
    l1 = l3[:2]
    if l1 not in l1_to_name:
        l1_to_name[l1] = Counter()
    l1_to_name[l1][name] += 1
l1_final = {l1: counter.most_common(1)[0][0] for l1, counter in l1_to_name.items()}

# 应用到全部股票
stock_to_industry: dict[str, str] = {}
for stock, l3 in xls_map.items():
    if l3 in l3_final:
        stock_to_industry[stock] = l3_final[l3]
    elif l3[:4] in l2_final:
        stock_to_industry[stock] = l2_final[l3[:4]]
    elif l3[:2] in l1_final:
        stock_to_industry[stock] = l1_final[l3[:2]]

final_counts = Counter(stock_to_industry.values())
print(f'\n最终映射: {len(stock_to_industry)}/{len(xls_map)} stocks ({100*len(stock_to_industry)/len(xls_map):.1f}%)')
print(f'行业数: {len(final_counts)}')
for name, n in final_counts.most_common():
    print(f'  {name}: {n}')

# 保存
output = {
    'description': 'Shenwan level-1 industry classification for A-shares',
    'source': 'swsresearch.com XLS + index_component_sw cross-reference',
    'fetchDate': '2026-05-24',
    'version': '2026-05-v3',
    'stockCount': len(stock_to_industry),
    'industryCount': len(final_counts),
    'industries': [{'name': name, 'stockCount': n} for name, n in final_counts.most_common()],
    'stockToIndustry': stock_to_industry
}

with open('data/industry-map.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f'\nSaved to data/industry-map.json')
