"""从Baostock获取CSRC行业分类（L1+L2），写入data/industry-map.json"""
import baostock as bs
import json
import re
from collections import Counter

lg = bs.login()
print(f'Login: {lg.error_code} {lg.error_msg}')

rs = bs.query_stock_industry()
industries = []
while (rs.error_code == '0') & rs.next():
    industries.append(rs.get_row_data())
bs.logout()
print(f'Total: {len(industries)} rows')

# Parse L1 + L2
stock_to_l2: dict[str, str] = {}  # L2 code like 'C39'
stock_to_l1: dict[str, str] = {}
l2_to_name: dict[str, str] = {}
l2_counts: Counter = Counter()

for row in industries:
    update_date, code, code_name, industry_str, classification = row
    if '.' in code:
        stock_code = code.split('.')[1]
    else:
        stock_code = code

    # Parse CSRC industry string: 'C39计算机、通信和其他电子设备制造业'
    # L2 = first 3 chars, name = rest
    if industry_str and len(industry_str) >= 3:
        l2 = industry_str[:3]
        name = industry_str[3:] if len(industry_str) > 3 else industry_str
        if re.match(r'^[A-S]\d{2}', l2):  # Valid CSRC code format
            stock_to_l2[stock_code] = l2
            stock_to_l1[stock_code] = l2[0]
            if l2 not in l2_to_name:
                l2_to_name[l2] = name
            l2_counts[l2] += 1

print(f'\nStocks with L2 industry: {len(stock_to_l2)}')
print(f'L2 categories: {len(l2_counts)}')
print(f'\nL2 categories (sorted by count):')
for l2, count in l2_counts.most_common():
    name = l2_to_name[l2]
    print(f'  {l2} {name}: {count}')

# Also save L1 for backward compatibility
l1_counts = Counter(stock_to_l1.values())
CSRC_L1 = {
    'A': '农、林、牧、渔业', 'B': '采矿业', 'C': '制造业',
    'D': '电力、热力、燃气及水生产和供应业', 'E': '建筑业',
    'F': '批发和零售业', 'G': '交通运输、仓储和邮政业',
    'H': '住宿和餐饮业', 'I': '信息传输、软件和信息技术服务业',
    'J': '金融业', 'K': '房地产业', 'L': '租赁和商务服务业',
    'M': '科学研究和技术服务业', 'N': '水利、环境和公共设施管理业',
    'O': '居民服务、修理和其他服务业', 'P': '教育',
    'Q': '卫生和社会工作', 'R': '文化、体育和娱乐业', 'S': '综合',
}

# Build L2 → L1 name mapping
l2_to_l1_name = {l2: CSRC_L1.get(l2[0], l2[0]) for l2 in l2_to_name}

# Output: use L2 code → L2 name as the main mapping
stock_to_industry_name = {}
for stock, l2 in stock_to_l2.items():
    stock_to_industry_name[stock] = l2_to_name.get(l2, l2)

output = {
    'description': 'CSRC (证监会) level-2 industry classification for A-shares, from Baostock',
    'source': 'Baostock query_stock_industry()',
    'fetchDate': '2026-05-24',
    'version': '2026-05-v1',
    'classification': 'CSRC L2',
    'stockCount': len(stock_to_l2),
    'industryCount': len(l2_counts),
    'industries': [
        {
            'code': l2,
            'name': l2_to_name[l2],
            'l1Name': l2_to_l1_name.get(l2, ''),
            'stockCount': count
        }
        for l2, count in l2_counts.most_common()
    ],
    'stockToIndustry': stock_to_industry_name  # L2 name as value
}

with open('data/industry-map.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f'\nSaved to data/industry-map.json')
print(f'Coverage: {len(stock_to_l2)} stocks, {len(l2_counts)} L2 industries')
print(f'Max industry: {l2_counts.most_common(1)[0][1]} stocks ({l2_counts.most_common(1)[0][1]/len(stock_to_l2)*100:.1f}%)')
