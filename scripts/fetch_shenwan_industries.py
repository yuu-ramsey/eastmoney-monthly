"""Fetch all Shenwan Level 1 industry constituents from legulegu.com, write to data/industry-map.json"""
import ssl, urllib3
urllib3.disable_warnings()
import requests
import re
import json
import time

SW_INDUSTRIES = [
    {'code': '801010.SI', 'name': 'Agriculture, Forestry, Animal Husbandry & Fishery'},
    {'code': '801030.SI', 'name': 'Basic Chemicals'},
    {'code': '801040.SI', 'name': 'Steel'},
    {'code': '801050.SI', 'name': 'Non-ferrous Metals'},
    {'code': '801080.SI', 'name': 'Electronics'},
    {'code': '801880.SI', 'name': 'Automobiles'},
    {'code': '801110.SI', 'name': 'Home Appliances'},
    {'code': '801120.SI', 'name': 'Food & Beverage'},
    {'code': '801130.SI', 'name': 'Textile & Apparel'},
    {'code': '801140.SI', 'name': 'Light Manufacturing'},
    {'code': '801150.SI', 'name': 'Pharmaceutical & Biotech'},
    {'code': '801160.SI', 'name': 'Public Utilities'},
    {'code': '801170.SI', 'name': 'Transportation'},
    {'code': '801180.SI', 'name': 'Real Estate'},
    {'code': '801200.SI', 'name': 'Commercial Retail'},
    {'code': '801210.SI', 'name': 'Social Services'},
    {'code': '801780.SI', 'name': 'Banking'},
    {'code': '801790.SI', 'name': 'Non-bank Financials'},
    {'code': '801230.SI', 'name': 'Conglomerates'},
    {'code': '801710.SI', 'name': 'Building Materials'},
    {'code': '801720.SI', 'name': 'Building Decoration'},
    {'code': '801730.SI', 'name': 'Electrical Equipment'},
    {'code': '801890.SI', 'name': 'Mechanical Equipment'},
    {'code': '801740.SI', 'name': 'Defense & Military'},
    {'code': '801750.SI', 'name': 'Computers'},
    {'code': '801760.SI', 'name': 'Media'},
    {'code': '801770.SI', 'name': 'Communications'},
    {'code': '801950.SI', 'name': 'Coal'},
    {'code': '801960.SI', 'name': 'Petroleum & Petrochemicals'},
    {'code': '801970.SI', 'name': 'Environmental Protection'},
    {'code': '801980.SI', 'name': 'Beauty & Personal Care'},
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

def fetch_industry_stocks(industry_code: str) -> list[str]:
    """Fetch all constituent stock codes for a Shenwan industry from legulegu"""
    code = industry_code.replace('.SI', '')
    url = f'https://legulegu.com/stockdata/index-composition?industryCode={code}'
    r = requests.get(url, headers=HEADERS, timeout=30, verify=False)
    if r.status_code != 200:
        raise Exception(f'HTTP {r.status_code}')

    html = r.text
    stocks: list[str] = []
    row_re = re.compile(r'<tr class="index-basic-composition-item">([\s\S]*?)</tr>', re.IGNORECASE)
    for row_match in row_re.finditer(html):
        row = row_match.group(1)
        code_match = re.search(r'>(\d{6})\.(?:SH|SZ)<', row)
        if code_match:
            stocks.append(code_match.group(1))
    return stocks


def main() -> None:
    stock_to_industry: dict[str, str] = {}
    industry_counts: dict[str, int] = {}

    for ind in SW_INDUSTRIES:
        try:
            stocks = fetch_industry_stocks(ind['code'])
            for sc in stocks:
                stock_to_industry[sc] = ind['name']
            industry_counts[ind['name']] = len(stocks)
            print(f'  {ind["name"]} ({ind["code"]}): {len(stocks)} stocks')
        except Exception as e:
            print(f'  {ind["name"]} ({ind["code"]}): ERROR - {e}')
        time.sleep(2)

    print(f'\nTotal unique stocks mapped: {len(stock_to_industry)}')
    for name, count in sorted(industry_counts.items(), key=lambda x: -x[1]):
        print(f'  {name}: {count}')

    output = {
        'description': 'Shenwan level-1 industry classification for A-shares, from legulegu.com',
        'source': 'https://legulegu.com/stockdata/sw-industry-overview',
        'fetchDate': '2026-05-24',
        'version': '2026-05-v3',
        'industryCount': len(SW_INDUSTRIES),
        'stockCount': len(stock_to_industry),
        'industries': [
            {'code': i['code'], 'name': i['name'], 'stockCount': industry_counts.get(i['name'], 0)}
            for i in SW_INDUSTRIES
        ],
        'stockToIndustry': stock_to_industry
    }

    out_path = 'data/industry-map.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\nSaved to {out_path}')

if __name__ == '__main__':
    main()
