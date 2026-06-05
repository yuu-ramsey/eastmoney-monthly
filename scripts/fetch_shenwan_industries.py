"""从 legulegu.com 获取申万一级行业全部成分股，写入 data/industry-map.json"""
import ssl, urllib3
urllib3.disable_warnings()
import requests
import re
import json
import time

SW_INDUSTRIES = [
    {'code': '801010.SI', 'name': '农林牧渔'},
    {'code': '801030.SI', 'name': '基础化工'},
    {'code': '801040.SI', 'name': '钢铁'},
    {'code': '801050.SI', 'name': '有色金属'},
    {'code': '801080.SI', 'name': '电子'},
    {'code': '801880.SI', 'name': '汽车'},
    {'code': '801110.SI', 'name': '家用电器'},
    {'code': '801120.SI', 'name': '食品饮料'},
    {'code': '801130.SI', 'name': '纺织服饰'},
    {'code': '801140.SI', 'name': '轻工制造'},
    {'code': '801150.SI', 'name': '医药生物'},
    {'code': '801160.SI', 'name': '公用事业'},
    {'code': '801170.SI', 'name': '交通运输'},
    {'code': '801180.SI', 'name': '房地产'},
    {'code': '801200.SI', 'name': '商贸零售'},
    {'code': '801210.SI', 'name': '社会服务'},
    {'code': '801780.SI', 'name': '银行'},
    {'code': '801790.SI', 'name': '非银金融'},
    {'code': '801230.SI', 'name': '综合'},
    {'code': '801710.SI', 'name': '建筑材料'},
    {'code': '801720.SI', 'name': '建筑装饰'},
    {'code': '801730.SI', 'name': '电力设备'},
    {'code': '801890.SI', 'name': '机械设备'},
    {'code': '801740.SI', 'name': '国防军工'},
    {'code': '801750.SI', 'name': '计算机'},
    {'code': '801760.SI', 'name': '传媒'},
    {'code': '801770.SI', 'name': '通信'},
    {'code': '801950.SI', 'name': '煤炭'},
    {'code': '801960.SI', 'name': '石油石化'},
    {'code': '801970.SI', 'name': '环保'},
    {'code': '801980.SI', 'name': '美容护理'},
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

def fetch_industry_stocks(industry_code: str) -> list[str]:
    """从legulegu获取某个申万行业的所有成分股代码"""
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
