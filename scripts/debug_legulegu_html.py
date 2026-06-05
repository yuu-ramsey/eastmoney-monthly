"""调试 legulegu HTML 结构"""
import ssl, urllib3
urllib3.disable_warnings()
import requests, re

r = requests.get(
    'https://legulegu.com/stockdata/index-composition?industryCode=801010',
    headers={'User-Agent': 'Mozilla/5.0'},
    timeout=15,
    verify=False
)
html = r.text

# Find tbody
table_match = re.search(r'<tbody[^>]*>([\s\S]*?)</tbody>', html, re.IGNORECASE)
if table_match:
    tbody = table_match.group(1)
    rows = re.findall(r'<tr[^>]*>([\s\S]*?)</tr>', tbody, re.IGNORECASE)
    print(f'Rows in tbody: {len(rows)}')
    for i, row in enumerate(rows[:5]):
        tds = re.findall(r'<td[^>]*>([\s\S]*?)</td>', row, re.IGNORECASE)
        clean = [re.sub(r'<[^>]+>', '', td).strip() for td in tds]
        print(f'Row {i}: {clean}')
else:
    print('No tbody found')
    # Try any tr
    trs = re.findall(r'<tr[^>]*>([\s\S]*?)</tr>', html, re.IGNORECASE)
    print(f'Total tr tags: {len(trs)}')
    for i, tr in enumerate(trs[:10]):
        if 'stock' in tr.lower() or 'code' in tr.lower() or 'td' in tr.lower()[:100]:
            tds = re.findall(r'<td[^>]*>([\s\S]*?)</td>', tr, re.IGNORECASE)
            clean = [re.sub(r'<[^>]+>', '', td).strip() for td in tds]
            if clean:
                print(f'Row {i}: {clean}')
