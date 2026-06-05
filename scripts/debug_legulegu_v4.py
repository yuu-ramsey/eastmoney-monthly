"""Debug legulegu - check if data already in raw HTML"""
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
print(f'HTML size: {len(html)}')

# Check for index-basic-composition-item
matches = re.findall(r'index-basic-composition-item', html)
print(f'"index-basic-composition-item" occurrences: {len(matches)}')

# Check for stock code patterns in HTML (6-digit.SH or .SZ)
codes = re.findall(r'\d{6}\.(?:SH|SZ)', html)
print(f'Stock codes (XXXXXX.SH/SZ): {len(codes)}')
if codes:
    print(f'Sample: {codes[:10]}')

# Check for data attribute patterns
data_codes = re.findall(r'data-stock-code', html)
print(f'data-stock-code: {len(data_codes)}')

# Check for the table structure
table_section = re.search(r'<table[^>]*id="tableID"[^>]*>([\s\S]*?)</table>', html, re.IGNORECASE)
if table_section:
    content = table_section.group(1)
    print(f'\nTable content size: {len(content)}')
    print(f'Table content preview:')
    print(content[:2000])
else:
    print('\nNo table#tableID found')

# Check for data embedded in script
scripts = re.findall(r'<script[^>]*>([\s\S]*?)</script>', html, re.IGNORECASE)
for i, s in enumerate(scripts):
    if 'var ' in s and ('stock' in s.lower() or 'data' in s.lower() or 'list' in s.lower()):
        print(f'\nScript {i} with data vars:')
        # Extract variable assignments
        vars_found = re.findall(r'(var\s+\w+\s*=\s*[^;]+)', s)
        for v in vars_found[:10]:
            print(f'  {v[:200]}')

# The JS file name suggests index-basic-composition data is loaded separately
# Check for axios.get or similar
all_http = re.findall(r'(?:url|href|src)\s*[:=]\s*["\']([^"\']+)["\']', html)
potential_api = [u for u in all_http if 'api' in u.lower() or 'stock' in u.lower() or 'index' in u.lower()]
print(f'\nPotential API endpoints in HTML:')
for u in potential_api[:20]:
    print(f'  {u}')
