"""Debug swsresearch.com response content"""
import ssl, urllib3
urllib3.disable_warnings()
import requests, re

r = requests.get(
    'https://www.swsresearch.com/institute_sw/allIndex/releasedIndex/releasedetail?code=801001&name=申万50',
    headers={'User-Agent': 'Mozilla/5.0'},
    timeout=15,
    verify=False
)
html = r.text
print(f'Status: {r.status_code}, Size: {len(html)}')

# Search for stock codes
codes = re.findall(r'\d{6}\.(?:SH|SZ)', html)
print(f'Stock codes found: {len(codes)}')
if codes:
    print(f'Sample: {codes[:10]}')

# Search for table/component patterns
for pat in ['component', 'constituent', '股票', '成分', 'stock']:
    matches = re.findall(pat, html, re.IGNORECASE)
    print(f'"{pat}": {len(matches)} occurrences')

# Check if data in scripts
scripts = re.findall(r'<script[^>]*>([\s\S]*?)</script>', html, re.IGNORECASE)
print(f'\nScripts: {len(scripts)}')
for i, s in enumerate(scripts):
    if len(s) > 200 and ('stock' in s.lower() or 'code' in s.lower() or 'data' in s.lower()):
        print(f'\nScript {i} ({len(s)} chars):')
        print(s[:500])
