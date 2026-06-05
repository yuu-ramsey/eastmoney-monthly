"""调试 legulegu - 找JS数据源"""
import ssl, urllib3
urllib3.disable_warnings()
import requests, re, json

r = requests.get(
    'https://legulegu.com/stockdata/index-composition?industryCode=801010',
    headers={'User-Agent': 'Mozilla/5.0'},
    timeout=15,
    verify=False
)
html = r.text

# Find all script src
script_srcs = re.findall(r'<script[^>]*src="([^"]+)"', html, re.IGNORECASE)
print('Script sources:')
for s in script_srcs:
    print(f'  {s}')

# Find inline scripts with data
scripts = re.findall(r'<script[^>]*>([\s\S]*?)</script>', html, re.IGNORECASE)
print(f'\nInline scripts: {len(scripts)}')
for i, s in enumerate(scripts):
    if len(s) > 100 and ('api' in s.lower() or 'fetch' in s.lower() or 'ajax' in s.lower() or 'stock' in s.lower()):
        print(f'\nScript {i} (len={len(s)}):')
        print(s[:1000])

# Check for API calls
for pattern in ['/api/', '/stockdata/', 'fetch(', 'ajax(', 'getJSON', 'axios']:
    matches = re.findall(pattern, html, re.IGNORECASE)
    if matches:
        print(f'\n{pattern}: {len(matches)} matches')

# Search for data URLs
urls = re.findall(r'https?://[^"\']+', html)
api_urls = [u for u in urls if 'api' in u.lower() or 'data' in u.lower()]
print(f'\nPotential API URLs:')
for u in api_urls[:20]:
    print(f'  {u}')
