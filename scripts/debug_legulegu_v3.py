"""Debug legulegu - fetch JS files to find API"""
import ssl, urllib3
urllib3.disable_warnings()
import requests, re

# Get the JS file
r = requests.get(
    'https://legulegu.com/static/js/index-basic-composition.min.js?date=20260523',
    headers={'User-Agent': 'Mozilla/5.0'},
    timeout=15,
    verify=False
)
js = r.text
print(f'JS size: {len(js)}')

# Search for API endpoints
api_patterns = re.findall(r'["\'](/[^"\']*api[^"\']*)["\']', js, re.IGNORECASE)
print(f'\nAPI endpoints in JS:')
for p in api_patterns:
    print(f'  {p}')

# Search for all URL patterns
urls = re.findall(r'["\']((?:/[a-zA-Z][^"\']{3,}))["\']', js)
print(f'\nAll endpoints:')
for u in urls[:30]:
    print(f'  {u}')

# Search for ajax/fetch calls
for pattern in [r'\.ajax\(', r'\.get\(', r'\.post\(', r'fetch\(', r'axios\.']:
    matches = [(m.start(), m.group()) for m in re.finditer(pattern, js)]
    if matches:
        print(f'\n{pattern}: {len(matches)} matches')
        for start, match in matches[:5]:
            print(f'  ...{js[max(0,start-50):start+200]}...')
