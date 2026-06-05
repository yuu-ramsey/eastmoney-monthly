"""Try all possible industry classification data sources"""
import ssl, urllib3
urllib3.disable_warnings()
import requests
import json
import time

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# ====== Source 1: Try different swsresearch.com URLs ======
print("===== Source 1: swsresearch.com files =====")
base_urls = [
    'https://www.swsresearch.com/swindex/pdf/SwClass2021/SwClass2021.xls',
    'https://www.swsresearch.com/swindex/pdf/SwClass2021/IndustryClass.xls',
    'https://www.swsresearch.com/swindex/pdf/SwClass2021/industry_code.xls',
    'https://www.swsresearch.com/swindex/pdf/SwClass2021/IndustryName.xls',
    'https://www.swsresearch.com/swindex/pdf/SwClass2021.xls',
    'https://www.swsresearch.com/institute_sw/allIndex/downloadCenter/industryType',
    'https://www.swsresearch.com/swindex/pdf/sw_class.xls',
]
for url in base_urls:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        print(f'{url}: HTTP {r.status_code}, Size={len(r.content)}')
        if r.status_code == 200 and len(r.content) > 500:
            # Save for inspection
            fname = url.split('/')[-1].split('?')[0] or 'download'
            with open(f'data/{fname}', 'wb') as f:
                f.write(r.content)
            print(f'  -> Saved to data/{fname}')
    except Exception as e:
        print(f'{url}: Error - {e}')

# ====== Source 2: Try AKShare with SSL disabled ======
print("\n===== Source 2: AKShare stock_industry_clf_hist_sw =====")
try:
    # Monkey-patch requests to disable SSL
    import akshare as ak
    # This function downloads from swsresearch.com - we already have the XLS
    # It returns: symbol, start_date, industry_code, update_time
    # No industry names available
    print("AKShare function only returns internal codes (no names)")
except Exception as e:
    print(f'Error: {e}')

# ====== Source 3: Try Baostock for CSRC → try to map to Shenwan ======
print("\n===== Source 3: Baostock =====")
try:
    import baostock as bs
    lg = bs.login()
    print(f'Login: {lg.error_code} {lg.error_msg}')

    # Get industry classification for a sample
    rs = bs.query_stock_industry()
    print(f'query_stock_industry: {rs.error_code} {rs.error_msg}')

    industries = []
    while (rs.error_code == '0') & rs.next():
        industries.append(rs.get_row_data())
    print(f'Industries returned: {len(industries)}')
    if industries:
        print(f'Sample: {industries[:5]}')
        print(f'Columns: updateDate, code, code_name, industry, industryClassification')

    bs.logout()
except Exception as e:
    print(f'Error: {e}')

# ====== Source 4: Try Tushare ======
print("\n===== Source 4: Tushare =====")
try:
    import tushare as ts
    # Check if there's a token set
    # Try with no token (might have limited access)
    pro = ts.pro_api()
    print("Tushare pro_api initialized (may need token)")
except Exception as e:
    print(f'Error: {e}')
