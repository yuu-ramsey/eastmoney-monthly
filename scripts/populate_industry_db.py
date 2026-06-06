"""Write CSRC industry mapping to DB stock_industry_mapping table"""
import sqlite3, json, os, time

PROJECT = '.'
DB = '.eastmoney-ai/db/klines-v2.sqlite'

print(f'DB exists: {os.path.exists(DB)}, Size: {os.path.getsize(DB)}')

db = sqlite3.connect(DB)

# Check table
tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(f'Tables: {[t[0] for t in tables]}')

for t in tables:
    tname = t[0]
    cnt = db.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
    print(f'  {tname}: {cnt} rows')

# Check monthly_klines
cnt = db.execute("SELECT COUNT(DISTINCT code) FROM monthly_klines").fetchone()[0]
print(f'\nmonthly_klines unique codes: {cnt}')
cnt84 = db.execute(
    "SELECT COUNT(DISTINCT code) FROM (SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84)"
).fetchone()[0]
print(f'codes with >=84 months: {cnt84}')

# Load new industry mapping
with open('data/industry-map.json', 'r', encoding='utf-8') as f:
    ind_map = json.load(f)
stock_to_ind = ind_map['stockToIndustry']
print(f'\nIndustry map: {len(stock_to_ind)} stocks, {ind_map["industryCount"]} industries')

# Create stock_industry_mapping table
snapshot_date = '2026-05-24'
db.execute("DROP TABLE IF EXISTS stock_industry_mapping")
db.execute("""
    CREATE TABLE stock_industry_mapping (
        stock_code TEXT NOT NULL,
        industry_code TEXT NOT NULL,
        industry_name TEXT,
        snapshot_date TEXT,
        PRIMARY KEY (stock_code, industry_code)
    )
""")

# Write data
insert_sql = "INSERT OR REPLACE INTO stock_industry_mapping (stock_code, industry_code, industry_name, snapshot_date) VALUES (?, ?, ?, ?)"
data = [(sc, ind, ind, snapshot_date) for sc, ind in stock_to_ind.items()]
db.executemany(insert_sql, data)
db.commit()

# Verify
cnt = db.execute("SELECT COUNT(*) FROM stock_industry_mapping").fetchone()[0]
print(f'\nWritten: {cnt} rows to stock_industry_mapping')

# Check coverage
monthly_codes = set()
for row in db.execute("SELECT DISTINCT code FROM monthly_klines"):
    monthly_codes.add(row[0])

mapped = [c for c in monthly_codes if c in stock_to_ind]
print(f'Monthly stocks covered: {len(mapped)}/{len(monthly_codes)} ({100*len(mapped)/len(monthly_codes):.1f}%)')

codes_84 = set()
for row in db.execute("SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84"):
    codes_84.add(row[0])

mapped_84 = [c for c in codes_84 if c in stock_to_ind]
print(f'>=84-month stocks covered: {len(mapped_84)}/{len(codes_84)} ({100*len(mapped_84)/len(codes_84):.1f}%)')

db.close()
print('\nDone!')
