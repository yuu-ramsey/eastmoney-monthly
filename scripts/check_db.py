"""Check industry mapping in SQLite DB"""
import sqlite3, os, glob

# Find DB files
db_patterns = ['.eastmoney-ai/*.db', '.eastmoney-ai/**/*.db', '*.db']
for pat in db_patterns:
    matches = glob.glob(pat)
    for m in matches:
        print(f'DB: {m} ({os.path.getsize(m)} bytes)')

# Try main DB
db_path = '.eastmoney-ai/ema.db'
if os.path.exists(db_path):
    db = sqlite3.connect(db_path)
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f'\nTables in {db_path}: {[t[0] for t in tables]}')

    for t in tables:
        tname = t[0]
        if 'industry' in tname.lower() or 'sector' in tname.lower():
            count = db.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
            cols = [d[0] for d in db.execute(f"SELECT * FROM {tname} LIMIT 0").description]
            print(f'\n{tname}: {count} rows, columns={cols}')
            rows = db.execute(f"SELECT * FROM {tname} LIMIT 3").fetchall()
            for r in rows:
                print(f'  {r}')
    db.close()
else:
    print(f'{db_path} not found')
