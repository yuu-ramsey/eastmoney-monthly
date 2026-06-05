import sqlite3
c = sqlite3.connect('.eastmoney-ai/lstm/overnight.db')
rows = c.execute("SELECT status, COUNT(*) FROM experiments GROUP BY status").fetchall()
print("Status:", {r[0]: r[1] for r in rows})
top = c.execute("SELECT id, phase, ic_y3 FROM experiments WHERE status='done' ORDER BY ic_y3 DESC LIMIT 5").fetchall()
print("Top 5:", [(r[0], r[1].split('/')[0], round(r[2], 4)) for r in top])
pend = c.execute("SELECT COUNT(*) FROM experiments WHERE status='pending'").fetchone()[0]
print(f"Remaining: {pend}")
c.close()
