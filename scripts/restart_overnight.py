"""Restart orchestrator: fix stale running entries, then launch."""
import sqlite3, subprocess, sys, time
db = '.eastmoney-ai/lstm/overnight.db'
conn = sqlite3.connect(db)
stale = conn.execute("SELECT COUNT(*) FROM experiments WHERE status='running'").fetchone()[0]
conn.execute("UPDATE experiments SET status='pending' WHERE status='running'")
conn.commit()
done = conn.execute("SELECT COUNT(*) FROM experiments WHERE status='done'").fetchone()[0]
pend = conn.execute("SELECT COUNT(*) FROM experiments WHERE status='pending'").fetchone()[0]
conn.close()
print(f"Fixed {stale} stale running. {done} done, {pend} pending.")
print(f"Launching orchestrator...")
subprocess.Popen([sys.executable, 'cli/overnight/orchestrator.py'], creationflags=subprocess.CREATE_NO_WINDOW)
time.sleep(3)
print("Launched.")
