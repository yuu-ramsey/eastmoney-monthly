"""Orchestrator: SQLite state machine, dispatch workers, 10h budget"""
import sqlite3, subprocess, sys, json, time
from pathlib import Path
from datetime import datetime, timedelta

PROJECT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT / '.eastmoney-ai' / 'lstm' / 'overnight.db'
PYTHON = sys.executable
WORKER = str(Path(__file__).parent / 'worker.py')
BUDGET_HOURS = 10.0
EXPERIMENT_TIMEOUT = 1800  # 30 min max per experiment

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(PROJECT / '.eastmoney-ai' / 'lstm' / 'logs' / 'orchestrator.log', 'a') as f:
        f.write(line + '\n')

def update_progress(conn):
    counts = {}
    for s in ['pending', 'running', 'done', 'failed']:
        counts[s] = conn.execute("SELECT COUNT(*) FROM experiments WHERE status=?", (s,)).fetchone()[0]
    total = sum(counts.values())
    with open(PROJECT / 'progress.json', 'w') as f:
        json.dump({'updated': datetime.now().isoformat(), 'counts': counts, 'total': total}, f, indent=2)
    return counts

def main():
    start_time = datetime.now()
    deadline = start_time + timedelta(hours=BUDGET_HOURS)
    log(f"Orchestrator starting. Budget: {BUDGET_HOURS}h. Deadline: {deadline.strftime('%H:%M')}")

    conn = sqlite3.connect(str(DB_PATH))

    while datetime.now() < deadline:
        # Get next pending experiment
        row = conn.execute(
            "SELECT id, phase, config_json FROM experiments WHERE status='pending' ORDER BY priority, id LIMIT 1"
        ).fetchone()

        if row is None:
            running = conn.execute("SELECT COUNT(*) FROM experiments WHERE status='running'").fetchone()[0]
            if running == 0:
                log("All experiments complete!")
                break
            log(f"No pending tasks, {running} still running. Waiting 30s...")
            time.sleep(30)
            continue

        exp_id, phase, config_json = row
        config = json.loads(config_json)
        model = config.get('model', '?')
        task = config.get('task', '?')

        log(f"Dispatching [{exp_id}] {phase}/{task}/{model}")

        try:
            proc = subprocess.run(
                [PYTHON, WORKER, str(exp_id)],
                timeout=EXPERIMENT_TIMEOUT,
                capture_output=True, text=True, cwd=str(PROJECT)
            )
            if proc.returncode != 0:
                log(f"  [{exp_id}] FAILED (exit {proc.returncode}): {proc.stderr[:200]}")
            else:
                row2 = conn.execute("SELECT ic_y3 FROM experiments WHERE id=?", (exp_id,)).fetchone()
                ic3 = row2[0] if row2 else None
                log(f"  [{exp_id}] OK: IC3={ic3:.4f}" if ic3 else f"  [{exp_id}] OK")
        except subprocess.TimeoutExpired:
            log(f"  [{exp_id}] TIMEOUT ({EXPERIMENT_TIMEOUT}s)")
            conn.execute("UPDATE experiments SET status='failed', error='timeout' WHERE id=?", (exp_id,))
            conn.commit()
        except Exception as e:
            log(f"  [{exp_id}] EXCEPTION: {e}")
            conn.execute("UPDATE experiments SET status='failed', error=? WHERE id=?", (str(e)[:500], exp_id))
            conn.commit()

        # Progress update every 10 completions
        done = conn.execute("SELECT COUNT(*) FROM experiments WHERE status IN ('done','failed')").fetchone()[0]
        if done % 10 == 0:
            counts = update_progress(conn)
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            log(f"Progress: {done}/{counts['total']} ({counts['done']} ok, {counts['failed']} fail) {elapsed:.1f}h")

    # Final summary
    counts = update_progress(conn)

    # Write final report
    top = conn.execute(
        "SELECT id, phase, config_json, ic_y3, ic_y6 FROM experiments WHERE status='done' AND ic_y3 IS NOT NULL ORDER BY ic_y3 DESC LIMIT 10"
    ).fetchall()

    report = [f"# Phase 17 v2 Overnight Report\n",
              f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
              f"Results: {counts['done']} ok / {counts['failed']} failed / {counts['pending']} pending\n\n",
              f"## Top 10 by IC y3\n"]
    for r in top:
        cfg = json.loads(r[2])
        report.append(f"- [{r[0]}] {r[1]}/{cfg.get('task')}/{cfg.get('model')}: IC3={r[3]:.4f} IC6={r[4]:.4f}\n")

    report.append(f"\n**Test NOT opened.**\n")
    with open(PROJECT / 'docs' / 'overnight_final_report.md', 'w') as f:
        f.writelines(report)

    log(f"Orchestrator done. {counts['done']} ok, {counts['failed']} fail")
    conn.close()

if __name__ == '__main__':
    main()
