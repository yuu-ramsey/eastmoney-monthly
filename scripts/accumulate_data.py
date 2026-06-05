"""Auto data accumulation — daily run, progressively accumulates MC Dropout history data
Usage:
  python scripts/accumulate_data.py              # Incremental: only accumulate today new data
  python scripts/accumulate_data.py --force-all  # Full rerun (first time or fix data)
  python scripts/accumulate_data.py --rebuild-v2 # Rebuild v2 dataset (when new monthly klines arrive)

Data flow:
  daily_klines (SQLite) → MC Dropout (today only) → mc_dropout_history.parquet (accumulated)
                                                      ↓
                                         mc_dropout_signals.parquet (latest, production use)
                                                      ↓
                                         mc_dropout/<code>.json (native-host)
                                                      ↓
                                         frozen-eval-dataset-v2.json (if new monthly klines available)
"""
import sys, os, argparse, json, time, shutil
from pathlib import Path
import numpy as np, pandas as pd, sqlite3

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
LSTM_DIR = PROJECT / '.eastmoney-ai' / 'lstm'
STORAGE_DIR = PROJECT / '.eastmoney-ai' / 'storage' / 'mc_dropout'
DATA_DIR = PROJECT / 'data'
PYTHON = os.environ.get('PYTHON_PATH', sys.executable)

# File paths
HISTORY_PARQUET = LSTM_DIR / 'mc_dropout_history.parquet'
LATEST_PARQUET = LSTM_DIR / 'mc_dropout_signals.parquet'
MC_PREDICT_SCRIPT = PROJECT / 'cli' / 'mc_dropout_predict.py'

def log(msg):
    print(f"[accumulate] {msg}")

# ======== 1. MC Dropout Incremental Accumulation ========
def run_mc_dropout_today():
    """Run MC Dropout, get latest daily signals for all stocks"""
    log("Running MC Dropout --all --latest ...")
    t0 = time.time()

    # Save to temp file
    tmp_path = LSTM_DIR / 'mc_dropout_tmp.parquet'
    cmd = [
        PYTHON, str(MC_PREDICT_SCRIPT),
        '--all', '--latest',
    ]
    # Directly call script (outputs to mc_dropout_signals.parquet by default)
    import subprocess
    result = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True)
    if result.returncode != 0:
        log(f"MC Dropout failed:\n{result.stderr[-500:]}")
        return None

    elapsed = time.time() - t0
    log(f"MC Dropout done ({elapsed:.0f}s)")

    if not LATEST_PARQUET.exists():
        log("ERROR: Output file not found")
        return None

    latest_df = pd.read_parquet(LATEST_PARQUET)
    log(f"  New data: {len(latest_df)} stocks, dates={latest_df['date'].unique()[:3].tolist()}...")
    return latest_df

def accumulate_history(new_df):
    """将New data合并到历史 parquet，去重 (code, date)，保留latest值"""
    if HISTORY_PARQUET.exists():
        old_df = pd.read_parquet(HISTORY_PARQUET)
        log(f"  History data: {len(old_df)} rows, {old_df['code'].nunique()} stocks")
        # 合并 + 去重（New data覆盖旧数据）
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['code', 'date'], keep='last')
        combined = combined.sort_values(['code', 'date']).reset_index(drop=True)
    else:
        log("  无History data，创建新文件")
        combined = new_df

    combined.to_parquet(HISTORY_PARQUET, index=False)
    log(f"  Saved history: {len(combined)} rows, {combined['code'].nunique()} stocks, "
        f"Date range: {combined['date'].min()} ~ {combined['date'].max()}")

    # Monthly statistics
    combined['month'] = combined['date'].str[:7]
    monthly_counts = combined.groupby('month').size()
    log(f"  Months covered: {len(monthly_counts)}, earliest: {monthly_counts.index[0]}, latest: {monthly_counts.index[-1]}")

    return combined

def export_native_host_json(latest_df):
    """导出每个股票的latest MC Dropout JSON 供 native-host 读取"""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    exported = 0
    for code, group in latest_df.groupby('code'):
        # 取latest日期
        row = group.sort_values('date').iloc[-1]
        ulevel = str(row.get('uncertainty_level', 'medium'))
        data = {
            'code': code,
            'date': str(row['date']),
            'signal': float(row.get('signal', 0)),
            'signal_raw': float(row.get('signal_raw', 0)),
            'y3_mean': float(row.get('y3_mean', 0)),
            'y3_std': float(row.get('y3_std', 0)),
            'y6_mean': float(row.get('y6_mean', 0)),
            'y6_std': float(row.get('y6_std', 0)),
            'overall_confidence': float(row.get('overall_confidence', 0)),
            'uncertainty_level': ulevel,
        }
        with open(STORAGE_DIR / f'{code}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        exported += 1
    log(f"  Exported native-host JSON: {exported} stocks → {STORAGE_DIR}")

# ======== 2. v2 Dataset 更新 ========
def check_new_monthly_data():
    """Check if new monthly data exists (since last v2 dataset build)"""
    v2_path = DATA_DIR / 'frozen-eval-dataset-v2.json'
    if not v2_path.exists():
        log("v2 dataset does not exist, need full rebuild")
        return True

    v2 = json.loads(open(v2_path, encoding='utf-8').read())
    existing_cutoffs = set(tp['cutoffDate'] for tp in v2.get('testPoints', []))

    conn = sqlite3.connect(str(DB))
    # 检查是否有比现有最大 cutoffDate 更新的月线
    max_existing = max(existing_cutoffs) if existing_cutoffs else '2010-01'
    new_months = conn.execute(
        "SELECT DISTINCT substr(date,1,7) FROM monthly_klines WHERE date > ?",
        (max_existing,)
    ).fetchall()
    conn.close()

    if new_months:
        log(f"Found new months: {[m[0] for m in new_months[:5]]}... (共 {len(new_months)} 个月)")
        return True
    log(f"无新月份（latest: {max_existing}）")
    return False

# ======== Main flow ========
def main():
    parser = argparse.ArgumentParser(description='Data Auto Accumulation')
    parser.add_argument('--force-all', action='store_true', help='Full rerun MC Dropout')
    parser.add_argument('--rebuild-v2', action='store_true', help='Force rebuild v2 dataset')
    parser.add_argument('--skip-mc', action='store_true', help='Skip MC Dropout, only update dataset')
    args = parser.parse_args()

    print("=" * 60)
    print("Data Auto Accumulation")
    print("=" * 60)

    # ---- MC Dropout ----
    if not args.skip_mc:
        if args.force_all:
            log("Full mode: rerun all historical MC Dropout")
            # 重命名旧文件作为备份
            if HISTORY_PARQUET.exists():
                bak = HISTORY_PARQUET.with_suffix('.parquet.bak')
                shutil.move(str(HISTORY_PARQUET), str(bak))
                log(f"  Backup old history: {bak}")

            import subprocess
            result = subprocess.run(
                [PYTHON, str(MC_PREDICT_SCRIPT), '--all'],
                cwd=str(PROJECT), capture_output=True, text=True,
            )
            if result.returncode != 0:
                log(f"全量 MC Dropout failed:\n{result.stderr[-500:]}")
                return
            # 将输出文件作为历史起点
            if LATEST_PARQUET.exists():
                full_df = pd.read_parquet(LATEST_PARQUET)
                full_df.to_parquet(HISTORY_PARQUET, index=False)
                log(f"  Full history saved: {len(full_df)} rows")
        else:
            # Incremental mode: run today only
            new_df = run_mc_dropout_today()
            if new_df is not None:
                accumulate_history(new_df)

        # 导出 production JSON
        if LATEST_PARQUET.exists():
            export_native_host_json(pd.read_parquet(LATEST_PARQUET))

    # ---- v2 Dataset ----
    need_rebuild = args.rebuild_v2 or check_new_monthly_data()
    if need_rebuild:
        log("Rebuilding v2 dataset...")
        import subprocess
        result = subprocess.run(
            ['node', 'scripts/build-frozen-dataset-v2.js', '--match-v1'],
            cwd=str(PROJECT), capture_output=True, text=True,
        )
        if result.returncode == 0:
            log("v2 dataset 重建done")
            # 打印最后几行输出
            for line in result.stdout.strip().split('\n')[-8:]:
                print(f"  {line}")
        else:
            log(f"v2 重建failed:\n{result.stderr[-300:]}")

    # ---- 摘要 ----
    print(f"\n{'='*60}")
    print("积累done")
    if HISTORY_PARQUET.exists():
        h = pd.read_parquet(HISTORY_PARQUET)
        months = sorted(h['date'].str[:7].unique())
        print(f"  MC Dropout 历史: {len(h)} rows, {h['code'].nunique()} stocks, "
              f"{len(months)} 个月 ({months[0]} ~ {months[-1]})")
    v2_path = DATA_DIR / 'frozen-eval-dataset-v2.json'
    if v2_path.exists():
        v2 = json.loads(open(v2_path, encoding='utf-8').read())
        print(f"  v2 Dataset: {len(v2['testPoints'])} testPoints, "
              f"{len(v2['stocks'])} stocks")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
