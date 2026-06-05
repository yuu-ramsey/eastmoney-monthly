"""将 mc_dropout_signals.parquet 导出为 per-stock JSON 文件
供 Chrome 扩展 native-host 读取。
输出位置：.eastmoney-ai/storage/mc_dropout/<code>.json
"""
import pandas as pd, json
from pathlib import Path

PROJECT = Path(__file__).parent.parent
PARQUET_PATH = PROJECT / '.eastmoney-ai' / 'lstm' / 'mc_dropout_signals.parquet'
OUT_DIR = PROJECT / '.eastmoney-ai' / 'storage' / 'mc_dropout'

def main():
    if not PARQUET_PATH.exists():
        print(f"文件不存在: {PARQUET_PATH}")
        print("请先运行: python cli/mc_dropout_predict.py --all --latest")
        return

    df = pd.read_parquet(PARQUET_PATH)
    print(f"加载 {len(df)} 条记录")

    # 每个股票取最新一条
    latest = df.sort_values('date').groupby('code').last()
    print(f"覆盖 {len(latest)} 只股票")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exported = 0
    for code, row in latest.iterrows():
        data = {
            'code': code,
            'date': str(row['date']),
            'y3_mean': float(row['y3_mean']),
            'y3_std': float(row['y3_std']),
            'y3_adjusted': float(row['y3_adjusted']),
            'y3_confidence': float(row['y3_confidence']),
            'y6_mean': float(row['y6_mean']),
            'y6_std': float(row['y6_std']),
            'y6_adjusted': float(row['y6_adjusted']),
            'y6_confidence': float(row['y6_confidence']),
            'overall_confidence': float(row['overall_confidence']),
            'uncertainty_level': str(row['uncertainty_level']),
            'signal': float(row['signal']),
            'signal_raw': float(row['signal_raw']),
        }
        out_path = OUT_DIR / f'{code}.json'
        out_path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        exported += 1

    print(f"导出完成: {exported} 文件 → {OUT_DIR}")

if __name__ == '__main__':
    main()
