"""Verify weekly_klines + CSI 500 data availability"""
import sqlite3
conn = sqlite3.connect('.eastmoney-ai/db/klines-v2.sqlite')

# Weekly
w = conn.execute("SELECT COUNT(*), COUNT(DISTINCT code), MIN(date), MAX(date) FROM weekly_klines").fetchone()
print(f"Weekly: {w[0]} rows, {w[1]} stocks, {w[2]}~{w[3]}")

w_hs = conn.execute("SELECT COUNT(DISTINCT code) FROM weekly_klines WHERE code IN (SELECT stock_code FROM stock_industry_mapping)").fetchone()[0]
print(f"Weekly HS300: {w_hs} stocks")

# Monthly
m_total = conn.execute("SELECT COUNT(DISTINCT code) FROM monthly_klines").fetchone()[0]
print(f"Monthly total: {m_total} stocks")

m_hs = conn.execute("SELECT COUNT(DISTINCT code) FROM monthly_klines WHERE code IN (SELECT stock_code FROM stock_industry_mapping)").fetchone()[0]
print(f"Monthly HS300: {m_hs} stocks")

conn.close()

# CSI 500
print("\nCSI 500 constituent lookup...")
try:
    import akshare as ak
    for name in ['index_stock_cons', 'index_stock_cons_csindex']:
        if hasattr(ak, name):
            try:
                df = getattr(ak, name)('000905')
                print(f"  {name}: {len(df)} stocks, cols={list(df.columns)[:5]}")
                break
            except Exception as e:
                print(f"  {name}: FAIL - {e}")
except Exception as e:
    print(f"  akshare import FAIL: {e}")
