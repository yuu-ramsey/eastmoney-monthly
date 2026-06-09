# §16 规范：PIT 市值 as-of join（防笛卡尔积 + 时点错配）

**脚本**：`scripts/koc/16_pit_marketcap.py`  
**前置**：`15_fetch_baostock_daily.py` 已完成（daily_kline + liqaShare 已入库）  
**记录日期**：2026-06-08

---

## 1. 问题：简单 code join 是错的

```sql
-- ❌ 错误写法
SELECT d.close * e.total_share AS market_cap
FROM daily_kline d
JOIN eps_baostock_raw e ON d.code = e.code
```

两个错：
1. 一只股票在 eps_baostock_raw 有 ~60 行（60 个财季的 total_share）
   → 每条日线 join 出 60 行 → 笛卡尔积爆炸
2. 即使去重，也不知道该用哪个财季的股本
   → total_share 会变（增发后变大、回购后变小）
   → 用错时点的股本 = 市值算错 = 前视/错配

---

## 2. 正确：as-of join（用 pub_date 做时间锚）

```
market_cap(code, date) = close(code, date) × total_share(code, 最近一个 pub_date ≤ date)
float_mktcap(code, date) = close(code, date) × liqaShare(code, 最近一个 pub_date ≤ date)
```

- 时间锚用 **pub_date（披露日）**，不是 stat_date（防前视）
- 条件是 pub_date ≤ date（只用已披露的，不用未来的）
- 无匹配股本 → 标记 `mktcap_missing=1`，不放行进 universe

---

## 3. 实现：pd.merge_asof（推荐）

```python
import pandas as pd

daily = daily.sort_values('date')
shares = shares[shares['total_share'].notna()].sort_values('pub_date')

merged = pd.merge_asof(
    daily,
    shares.rename(columns={'pub_date': 'date'})[['code','date','total_share','liqaShare']],
    on='date',
    by='code',
    direction='backward',   # 只取 ≤ 当前 date 的最近一条，不用未来
)
merged['market_cap_yi']   = merged['close'] * merged['total_share'] / 100   # 亿元
merged['float_mktcap_yi'] = merged['close'] * merged['liqaShare']  / 100   # 亿元
merged['mktcap_missing']  = merged['total_share'].isna().astype(int)
```

单位：total_share/liqaShare 是 万股；close 是 元/股；
market_cap 万元 = close × total_share；÷100 得亿元。

---

## 4. 边界情况

| 情况 | 处理 |
|------|------|
| 上市初期 date < 第一个 pub_date | total_share = NaN → mktcap_missing=1 |
| total_share = null 的财季 | merge_asof 前 dropna(shares) 跳过，找再往前一个非 null |
| 复权问题 | close 用不复权（adjustflag='3'）；收益用复权 close，两者不混 |

---

## 5. 验证（写完必做）

1. **笛卡尔积检查**：join 后行数应 ≈ daily_kline 行数，不是 ×60
2. **市值量级抽查**：贵州茅台某日市值应在万亿量级（market_cap_yi ≈ 20000亿）
3. **PIT 跳变验证**：找一只增发过的股票，增发 pub_date 前后 total_share 应跳变

---

## 6. 硬性纪律

1. 必须 merge_asof(direction='backward')，禁止简单 code join
2. 时间锚用 pub_date（披露日），不是 stat_date
3. 市值用不复权 close；收益用复权 close，不混
4. 无匹配股本标记 mktcap_missing=1，不放行
5. 写完跑三项验证
