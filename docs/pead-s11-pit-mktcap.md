# §11 补丁规范：PIT 市值 as-of join（防笛卡尔积 + 时点错配）

**适用**：§11 日线脚本里 `close × total_share = 市值` 这一步的正确实现。  
**其余 §11 设计**（字段/窗口/全量拉窗口存）不变，本文件只规定市值计算步骤。  
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

市值定义：
```
market_cap(code, date) = close(code, date) × total_share(code, 最近一个 pub_date ≤ date)
```

即：某只股票在某交易日的市值 = 当日收盘价 × 「该日之前、已经披露的最近一期财报的总股本」。

关键：
- 用 **pub_date（披露日）** 做时间锚，不是 stat_date（财报截止日），更不是 fiscal_quarter
- 为什么用 pub_date：股本变更是在财报披露时才公开知道的，用 pub_date 才是 point-in-time
  （用 stat_date 会前视——财报截止日时市场还不知道新股本）
- 条件是 pub_date ≤ date（只用已披露的，不用未来的）

---

## 3. 实现（两种方式，任选）

### 方式A：SQL 关联子查询
```sql
SELECT
    d.code,
    d.date,
    d.close,
    d.close * (
        SELECT e.total_share
        FROM eps_baostock_raw e
        WHERE e.code = d.code
          AND e.pub_date <= d.date          -- 只用已披露的
          AND e.total_share IS NOT NULL
        ORDER BY e.pub_date DESC             -- 最近的一期
        LIMIT 1
    ) AS market_cap
FROM daily_kline d
```

### 方式B：Python 端 merge_asof（推荐，pandas 高效）
```python
import pandas as pd
# daily: 日线 (code, date, close, ...)
# shares: 股本 (code, pub_date, total_share)，按 pub_date 排序
daily = daily.sort_values('date')
shares = shares.sort_values('pub_date')
# 按 code 分组做 as-of join，date 匹配 ≤ 它的最近 pub_date
merged = pd.merge_asof(
    daily,
    shares.rename(columns={'pub_date': 'date'}),  # 对齐列名给 merge_asof
    on='date',
    by='code',
    direction='backward'        # 只取 ≤ 当前 date 的最近一条（不用未来）
)
merged['market_cap'] = merged['close'] * merged['total_share']
```

`direction='backward'` 是关键：只匹配 date 之前的最近股本，不用未来 = 无前视。

---

## 4. 边界情况

```
1. 日线 date 早于该股第一个 pub_date（上市初期还没第一份财报）
   → total_share 无匹配 → market_cap = NaN
   → 这些行标记 mktcap_missing，universe filter 里当 DATA_PENDING/剔除，不放行

2. total_share 为 null 的财季（个别缺）
   → as-of 时跳过 null，找再往前一个非 null 的
   → 方式A 已加 total_share IS NOT NULL；方式B 需先 dropna(shares)

3. 复权问题
   → close 用后复权还是不复权？市值用【不复权收盘价 × 总股本】才是真市值
   → baostock query_history_k_data 的 adjustflag：市值用 adjustflag='3'(不复权)
   → 但持有期收益要用复权价 → 收益和市值可能要两套 close，或单独处理
   → 提醒：市值算用不复权 close，收益算用复权 close，别混
```

---

## 5. 验证（写完必做）

```
1. 笛卡尔积检查：join 后行数应 ≈ 日线行数（~1500万），
   不是日线×60。若爆炸到上亿行 → as-of 没生效，退回了普通 join

2. 市值合理性抽查：
   - 贵州茅台某日市值应在万亿量级
   - 抽 3-5 只已知股票某日市值，对照公开数据（同花顺/东财）量级

3. PIT 正确性：
   - 找一只增发过的股票，增发 pub_date 前后的 total_share 应跳变
   - 市值在增发披露日后才用新股本，之前用旧股本
   - assert 某 date 用的 total_share，其 pub_date ≤ date
```

---

## 6. 硬性纪律

1. **必须 as-of join**，pub_date ≤ date，direction=backward，禁止简单 code join。
2. 时间锚用 **pub_date**（披露日），不是 stat_date（防前视）。
3. 市值用**不复权** close × total_share；收益用复权 close，两者别混。
4. 无匹配股本的日线标记 mktcap_missing，不放行。
5. 写完跑§5验证（笛卡尔积/市值量级/PIT跳变三项）。
