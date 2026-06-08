# §11 PIT Market Cap / Turnover / Amihud — Design Spec

**Branch**: pead-koc-sue  
**Date**: 2026-06-08  
**Prerequisite**: §10 `eps_baostock_raw` complete (provides `total_share`, `pub_date`)  
**Output table**: `daily_kline` in `data/pead-baostock.sqlite`

---

## 目标

用 baostock 日线数据计算三个 PIT 控制变量：

| 变量 | 公式 | 用途 |
|------|------|------|
| `market_cap_pit` | `close(pub_date) × total_share(最近已披露季报)` | 市值中性化 |
| `turn_20d` | 前 20 交易日 `turn` 均值 | 流动性过滤（液态池） |
| `amihud_20d` | 前 20 交易日 `mean(|pctChg| / amount)` | 流动性中性化 |

---

## 数据来源

`baostock.query_history_k_data_plus`，调用参数：

```python
fields = "date,code,close,volume,amount,turn,pctChg"
frequency = "d"       # 日线
adjustflag = "3"      # 不复权（市值用原始价×总股本）
```

字段说明：
- `turn`：baostock 直接返回换手率（%），**不需要自己算**
- `pctChg`：涨跌幅（%），用于 Amihud，**不需要自己算**
- `close`：收盘价，乘以 `total_share` 得总市值

---

## 拉取策略

**每只股票一次查询**，覆盖全部需要窗口的并集：

```
start_date = min(pub_date for this stock) - 60 calendar days
end_date   = max(pub_date for this stock) + 90 calendar days
```

用自然日宽窗（-60/+90）保证跨长假（五一、国庆）后仍有足够交易日。
宽拉、窗口内过滤，不按 pub_date 逐季单独查。

---

## 窗口过滤（内存中执行）

拿到一只股票的完整日线后，对每个 `pub_date` 计算：

```python
# 升序排列（baostock 返回升序，但需显式校验）
assert df['date'].is_monotonic_increasing, f"{code}: date not sorted"

before = df[df['date'] < pub_date]   # pub_date 当日不含，取前置数据
after  = df[df['date'] >= pub_date]  # pub_date 当日及之后

# 换手率：前 20 交易日（按行数，不按日期差）
turn_rows = before.tail(20)
if len(turn_rows) < 20:
    turn_20d = None          # 标记 turn_insufficient，不用不足行数算
    turn_flag = 'insufficient'
else:
    turn_20d = turn_rows['turn'].mean()
    turn_flag = 'ok'

# Amihud：前 20 交易日
amihud_rows = before.tail(20)
if len(amihud_rows) < 20:
    amihud_20d = None
else:
    amihud_20d = (amihud_rows['pctChg'].abs() / amihud_rows['amount']).mean()

# 市值：取 pub_date 当日或之后最近一个交易日的 close
close_row = after.iloc[0] if len(after) > 0 else None
close_pit = float(close_row['close']) if close_row is not None else None
```

---

## PIT 市值 as-of join（`total_share`）

`total_share` 来自 §10 的 `eps_baostock_raw`，按披露日做 PIT join：

```sql
-- 每条 (code, date) 对应的 PIT 总股本
SELECT
    d.code,
    d.date,
    d.close * e.total_share  AS market_cap_pit
FROM daily_kline_snapshot d
JOIN eps_baostock_raw e
  ON  e.code     = d.code
  AND e.pub_date = (
        SELECT MAX(pub_date)
        FROM   eps_baostock_raw
        WHERE  code     = d.code
          AND  pub_date <= d.date      -- 只用已披露季报
      )
```

不能写 `JOIN eps_baostock_raw e ON e.code = d.code`（会产生笛卡尔积，一只股票约60个 `total_share` 行）。

Python 侧等价逻辑（按股票分组，避免重复 SQL 查询）：

```python
# eps 按 code 分组，已按 pub_date 升序
eps_by_code: dict[str, pd.DataFrame]  # {code: df[pub_date, total_share]}

def get_total_share_pit(code: str, target_date: str) -> float | None:
    eps = eps_by_code.get(code)
    if eps is None:
        return None
    mask = eps['pub_date'] <= target_date
    if not mask.any():
        return None
    return float(eps.loc[mask, 'total_share'].iloc[-1])  # 升序最后一行 = 最近一期

market_cap_pit = close_pit * get_total_share_pit(code, pub_date)
```

---

## 输出 Schema

```sql
CREATE TABLE daily_kline (
    code            TEXT    NOT NULL,
    date            TEXT    NOT NULL,   -- YYYY-MM-DD，pub_date 当日
    close           REAL,               -- 收盘价（不复权）
    turn_20d        REAL,               -- 前20交易日换手率均值（%），null=不足
    turn_flag       TEXT,               -- 'ok' | 'insufficient'
    amihud_20d      REAL,               -- 前20交易日 Amihud 均值，null=不足
    market_cap_pit  REAL,               -- close × total_share_pit（元），null=无季报
    PRIMARY KEY (code, date)
);
```

每行对应一个 `(code, pub_date)` 观测，不存全部交易日，只存财报披露日快照。

---

## 数据量估算

- 有效观测：~166K (code, pub_date) 对（pead.sqlite trusted=1）
- baostock 查询次数：~5,030 次（每只股票一次）
- 估计耗时：40ms 速率限制 × 5030 ≈ 200s + 网络开销 ≈ 30–60 分钟
- 输出行数：约 166K 行（一行一个披露日快照）

对比 §10：query_profit_data 按年查 ≈ 85K 次查询，耗时 6–10 小时。
**§11 查询次数是 §10 的 1/17，不是"比 §10 还久"。**

---

## 已知局限

1. **`total_share` 精度**：baostock `total_share` 来自季报，季内若有增发/回购，
   中间的市值会有偏差。对月频分析影响可忽略。

2. **节假日当日无交易**：若 `pub_date` 恰好是节假日（极少见），`after.iloc[0]`
   取的是下一个交易日收盘价，略有偏差（< 1 交易日）。

3. **`pctChg = 0` 导致 Amihud = 0**：停牌日 `pctChg=0, amount=0`，需在 Amihud
   计算前过滤 `amount <= 0` 的行，否则分母为零。

---

## 执行顺序

```
§10 eps_baostock_raw 完成
    ↓
§11 daily_kline 拉取（本文档）
    ↓
§12 reconcile（可能已有，验证两表 code 覆盖率）
    ↓
universe.sqlite 重建（01_universe.py --pit-mode）：
    size_missing=1 → 用 market_cap_pit 重新判 pass_mktcap
    pass_turnover=-1 → 用 turn_20d 重新判 pass_turnover
```
