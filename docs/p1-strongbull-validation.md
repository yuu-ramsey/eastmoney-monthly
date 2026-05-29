# P1 strong_bull 验证：LLM 唯一 edge vs 朴素动量基线

> 分支: p0a-verify-3-4 | 日期: 2026-05-29
>
> **判读: C — strong_bull 自身不显著。unique-pair 层级 CI 下界 (5.02%) < 无条件均值 (6.07%)。
> 34 个独立样本的误差棒覆盖一切。下任何结论前必须先扩样本。**

---

## 前置：40 只股票如何选出

**`scripts/build-frozen-dataset.js:1-10`** — frozen dataset 从 runA eval JSONL 提取：

```js
stocks.set(r.stockCode, { category: r.category || 'hs300' });
```

**`lib/eval/seed-stocks.json`** — 种子股票 6 类形态各 5-8 只，上市 ≥5 年。用户手工挑选。

40 只股票来自手工选的种子池 + 数据可用性过滤（≥24 个月 K 线）。**幸存者偏差**明显：能活到 2024 年且有 15+ 年连续月线的股票本身偏强（无条件 alpha 均值 6.07%/6 个月）。

---

## 方法

**`cli/eval-strongbull-vs-momentum.js`** — 在 unique (股票, 时点) 粒度上分析：

1. 对每对 (stock, cutoffDate)，用 cutoff 前的月线数据计算朴素动量（trailing 6m/12m 收益 + MA60 偏离，Z-score 等权复合）
2. 取与 strong_bull 同数量的 top 动量对
3. 所有 CI 按 block bootstrap（block = (stock, cutoffDate)）
4. 不重跑 LLM

---

## 1. strong_bull 自身显著性

| 指标 | 值 |
|------|-----|
| 全样本 unique pairs | 160 |
| strong_bull unique pairs | 34 (21.3%) |
| strong_bull alpha 均值 | 20.13% |
| 无条件 alpha 均值 | 6.07% |
| **95% CI (block bootstrap)** | **[5.02%, 41.06%]** |

**真实命令输出**:
```
alpha 均值: 20.13% (无条件: 6.07%)
95% CI: [5.02%, 41.06%]
→ CI含/低于无条件 → 不显著 ✗
```

CI 下界 5.02% < 无条件 6.07%。即使 strong_bull 均值高达 20%，只有 34 个独立样本时，无法拒绝 null。

---

## 2. 动量基线

| 指标 | 值 |
|------|-----|
| 动量 top-34 alpha 均值 | 12.72% |
| **95% CI** | **[4.32%, 22.58%]** |

**真实命令输出**:
```
alpha 均值: 12.72%
95% CI: [4.32%, 22.58%]
```

动量本身也不显著（CI 含无条件均值）。160 pairs 上取 top-34，任何排序指标的 CI 都会很宽。

---

## 3. 重叠度

| 指标 | 值 |
|------|-----|
| LLM strong_bull | 34 |
| 动量 top-34 | 34 |
| 交集 | 16 |
| **Jaccard** | **0.308** |

**真实命令输出**: `LLM: 34  动量: 34  交集: 16  Jaccard: 0.308`

重叠不高。LLM 和动量只有 ~1/3 的共同选择。说明即使都不显著，它们选的股票也不同。

---

## 4. 差值显著性

| 指标 | 值 |
|------|-----|
| strong_bull alpha | 20.13% |
| 动量 alpha | 12.72% |
| 差值 (sb − mom) | +7.40% |
| **95% CI** | **[-7.01%, +27.52%]** |

**真实命令输出**: `差值: 7.40%, 95% CI: [-7.01%, 27.52%]`

差值 CI 跨零，跨度 34pp。无法说 strong_bull 显著优于动量。

---

## 判读

```
→ C) strong_bull 自身不显著 → 先扩样本再下结论
```

| 判读条件 | 结果 |
|----------|------|
| strong_bull CI 下界 > 无条件 6.07%? | **否** (5.02% < 6.07%) |
| 差值 CI > 0? | **否** (含 0) |

**不是 A 也不是 B。** strong_bull 桶在 unique-pair 层级没有统计显著的 alpha 优势。problem not in model — in sample.

---

## 脚本

`cli/eval-strongbull-vs-momentum.js`:
```
node cli/eval-strongbull-vs-momentum.js
```

## 未改动

| 模块 | 状态 |
|------|------|
| 任何 eval 逻辑 | 未碰 |
| prompt/LLM | 未碰 |
| frozen dataset | 只读 |
