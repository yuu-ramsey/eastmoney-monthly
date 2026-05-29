# P1 Eval 重建设计：低位子集 + 双尺子 + hold-out

> 分支: `p1-eval-rebuild` | 日期: 2026-05-29
>
> UNIVERSE = **low_position**
>
> 背景审计: `docs/p1-eval-characterization.md`（评分矩阵病态）、`docs/p1-power-analysis.md`（MDE=31.5%）
>
> **设计目标**：退役手工评分矩阵，在 point-in-time 低位子集上用两把独立尺子评估 LLM 预测质量，hold-out 时点永不用于调 prompt。胜负手：LLM 能否跑赢"越跌越买"反转因子。

---

## 0. 退役评分矩阵

`lib/eval/compute-score.js` 的 `scorePrediction()`:
- 保留代码不删，顶部加注释标注已知缺陷
- 不再作为主指标。后续 cli/eval-*.js 改用新双尺子

```js
// ⚠ DEPRECATED as primary metric — see audit docs:
//   docs/p1-eval-characterization.md (neutral 0.3 floor, always-neutral=0.401)
//   docs/p1-conditional-signal.md   (spread 7.5% not significant, score matrix noise)
//   docs/p1-power-analysis.md       (MDE=31.5%, 34 sb pairs cannot detect effect)
```

---

## 1. 低位口径定义

### 主定义（一次定死）

```
N = 12  (回看月数)
X = 20% (区间底部阈值)

条件1: close_position_12m = (close − min(close_t-12 .. close_t-1)) / (max_t-12..t-1 − min_t-12..t-1) ≤ 0.20
条件2: close < MA60 (60月移动平均, 含当月close)

入选 = 条件1 AND 条件2
```

- **条件 1**: close 处于过去 12 个月区间的底部 20% → 近期持续下跌/回踩支撑
- **条件 2**: 价格低于长期均线 → 中长线也在低位
- 回看窗口和 MA 的计算**仅用 cutoff 前数据**（point-in-time）

### 敏感性分析（另跑）

固定条件 2 (`< MA60`)，对 X ∈ {10%, 20%, 30%} 各跑一次，看结论是否稳健。

### 每时点入选数波动实测

**真实 DB 查询输出**（含 MA60 的 both 列）:
```
2018-12 (bear bottom):  110 stocks
2019-06 (recovery):      22 stocks
2020-03 (COVID):         84 stocks
2020-12 (bull):          29 stocks
2022-10 (bear):         139 stocks
2024-10 (strong bull):   12 stocks
```

**入选数波动 12~170，处理规则：**
1. 每时点**上限 140 只**（超过则随机抽 140，种子固定可复现）
2. 不足时不补，如实记录实际入选数
3. 报告每时点 n，标注"样本量不足月"（< 30 时）

---

## 2. 时点选择 & hold-out

### 时点（12 个，跨 4 种 regime，留 6m horizon）

| 时点 | Regime | 后 6 月方向 | Train/Test |
|------|--------|-----------|------------|
| 2018-06 | bear mid | ↓ (2018 H2 bear) | Train |
| 2018-12 | bear bottom | ↑ (2019 H1 recovery) | Train |
| 2019-06 | recovery mid | ↑ (2019 H2) | Train |
| 2020-03 | COVID crash | ↑ (V-recovery) | **Test** |
| 2020-09 | recovery | ↑ (2021 bull) | Train |
| 2021-06 | bull mid | ↑→↓ (top) | Train |
| 2021-12 | bull top | ↓ (2022 bear) | **Test** |
| 2022-06 | bear mid | ↓→ | Train |
| 2022-10 | bear bottom | ↑ (2023 H1) | **Test** |
| 2023-06 | sideways | ↓→ | **Test** |
| 2024-02 | recovery dip | ↑ (2024 H1) | **Test** |
| 2024-10 | bull | ? (2025 H1) | **Test** |

**Hold-out**: 6 个 Test 时点。覆盖 crash/reversal/bear-bottom/sideways/bull——确保 test 跨 regime。

**Train 时点（6 个）仅供 prompt 迭代。Test 时点（6 个）只在最终验收跑一次。**

### 估算样本量

| 指标 | Train | Test | Total |
|------|-------|------|-------|
| Unique pairs (~80/时点) | ~480 | ~480 | ~960 |
| Records (×4 templates) | ~1920 | ~1920 | ~3840 |
| Winsorize MDE (std≈20%, n_sb~96/侧) | ~8% | ~8% | — |
| LLM cost (DeepSeek ~¥0.01/call) | ¥19 | ¥19 | ¥38 |

vs 现状 160 pairs, MDE=31.5%：独立样本扩大 **6×**，MDE 从 31.5% 降到 **~8%**。

---

## 3. 停牌/涨跌停/退市 alpha 处理

### 退市

**必须包含**。Point-in-time 构建时该时点存在的股票即使后来退市也入选。
退市后 close=摘牌价（若无则 0），alpha 计算到最后一月。

### 停牌

停牌月 close 不变（东财用 last traded close 填充）。
连续停牌 > 3 月 → 标记 `suspended: true`，分析时单独报。

### 涨跌停封板

月线 close 反映真实板价，不做修正。
月内交易天数 < 5 → 标记 `thin_trading: true`。

---

## 4. 双尺子

所有指标按 **(股票,时点) 分块 bootstrap** 出 95% CI。

### 尺子 1: 稳健中心（Winsorize）

| 指标 | 定义 |
|------|------|
| alpha 中位数 | 桶内 alpha 的 median（对极端值天然稳健） |
| Winsorize 均值 | trim α < p1 或 > p99 至边界值，再算均值 |
| Winsorize spread | Winsorize bullish 均值 − Winsorize bearish 均值 |

**判读**: spread > 0 且 95% CI 不含 0 → 方向区分力。

### 尺子 2: 尾部捕获

| 指标 | 定义 |
|------|------|
| Top-20% 捕获率 | 真实 alpha top-20% 中被 bullish 预测的比例 |
| >20% 大牛命中率 | alpha > 20% 中被 strong_bull/bull 预测的占比 |
| Bullish 组合期望收益 | 买全部 bullish 预测 · 等权持有 6m · 真实 mean return |

**判读**: 捕获率 > 20% → 模型识别 alpha outliers。

---

## 5. Null + 反转因子基线

⚠ **不使用**动量因子 — 低位股动量一律为负，动量不是有效 null。

### 5a. 简单 null

| 基线 | 策略 |
|------|------|
| Always-bullish | 所有 pair 预测 bull |
| Always-neutral | 所有 pair 预测 neutral |
| Uniform random | 5 向均匀随机 × 1000 次 |

### 5b. 反转/超卖因子（LLM 要跑赢的目标）

每个 pair 用 cutoff 前数据计算（无前视）：

| 因子 | 公式 | 方向 |
|------|------|------|
| 短期反转 1m | trailing 1-month return | 越低越 bullish |
| 短期反转 3m | trailing 3-month return | 越低越 bullish |
| MA60 折扣 | (close − MA60) / MA60 | 越负越 bullish |
| RSI 超卖 | RSI(14) < 30 | binary bullish |

**复合反转分**: 等权 Z-score 平均 → 排序 → top 20% 标 bullish, bottom 20% 标 bearish。

### 5c. 关键判读

```
模型 spread > 反转因子 spread 且 CI 不含 0 → LLM 有独立 edge
模型 spread ≤ 反转因子 spread               → LLM 不增值，一行因子替代
```

---

## 6. 实现计划

### 阶段 A: 构建样本池（本次设计，实施另开分支）

文件: `lib/eval/dataset-builder-lowpos.js`

1. 从 SQLite 加载全量月线 + 行业映射
2. 对 12 个 cutoff 月份，point-in-time 计算低位筛选 + MA60
3. 对每 (stock, cutoff)，计算 6m forward alpha（含退市/停牌处理）
4. 记录每时点实际入选数
5. 输出: `data/frozen-eval-lowpos-v1.json`
6. 跑 X ∈ {10%, 20%, 30%} 敏感性

### 阶段 B: 新评分 + 基线（实施另开分支）

文件: `lib/eval/rulers.js`

1. 退役 `compute-score.js`（加注释）
2. 实现 `dualRulers(results)` + `blockBootstrap()`
3. 实现 `reversalFactor(klines, cutoffIdx)` — 纯 cutoff 前数据
4. 实现 `evalLowPosition(dataset, results) → report`

### 阶段 C: 跑 LLM + 出报告（实施另开分支）

1. Train 时点上建 baseline prompt
2. Test 时点验收（仅一次）
3. 对比反转因子基线
4. 产出 `docs/p1-eval-rebuild-results.md`

---

## 7. 不碰清单

| 模块 | 状态 |
|------|------|
| Agent prompt (bull/bear/predictor/judge) | 不碰 |
| LLM provider (anthropic/deepseek) | 不碰 |
| score-fusion | 不碰 |
| 结构化输出解析 | 不碰 |
| `lib/eval/compute-score.js` | 保留 + 弃用注释 |
| 低位口径定义的公式 | 本次敲定后不改 |
