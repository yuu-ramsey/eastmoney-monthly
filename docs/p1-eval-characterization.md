# P1 Eval 特征化审计：0.187 分数到底测什么

> 分支: p0a-verify-3-4 (只读审计) | 日期: 2026-05-29
> 
> **结论先行**：v4=0.187 是在 40 只股票、4 个时间点、640 个样本上，用 6 个月 alpha 阈值标签 + 加权命中率矩阵算出的分数。该分数略高于均匀随机（0.144），但**远低于"永远预测 neutral"（0.401）**——评分函数对 neutral 有结构性奖励。样本量太小（40 只/4 时点），误差棒覆盖大部分"提升"。未检测到代码级前视泄漏，但所有 prompt 迭代共享同一份 frozen dataset，存在过度调参风险。

---

## 1. Ground Truth 是什么？

### 定义：6 个月 alpha（超额收益）的离散化标签

**`lib/eval/dataset-builder.js:14-27`**:

```js
const GT_RULES = [
  { minAlpha: 10, label: 'strong_bull' },   // alpha >= 10%
  { minAlpha: 3, label: 'bull' },            // alpha >= 3%
  { minAlpha: -3, label: 'neutral' },        // alpha >= -3%
  { minAlpha: -10, label: 'bear' },          // alpha >= -10%
  { minAlpha: -Infinity, label: 'strong_bear' }, // alpha < -10%
];
```

### alpha 计算：stockReturn - CSI300 indexReturn

**`lib/eval/dataset-builder.js:122-148`**:

```js
const fromClose = klines[idx].close;
const toClose = klines[endIdx].close;
const stockReturn = +((toClose - fromClose) / fromClose * 100).toFixed(2);

// CSI300 同期
const iFrom = indexCache.klines[idxFrom].close;
const iTo = indexCache.klines[idxTo].close;
const indexReturn = +((iTo - iFrom) / iFrom * 100).toFixed(2);

const alpha = +(stockReturn - indexReturn).toFixed(2);
```

- **Horizon**: 6 个月（`evaluationHorizonMonths = 6`, `dataset-builder.js:46`）
- **价格**: close-to-close（月线收盘价）
- **复权**: 前复权（东财 API 默认）
- **停牌/涨跌停**: 未处理。停牌月 close 不变 → return≈0；涨跌停板封死时月线 close 反映极限价格
- **endIdx 选择**: 从 cutoff 之后找第一个 `date >= horizonEnd` 的 bar；找不到取最后一条 kline（`dataset-builder.js:116-121`）

### GT 分布（v4 eval 640 样本）

```
strong_bull: 180 (28.1%)  ← alpha >= 10%
strong_bear: 176 (27.5%)  ← alpha < -10%
bear:        104 (16.3%)  ← -10% <= alpha < -3%
neutral:      92 (14.4%)  ← -3% <= alpha < 3%
bull:         88 (13.8%)  ← 3% <= alpha < 10%
```

**真实命令输出**（node 计算 v4 jsonl）:
```
GT dist: {"bear":104,"bull":88,"strong_bull":180,"strong_bear":176,"neutral":92}
```

强标签占比 55.6%（strong_bull + strong_bear），neutral 仅 14.4%。GT 分布高度两极化。

---

## 2. Score 公式到底是什么？

### 核心加权得分 = sum(scores) / n

**`lib/eval/compute-score.js:13-24`**:

```js
export function scorePrediction(predictedSignal, groundTruth) {
  const p = mapSignal(predictedSignal);  // strong_bull=2, bull=1, neutral=0, bear=-1, strong_bear=-2
  const g = mapSignal(groundTruth);
  if (p === g) return 1.0;              // 完全命中
  if (p * g < 0) {                       // 方向相反
    return (Math.abs(p) === 2 && Math.abs(g) === 2) ? -1.0 : -0.5;  // 强反向=-1.0, 一般反向=-0.5
  }
  if (p === 0 || g === 0) return 0.3;   // 有一方 neutral → 0.3
  return 0.5;                            // 同方向不同强度
}
```

评分矩阵:

| pred ↓ / GT → | strong_bull | bull | neutral | bear | strong_bear |
|---------------|-------------|------|---------|------|-------------|
| strong_bull   | 1.0 | 0.5 | 0.3 | -0.5 | **-1.0** |
| bull          | 0.5 | 1.0 | 0.3 | -0.5 | -0.5 |
| neutral       | 0.3 | 0.3 | **1.0** | 0.3 | 0.3 |
| bear          | -0.5 | -0.5 | 0.3 | 1.0 | 0.5 |
| strong_bear   | **-1.0** | -0.5 | 0.3 | 0.5 | 1.0 |

### 加权得分计算

**`lib/eval/compute-score.js:31-47` (computeScoreTransparent)**:

- `full.weightedScore` = 所有记录的 score 之和 / 总记录数（含 parse_failed, score=0）
- `exclPf.weightedScore` = 排除 `signal === 'parse_failed'` 后的均值

v4 实测:
```
full.weightedScore = 107.4 / 640 = 0.1678
exclPf.weightedScore = 107.4 / 575 = 0.1868  (排除 65 条 parse_failed)
```

**PROGRESS.md 的 0.187 是 exclPf 值**（107.4/575），来自 `audit-v4-deep.js:140`。

### "bear bias 43%" 在公式里怎么体现？

不在公式里。bear bias 是 signal 分布的**诊断指标**（`report.js:202-216`），用于标记模型方向偏差，不参与 score 计算:

```js
if ((s === 'bear' || s === 'strong_bear') && diff > 0) bearBias += diff;
```

v4 实际: bear 203 + strong_bear 63 = 266/640 = **41.6%** 预测看空。

### 关键发现："永远 neutral"策略碾压模型

neutral 行在评分矩阵中**对所有 GT 都有保底 0.3**（除了 neutral→neutral 的完美 1.0）。实测:

```
Always neutral:    0.4006  ← 最优 null 策略
Always bear:       0.1337
Always bull:       0.1025
Always strong_bull: 0.0369
Uniform random:    0.1444
GT-weighted random: 0.1176
v4 model:          0.1678  ← 仅比 random 高 0.023
Frozen baseline:   0.1966  ← 已知最佳 prompt
```

**真实命令输出**:
```
=== Null baselines (n=640) ===
Always strong_bull: 23.6  weighted=0.0369
Always bull       : 65.6  weighted=0.1025
Always neutral    : 256.4  weighted=0.4006
Always bear       : 85.6  weighted=0.1337
Always strong_bear: 31.6  weighted=0.0494
Uniform random:   weighted=0.1444
```

0.187 既不是"弱到无意义"也不是"高到反常/泄漏"——它略高于随机基线，但被 neutral 保底分定义的上界（0.401）压制。模型做方向预测会承担 -0.5/-1.0 的惩罚，而"永远 neutral"零风险拿 0.3。

---

## 3. 样本量与时间范围

### Frozen Dataset v1

**`lib/eval/load-frozen-dataset.js` → `data/frozen-eval-dataset-v1.json`**

**真实命令输出**:
```
stocks: 40
testPoints: 160
version: frozen-v1
createdAt: 2026-05-18T13:35:49.837Z
uniqueMonths: 4
months: 2024-05,2024-11,2025-05,2025-11
```

### v4 eval 规模

**真实命令输出**:
```
total lines: 640          (160 testPoints × 4 templates)
valid: 640 | errors: 0
unique codes: 40
per stock: 16 (4 dates × 4 templates)
date range: 2024-05 ~ 2025-11
date list: 2024-05, 2024-11, 2025-05, 2025-11
```

### 统计效力评估

- **n = 640** 条 LLM 调用，但只有 **40 只股票 × 4 个时点 = 160 个独立的 (股票,时点) 对**
- 4 个 template 对同一样本重复评估 → 有效的独立样本量更接近 160 而非 640
- 40 只股票仅占 A 股的 ~1%，无行业/市值代表性
- 4 个时点全是 5 月或 11 月（半年间隔），覆盖 1.5 年
- **0.187 的误差棒**: 假设 n_eff≈160, score std≈0.5, 则 SE≈0.5/√160≈0.04, 95% CI≈±0.08
- 0.187 的 CI 覆盖 [0.11, 0.27]，无法区分"好于 random"和"与 random 无差异"

---

## 4. 训练/测试方法学泄漏检查

### 4a. 特征计算：无前视泄漏 ✓

**`lib/eval/runner.js:187-194`**（eval-mc-dropout.js 同样逻辑）:

```js
const cutoffKlines = allKlines.slice(0, tp.cutoffIndex + 1);
const closes = cutoffKlines.map(k => k.close);
const ma5 = computeMA(closes, 5);
const ma20 = computeMA(closes, 20);
const ma60 = computeMA(closes, 60);
const { dif, dea, hist } = computeMACD(closes);
```

所有指标仅在 `[0, cutoffIndex]` 区间计算，不涉及未来数据。MA/MACD 函数（compute-ma.js, compute-macd.js）为标准历史窗口计算。

### 4b. 数据切分：非 walk-forward，同一数据集迭代调参

- **不是** walk-forward / 时间外推 / rolling window
- 所有 2024-05 ~ 2025-11 的 4 个时间点**一次性固定**为 frozen dataset
- 所有 prompt 版本（baseline → v4 → v13 → phase15）在**同一数据集**上评估
- Frozen baseline 0.1966 是在该数据集上调出的第一个高分（`runA-no-sector-2026-05-18`）
- **风险**: prompt 工程本质上是"调超参用同一测试集"，0.187→0.197 的增益可能来自过拟合到 40 只股票/4 时点

### 4c. v4 eval 是否注入 LSTM 信号: 否 ✓

**真实命令输出**: 抽查 v4 prompt 开头:
```
LSTM in prompt: false
Prompt start: 你是 A 股技术分析师。以下是 000001(000001) 近 194 个月的月线数据...
```

v4 prompt 仅含 K 线表格 + MA/MACD，无 LSTM/ML 信号注入。

### 4d. LSTM MC Dropout eval 的 train/eval 重叠: 需要验证

`cli/eval-mc-dropout.js` 使用 `mc_dropout_signals.parquet`（预计算的 LSTM 预测 JSON），但该文件的训练时间窗未知。若 LSTM 训练用了 >=2024-05 的数据，则为前视泄漏。需要确认 LSTM 训练截止日期。

---

## 5. Null 基线对照

见第 2 节。完整 null 基线:

| 策略 | weightedScore |
|------|--------------|
| Always neutral | **0.4006** |
| Frozen baseline prompt | 0.1966 |
| v4 model (excl PF) | 0.1868 |
| v4 model (full) | 0.1678 |
| Uniform random | 0.1444 |
| Always bear | 0.1337 |
| GT-weighted random | 0.1176 |
| Always bull | 0.1025 |

**关键结论**: 0.187 仅比随机高 +0.023，比"永远 neutral"低 -0.214。该分数在统计上无法可靠地与随机区分（给定 n_eff≈160）。它不是"弱信号"也不是"泄漏"——它是在一个对 neutral 有结构性奖励的评分函数下、用极小样本测出的、存在过拟合风险的数字。

### 解释 0.187 需要的上下文

任何报告 0.187（或后续分数）时，必须同时报告:
1. 同期 null baseline（尤其是 always neutral = 0.401）
2. 样本量 n + 有效独立样本数
3. 是否与 frozen baseline（0.1966）共享同一数据集
4. 排除 PF 的样本数

---

## 未改动（审计约束）

| 模块 | 状态 |
|------|------|
| lib/eval/*.js | 只读 |
| lib/evaluation/*.js | 只读 |
| cli/eval*.js | 只读 |
| 任何 LLM 调用 | 未触发 |
| prompt 模板 | 未碰 |

## 审计后建议（不改代码，供人决策）

1. **优先换评分函数**: 移除 neutral 的结构性优势（0.3 保底），或改用 rank-based IC 与命中率双指标
2. **扩大样本**: 40 只股票 4 时点不足。目标 >=100 只、>=12 时点（覆盖牛熊周期）
3. **时间切分**: 留出一半时间点做 hold-out，不在上面调 prompt
4. **确认 LSTM 训练截止日**: 若 mc_dropout_signals.parquet 的训练数据覆盖了 eval 时点，mc-dropout eval 全部结果作废
