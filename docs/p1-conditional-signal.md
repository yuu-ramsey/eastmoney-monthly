# P1 条件信号分析：不依赖评分矩阵的真 alpha 评估

> 分支: p0a-verify-3-4 | 日期: 2026-05-29
> 
> **结论先行：模型能识别极端强势股（strong_bull 桶 alpha=18-20% vs 无条件 6%），
> 但在区分一般 bullish vs bearish 上无显著能力（spread 95% CI 含 0）。
> 方向命中率 59% 仅比 always-bullish 高 8pp，统计上不可靠。
> 原评分 0.187→0.197 的"提升"在 alpha 空间完全不可见。**

---

## 分析方法

**`cli/eval-reanalyze-conditional.js`** — 完全不使用 `scorePrediction` 评分矩阵，
只用真实 6 个月 alpha（来自 `frozen-eval-dataset-v1.json`）评估预测信息量。

对比两个 eval:
- v4-signals (0.187 exclPF, 640 records)
- frozen-baseline (0.1966, 640 records)

---

## 1. 条件实现收益

### v4-signals

```
全样本 (n=640): alpha均值=6.07%  中位数=0.88%  alpha>0占比=51.2%

预测桶            n    alpha均值  alpha中位  alpha>0%   vs无条件Δ
────────────────────────────────────────────────────────────────
strong_bull      104    18.54%     6.19%    61.5% +   12.47%
bull             141     4.50%     2.41%    58.9%    -1.57%
neutral           64     1.17%    -4.22%    45.3%    -4.89%
bear             203     0.79%    -3.56%    39.9%    -5.27%
strong_bear       63     9.94%    -2.27%    49.2% +    3.87%
```

**真实命令输出**（`node cli/eval-reanalyze-conditional.js`）:
```
strong_bull      104    18.54%     6.19%    61.5% +   12.47%
bull             141     4.50%     2.41%    58.9%    -1.57%
neutral           64     1.17%    -4.22%    45.3%    -4.89%
bear             203     0.79%    -3.56%    39.9%    -5.27%
strong_bear       63     9.94%    -2.27%    49.2% +    3.87%
```

**判读**:
- strong_bull 桶: alpha=18.54%，远高于无条件 6.07%。✓ 模型能识别极端强势股
- bull 桶: alpha=4.50%，**低于**无条件均值。✗ "一般看多"不带来超额收益
- bear 桶: alpha=0.79%，**为正**。✗ "看空"的股票实际仍在涨
- **strong_bear 桶: alpha=9.94%，为正且高于无条件**。✗✗ 模型认为"最差"的股票反而涨得更好——这不是反向信号，是无区分力

### frozen-baseline

```
预测桶            n    alpha均值  alpha中位  alpha>0%   vs无条件Δ
────────────────────────────────────────────────────────────────
strong_bull      107    20.22%     7.24%    65.4% +   14.16%
bull             150     3.44%     1.36%    58.0%    -2.62%
neutral           72     4.49%    -3.74%    48.6%    -1.57%
bear             250     1.55%    -3.56%    40.8%    -4.52%
strong_bear       54     7.24%     1.77%    53.7% +    1.18%
```

同样 pattern: strong_bull 一骑绝尘，其余桶乱序。

---

## 2. 方向区分度 (spread)

| 指标 | v4 (0.187) | baseline (0.197) |
|------|-----------|-----------------|
| bullish alpha均值 | 10.46% | 10.43% |
| bearish alpha均值 | 2.96% | 2.56% |
| **spread** | **7.50%** | **7.87%** |
| **95% CI (block bootstrap)** | **[-2.16%, +19.77%]** | **[-1.67%, +19.45%]** |
| n_eff (blocks) | 160 | 160 |

**真实命令输出**:
```
v4: spread 95% CI: [-2.16%, 19.77%], n_eff ≈ 160
bl: spread 95% CI: [-1.67%, 19.45%], n_eff ≈ 160
→ ✗ CI 含 0 → 无法拒绝 spread=0，预测无显著方向区分力
```

**spread 7.5% 看起来很大，但 CI 宽达 ±11%。** 这是 40 只股票 × 4 时点 = 160 块的直接后果——块内 4 个 template 高度相关，有效样本量不足以区分信号和噪声。

v4 (0.187) 和 baseline (0.197) 的 spread 几乎一样（7.50% vs 7.87%），0.01 的评分差异在 alpha 空间**完全不可见**。

---

## 3. 方向命中率

| 指标 | v4 (0.187) | baseline (0.197) |
|------|-----------|-----------------|
| 方向性预测 n | 511 | 561 |
| 命中 n | 301 | 330 |
| **命中率** | **58.9%** | **58.8%** |
| Always-bullish 对照 | 50.7% | 51.3% |
| **超额** | **+8.2pp** | **+7.5pp** |

分桶（v4）:
```
strong_bull  n=104  命中=64 (61.5%)  ← 确实偏高
bull         n=141  命中=83 (58.9%)  ← 接近先验
bear         n=203  命中=122 (60.1%) ← 但 bear 需要 alpha<0 才算"命中"
strong_bear  n= 63  命中=32 (50.8%)  ← 接近抛硬币
```

注意 "命中" 对 bear/strong_bear 的定义是 alpha < 0。bear 桶的 alpha 均值是 +0.79%（实际在涨），但 60.1% 的 bear 预测对应 alpha < 0 → 这是"停止的钟一天对两次"——因为全样本 49% alpha < 0 的先验就已经很高。

---

## 4. 与原评分矩阵的对比

| 维度 | 原评分矩阵 | 真 alpha |
|------|----------|---------|
| Always neutral | 0.401（最强策略） | 1.17%（低于无条件均值） |
| v4 vs baseline 差异 | 0.187 vs 0.197 (+5.3%) | spread 7.50 vs 7.87（几乎相同） |
| 统计显著性 | 未报告 | ✗ 不显著 |
| 能区分 bull vs bear | 矩阵保证有分（0.5） | 实际不能 |

原评分矩阵存在结构性缺陷（0.3 保底分），且把 neutral 变成最优策略。真 alpha 分析显示"预判 bull"和"预判 bear"的实际收益差异宽得不显著。

---

## 5. 总结

1. **模型信息量集中在 strong_bull 桶** — 约 100/640 条预测能选出 alpha~19% 的股票，但占不到 1/6
2. **bull / bear / neutral / strong_bear 四桶之间无显著区分** — 排序乱、CI 重合
3. **spread 7.5% 在 n_eff=160 时统计不显著** — 需要 ~4× 样本（≥40 stocks × ≥20 timepoints）才能把 CI 压到不含 0
4. **0.187 vs 0.197 在 alpha 空间是 no-op** — 两个 prompt 版本的预测质量完全相同。评分矩阵的微小差异来自 parse_failed 比例和矩阵噪声，不是预测能力提升

**对下一步的建议**: 在扩展样本（≥100 stocks, ≥12 timepoints）之前，任何 <0.02 的评分差异不应被解释为"提升"。优先做样本扩展而非继续调 prompt。

---

## 脚本

`cli/eval-reanalyze-conditional.js` — 独立 CLI，用法:
```
node cli/eval-reanalyze-conditional.js
```
不调 LLM，不写文件，仅读取已有 jsonl + frozen dataset 做纯数值统计。

## 未改动

| 模块 | 状态 |
|------|------|
| lib/eval/compute-score.js | 未碰 |
| 任何 LLM 调用 | 未触发 |
| frozen dataset | 只读 |
