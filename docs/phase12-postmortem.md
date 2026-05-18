# Phase 12: HS300 行业 Alpha — 事后总结

## 目标

通过注入 HS300 内部行业 Alpha（个股相对其行业基准的超额收益）到 LLM prompt 中，提升模型月度趋势判断的准确率。

## 实验设计

### 实验 1: v1 注入（弱约束）

- Prompt 差异：Run A (no sector) vs Run B (with sector alpha block + #12 约束)
- 640 样本（40 只 × 4 模板 × 4 cutoffDate）
- 模型：deepseek-chat, maxTokens=4000
- **结果**：Δ score = -0.0032（统计不显著）
  - Run A weighted score: 0.1966
  - Run B weighted score: 0.1934
  - strong_bull FP: 57.9% → 57.8%
  - parse_failed: 7 → 14

### 实验 2: v2 强约束（强化 #12）

- 强化 #12 约束：必须推理 alpha 等级（强势/中性/弱势），strong_bull 要求强势，strong_bear 要求弱势
- SectorAlphaBlock 等级化呈现（表格格式）
- 160 样本（10 只 × 4 模板 × 4 cutoffDate）
- **结果**：Δ score = -0.0306（变差）
  - Run A weighted score: 0.0806
  - Run B weighted score: 0.0500
  - strong_bull FP: 56.0% → 71.4%（恶化）
  - bear bias: 31.3% → 40.0%（看空偏差增大）

### 离线量化矩阵（零 LLM 成本）

- 298 只 HS300 股票 × 2018-2025 月度数据 = 88,511 观察
- 16 个组合（4 lookback × 4 holding）
- 指标：Spearman IC / Hit Rate / Long-Short Sharpe

**IC 矩阵**：

| lookback \ holding | 1m | 3m | 6m | 12m |
|---|---|---|---|---|
| 3m | -0.0075 | -0.0013 | -0.0121 | -0.0015 |
| 6m | -0.0008 | -0.0069 | -0.0103 | -0.0086 |
| 12m | +0.0075 | +0.0056 | +0.0004 | -0.0242 |
| 24m | -0.0017 | -0.0174 | -0.0409 | **-0.0738** |

**Hit Rate 矩阵**：

全部 47.8%-51.4%，无法区分随机。

**Long-Short Sharpe 矩阵**：

**全部为负**（-0.119 至 -0.017）。最优组合 sharpe=-0.017。

## 根因分析

1. **12 个月 lookback 的行业 Alpha 对 6 个月前瞻收益无预测力**：IC 在 [-0.01, +0.01] 范围内，与零无差异
2. **24 个月 lookback 存在微弱的反向预测**（IC=-0.074）：个股长期跑赢同行后倾向均值回归，但预测强度太低（IC < 0.10），无法作为 LLM 硬约束
3. **LLM 被强制使用弱信号做出错误判断**：v2 强约束实验中，LLM 确实遵守了 #12 指令（rawResponse 审计 10/10 明确引用 alpha 等级），但 alpha 方向与 ground truth 不一致的案例中，约束反而引入了系统性噪声
4. **同行业内个股 alpha 的时序自相关性极低**：行业 alpha 本质是一个高度均值回归的信号，适合反向交易策略而非趋势跟踪

## 最终结论

**Phase 12: 关闭。IC 矩阵 16 个 cell 全部 |IC| < 0.10，无法作为 LLM prompt 约束。**

## 副产品

- **IC 矩阵方法论**：可复用的行业 alpha 预测力评估框架（cli/analyze-sector-predictive-power.js）
- **24m lookback, 12m holding 最强信号**（IC=-0.07）：留待 Phase 17（LSTM 训练）作为特征备选
- **lib/sector/ 代码保留**：calSectorAlpha 函数仍可用于后置过滤或其他离线分析场景，prompt 注入路径关闭

## 教训

1. **先离线量化验证再做 prompt 注入**：Phase 12 本应先跑 IC 矩阵再决定是否注入 LLM。顺序反了导致浪费 ¥29 eval 成本（640 实验）+ ¥3.6（160 实验）
2. **约束措辞的强度不改变信号本身的信息量**：v1→v2 约束强化没有改善效果，因为信息不增
3. **IC < 0.10 的信号不适合直接喂给 LLM**：LLM 难以区分"有信息量的约束"和"噪声约束"，强约束低信噪比数据会导致系统性判断偏差
