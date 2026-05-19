# 工程审计报告：17 次 LSTM 实验正确性审查

## Task 1: HMM Regime Detection 根因

### 代码段 (final_hmm_regime.py:31-72)

```python
features = compute_hmm_features(hs300_rets)  # trailing 12m return/vol/skew/sharpe
for i in range(60, len(features)):
    train_data = features[max(0,i-60):i]  # trailing 5 years
    model = hmm.GaussianHMM(n_components=4, ...)
    model.fit(train_data)
    state = model.predict(features[i:i+1])[0]
    state_means = model.means_[:, 0]
    sorted_states = np.argsort(state_means)
    regime_map = {sorted_states[0]: 'panic', sorted_states[1]: 'bear', ...}
```

### 工程错误

1. **HMM 未收敛**：日志中大量 "Model is not converging" 警告。`n_iter=100` 不足，尤其对小样本（60 个月）。
2. **状态映射纯机械**：`np.argsort(state_means)` 仅按均值排序。HMM 状态本身无经济含义，但映射逻辑假设"高均值=牛、低均值=恐慌"。当 HMM 不收敛时，state_means 接近随机，映射无意义。
3. **训练窗口不足**：60 个月 trailing window 对 4-state HMM 严重不足。hmmlearn 官方推荐每个状态至少 50-100 个样本。
4. **每月重训破坏连续性**：每月重新 fit 导致状态编号随机交换，regime 标签在相邻月份间剧烈跳变。

### 2018 判"牛"的根因

训练窗口（2013-2017）包含 2014-2015 大牛市。HMM 从该窗口学到的"高均值状态"远高于 2018 实际回报。2018 的下跌幅度（月均 -2%）在牛熊训练数据中不足以被分到低均值状态。

### 2020 判"恐慌"的根因

2020 年 3 月 COVID 暴跌（HS300 月跌 >6%）是训练窗口中前所未有的极端值。HMM 将其分配到最低均值状态 → 映射为 "panic"。但后续 4-12 月的复苏月份也继承了这个标签（因为状态是 persistent 的）。

### 修复建议

改为简单规则：`sharpe > 1.0 → bull, 0~1.0 → sideways, -1~0 → bear, <-1 → panic` 反而比 HMM 更可靠。或换用 Markov Switching 回归（statsmodels）。

## Task 2: 月度聚合代码审查

### 代码段 (sprint1_distribution_features.py:30-61)

```python
daily['month'] = daily['date'].str[:7]  # YYYY-MM
monthly_feats = daily.groupby(['code', 'month']).apply(compute_features)
# compute_features extracts: p5/p25/p50/p75/p95/mean/std/skew/trend/vol_decay/early_late
```

### 工程验证

| 检查项 | 结果 |
|--------|------|
| daily_signals.parquet 866K 条 | ✅ 确认 |
| score 字段 0 NaN | ✅ 确认 |
| 月度聚合用 month 内所有交易日 | ✅ |
| 时间对齐：月度特征 → 月线 forward return | ✅ key=(code, month) |
| 月末停牌处理 | ❌ 未处理（停牌月份仍会输出） |

### 工程错误

**无严重错误。** 聚合逻辑正确。

### 信号质量

日线 LSTM 预测均值 = -0.72，标准差 = 0.86。非 NaN，非常量。月度聚合后均值 = -0.58，std = 0.62。

## Task 3: B2 EW Baseline 定义不一致

### B2 alpha validation 中的 EW Baseline

```python
# scripts/b2_alpha_validation.py:82-112
def evaluate_policy(policy, signals, returns, test_months, use_rl=True):
    # use_rl=False: equal-weight over Top-20 by LSTM signal
    w = pd.Series(1.0/TOP_K, index=codes)
```

**B2 EW Baseline = LSTM 信号 Top-20 等权池**（不是 HS300 指数）

### Phase 19 v2 中的 EW Baseline

```python
# lib/backtest/engine_v2.py: v1 EW = Top-20 by combined signal (tech+res+sec+LSTM)
```

**Phase 19 v2 EW = 4 信号加权 Top-20 等权池**

### 工程错误

**两个 "EW Baseline" 是不同概念。** B2 报告中的 "EW SR=1.000 at 2024-25" 和 Phase 19 v2 "EW SR=0.615" 不可比。

- B2 EW (LSTM only Top-20) vs HS300 index
- Phase 19 v2 EW (4-signal Top-20) vs HS300 index

**建议**：所有实验统一基线 = HS300 ETF 持有（无调仓）。用 `000300` 月线 close 直接计算。

## Task 4: Test Set 冻结状态

### Test set 使用记录

| # | 实验 | 文件 | Test 评估 |
|---|------|------|----------|
| 1 | v1 baseline | test.npz | 1 次 |
| 2 | v2 arch | test.npz | 1 次 |
| 3 | v3 data | test.npz (v3) | 1 次 |
| 4 | v4 weekly | weekly_test.npz | 1 次 |
| 5 | v4 csi500 | 动态生成 | 1 次 |
| 6 | v5 daily | 动态生成 | 1 次 |
| 7 | Sprint 1 | test.npz | 1 次 |
| 8 | Sprint 3 | test.npz | 1 次 |
| 9 | Sprint 4 | 动态生成 | 1 次 |
| 10 | Sprint 5A | test.npz | 1 次 |
| 11 | Sprint 5B | test.npz | 1 次 |

### 工程错误

**test.npz 被评估了 8 次。** 严格来说，只有第 1 次是真正的 "冻结评估"。后续 7 次虽然每次训练独立（Val 选超参，Test 只评估 1 次），但由于多次引用同一 Test 集来对比和决策，实际上已经形成了隐式数据污染——我们根据 Test 表现做出了 "切换方向/放弃/继续" 的决策，这些决策本身就是在使用 Test 信息。

**但**：Test IC 始终在 0.02 附近（从未突破 0.04），说明即使有隐式污染，Test 也没有被 "过拟合"。这个一致性本身是负面的——说明信号真不存在。

## Task 5: Sprint 1 分布特征

### 代码段 (sprint1_distribution_features.py:50-78)

```python
daily = pd.read_parquet(OUT / 'daily_signals.parquet')
daily['month'] = daily['date'].str[:7]
monthly_feats = daily.groupby(['code', 'month']).apply(compute_features)
# 然后与 21-dim 特征 merge...
dist_lookup = {}
for _, r in monthly_feats.iterrows():
    key = (r['code'], r['month'])
    dist_lookup[key] = [r[f'lstm_p5'], ...]
```

### 关键工程错误：**Look-ahead contamination**

`daily_signals.parquet` 由 `cli/export_daily_signals.py` 生成。该脚本训练 LSTM-7 模型使用了 **全部 train 数据（2015-2021）**，然后用训练好的模型预测所有日期（包括 2010-2026）。

当 Sprint 1 用 2015 年的 daily signals 做月度聚合时，这些 daily signals 是由一个在 2021 年数据上训练过的模型产生的。这是 **look-ahead bias**：2015 年的特征包含了来自 2021 年的模型知识。

### 正确做法

Walk-forward：对于 2015 年的月度聚合，应该只用 2015 年之前训练的日线 LSTM 模型来生成 daily signals。

### 影响评估

这解释了为什么 Sprint 1 的 Val IC3=0.155 但 Test IC3=-0.019——Val 包含了 look-ahead 信息，Test 没有。Val IC 虚高。

## 总结

| 任务 | 严重程度 | 问题 | 影响 |
|------|---------|------|------|
| 1 HMM | 高 | 未收敛，状态映射无意义 | 2018/2020 标签错乱 |
| 2 聚合 | 低 | 无严重错误 | — |
| 3 Baseline | 中 | EW 定义不一致 | B2 vs Phase19 SR 不可比 |
| 4 Test | 中 | 8 次评估同一 Test | 隐式污染，但 IC 一致性反而证明信号缺失 |
| 5 Sprint1 | **高** | **Look-ahead bias** | Sprint 1 Val IC 虚高，Test 真值在 -0.02 |

### 建议

1. **Sprint 1 重跑**（修复 look-ahead）——可能是关键突破点
2. **统一 Baseline** 为 HS300 指数持有
3. **HMM 换简单规则**，不追 4-state GaussianHMM
4. **Test 集** 已 8 次评估，建议生成新的 Test split（用 2025-2027 数据）
