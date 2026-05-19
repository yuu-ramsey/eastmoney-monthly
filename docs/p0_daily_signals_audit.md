# P0: daily_signals.parquet Look-Ahead 审计

## 1. 模型训练数据范围

**代码**: `cli/export_daily_signals.py:100-106`

```python
# Split by PROPORTIONAL index (not by date)
n = len(X); n_tr = int(n*0.55); n_va = int(n*0.72)
Xtr, ytr = X[:n_tr], y[:n_tr]
Xva, yva = X[n_tr:n_va], y[n_tr:n_va]
Xte, yte = X[n_va:], y[n_va:]
```

- Train: 前 55% 序列 (~416K 条, 覆盖日期约 2015-2021)
- Val: 中间 17% (~129K 条, 覆盖日期约 2022-2023)  
- Test: 最后 28% (~212K 条, 覆盖日期约 2024-2026)
- Split 方式: **按索引比例** (序列按 stock_code 排序后切分), 不是按日期
- 最佳 checkpoint: val loss 最低的 epoch (early stop patience=15)

## 2. 推理数据范围

**代码**: `cli/export_daily_signals.py:156-194`

```python
for code in stocks:
    g = d_df[d_df['code'] == code].sort_values('date')
    # 重新计算特征...
    seqs = [feats[i-252+1:i+1] for i in range(251, n)]
    model.eval()
    with torch.no_grad():
        for j in range(0, len(seqs), 512):
            batch = torch.from_numpy(np.array(seqs[j:j+512])).float().to(DEVICE)
            p = model(batch).cpu().numpy()[:, 0]
```

- 推理范围: 所有股票, 从 `date >= '2010-01-01'` 开始的全部交易日
- 模型: 一个 final checkpoint (在 train+val 上早停选定)
- **无 walk-forward retrain**: 同一模型推理整个 2010-2026

## 3. 关键判断

### Case A: LOOK-AHEAD CONFIRMED

| 检查项 | 实际状态 | 泄露? |
|--------|---------|-------|
| 训练数据范围 | 55% 序列 (~2015-2021) | — |
| 推理数据范围 | **全部日期 (2010-2026)** | — |
| Train 期是否预测 | **是** (2015-2021 的 daily signals 由同模型产生) | ⚠️ |
| Walk-forward retrain | **否** | ⚠️ |
| 最终判定 | **Case A** | ⚠️ |

### 两层泄露

1. **特征层**: 每天的特征使用 trailing 252 日数据（无 forward look）✅
2. **参数层**: 模型参数从 2015-2021 训练中学到的模式, 用于预测 2015 年本身的数据 ⚠️

2015 年的 daily signals 来自一个 "见过 2021" 的模型。对于 2015 年的预测, 这不是严格的 out-of-sample。

### 但泄露程度有限

模型只在 train set (2015-2021) 上见过数据。2015 年的预测不是 "用 2021 年的特征预测 2015 年"——特征仍是 walk-forward。泄露仅来自参数层面: 模型学会了 2015-2021 年间的 cross-sectional 模式, 然后用于 2015 年。

对于下游 Sprint 1 的影响:
- Sprint 1 用 daily_signals 的 2015 年数据生成月度特征
- 这些月度特征包含的模型参数信息量有限（参数来自整个 train set 的平均模式）
- Sprint 1 的 Val IC=0.155 可能部分虚高, 但 Test IC=-0.019 是真值（Test 2024-2026 不在模型训练集中）

## 4. 影响范围

| 实验 | 依赖 daily_signals? | 影响 |
|------|-------------------|------|
| Phase 17 v5 Test IC=0.114 | 否(Test 独立评估) | 不受影响 ✅ |
| Sprint 1 33-dim | **是** | Val 虚高, Test 真值 -0.019 |
| Phase 19 v3 | **是**(LSTM信号) | 月度信号可能含轻微参数泄露 |
| B1 signal fusion | **是** | 同上 |
| B2 RL/B3 Risk | **是** | 同上 |

Phase 17 v5 的 Test IC=0.114 独立于 daily_signals.parquet——它在 `build_daily_v5.py` 中原位评估, 不依赖导出文件。

## 5. 修复方案

```python
# 正确做法: 逐年 retrain
for year in [2015, 2016, ..., 2025]:
    train_data = features[dates < f'{year}-01-01']
    model = train_model(train_data)
    predictions[year] = model.predict(features[dates year])
```

工时: 2-3 天 (需要 10 次 retrain × ~5 min = 50 min GPU)

## 6. 结论

**Case A 确认。** 但实际泄露程度中等（参数层, 非特征层）。最关键的影响: Sprint 1 Val IC 虚高。建议修后重跑 Sprint 1。
