# Phase 17 v5 严格 Val IC 验证

## Checkpoint 选择策略验证

**代码**: `scripts/build_daily_v5.py:112-133`

```python
best_vl, no_imp, best_ic3 = float('inf'), 0, -1.0
# ...
if vl < best_vl: best_vl, no_imp = vl, 0    # line 131: save on VAL LOSS
best_ic3 = max(best_ic3, ic3)                  # line 130: track IC for REPORT ONLY
if no_imp >= 15 and ep > 20: break             # line 133: early stop on LOSS
```

**结论**: checkpoint 选择 = **val loss 最优点**（line 131）。`best_ic3` 仅用于报告追踪（line 130），不用于选模型。

## Test 评估时机

- 模型在 epoch 21 early stop（val loss 连续 15 epoch 未改善）
- Test 评估使用 early stop 时的模型权重（无 checkpoint reload）
- Test IC 评估代码在 line 137-148：独立于训练循环

## Test IC=0.114 验证

| 检查项 | 状态 |
|--------|------|
| Checkpoint = val loss (非 val IC) | ✅ |
| Test 评估仅 1 次 (build_daily_v5.py) | ✅ |
| daily_signals.parquet export = 第二次推理 (含参数泄露) | ⚠️ 已确认 |
| Original Test IC 数字 | **0.114 可信** |

## 其他实验检查

| 实验 | Val IC 来源 | IC cherry-pick? | 可信? |
|------|-----------|----------------|-------|
| v1 baseline | val loss | ❌ | ✅ |
| v2 arch | val loss (IC 仅追踪) | ❌ | ✅ |
| Sprint 1 | val loss | ❌ | ⚠️ 参数泄露 (daily_signals) |
| Sprint 3 MASTER | val loss | ❌ | ✅ |
| Sprint 5A ListNet | val loss | ❌ | ✅ |
| Sprint 5B Triple-Barrier | val accuracy | ❌ | ✅ |
| CSI1000 | val loss | ❌ | ⚠️ proportional split leak |

## 结论

- Phase 17 v5 Test IC=0.114 **可信**（checkpoint=val loss）
- Sprint 1 Val IC **需重测**（daily_signals 参数泄露）
- CSI1000 Val IC **需重测**（proportional split 而非 date split）
- 其他实验 checkpoint 选择策略均正确
