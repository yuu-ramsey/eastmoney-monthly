# P1 低位池 LLM 预注册 — Phase C0

> 分支: `p1-eval-lowpos-run` | 日期: 2026-05-29 | **跑 LLM 前 commit，跑后不得改**

## 主指标

**(LLM Winsorize spread − 动量 Winsorize spread) 的分块 bootstrap 差值 CI**

同批重采样块上算两侧 spread 再取差值，CI 来自差值分布。

## 判读规则（三分）

| 差值 CI | 结论 |
|---------|------|
| 下界 > 0 | LLM 在动量之上有增量 edge（保守可信） |
| 含 0 | **inconclusive**（默认预期——Test MDE=31.7%，功效不足；非阴性结论） |
| 上界 < 0 | LLM 劣于动量 |

## 参照系

| 基线 | 预期 |
|------|------|
| 动量因子 | +29.5%, CI [19.6%, 40.7%] |
| 反转因子 | -32% |
| Always-bullish | N/A |
| LLM | 待测 |

## 实验参数

- 数据集: `frozen-eval-lowpos-v1.json` (732 pairs)
- Prompt: technical template, DeepSeek-chat, temperature=0.0
- Train: 2018-06, 2018-12, 2019-06, 2020-09, 2021-06, 2022-06
- Test: 2020-03, 2021-12, 2022-10, 2023-06, 2024-02, 2024-10
- 成本上限: ¥38

## 硬约束

- ✓ "LLM 相对动量因子有/无可检出的增量(功效受限)"
- ✗ "低位股回报 X%" / "LLM 验证有效"
- 绝对收益指标盖戳"存活偏差、仅内部参考"
