# KoC §4 前置检验报告（Pre-check Kill Point）

**生成时间**：2026-06-15 15:36
**模式**：`real`
**样本**：104,002 观测 / 4,083 只股票
**耗时**：23.8s

## FM 设定

```
Step1 每截面期 OLS：
  future_ret_i = γ0 + γ1·SUE_i + γ2·sue_autocorr_i
               + Σ industry_dummies（industry-map.json L1） + Σ season_dummies + ε_i

Step2 时序均值 γ1，Newey-West t 检验（lags=4）
Kill 门槛：t(γ1) < 3.0
```

**SUE_autocorr** = 该股截至当期的 SUE 序列 PIT AR(1) 系数（persistence 代理）。
控住它后 γ1 还显著 = 投资者对盈余冲击反应不足（under-reaction），而非盈余本身持续。

## FM 回归结果（控 persistence 后 SUE 系数 γ1）

| 指标 | 值 |
|------|----|
| γ1 均值 | 0.008801 |
| Newey-West SE（lags=4） | 0.002069 |
| t-statistic | 4.253 |
| p-value（双尾） | 0.0001 |
| 有效截面期数 | 55 |
| 跳过期数（n<30） | 5 |

**Kill 门槛**：t ≥ 3.0 才 PASS；t < 3.0 → KILL。

**判定：🟢 PASS**

### 各期 γ1（最近 12 期）

| 期 | γ1 | n |
|----|----|-|
| 2021Q4 | +0.0048 | 2223 |
| 2022Q1 | +0.0045 | 2221 |
| 2022Q2 | -0.0143 | 2371 |
| 2022Q3 | -0.0117 | 2257 |
| 2022Q4 | -0.0003 | 2463 |
| 2023Q1 | -0.0055 | 2753 |
| 2023Q2 | -0.0173 | 2684 |
| 2023Q3 | +0.0231 | 2631 |
| 2023Q4 | +0.0062 | 2721 |
| 2024Q1 | +0.0079 | 2784 |
| 2024Q2 | -0.0329 | 2148 |
| 2024Q3 | +0.0144 | 3232 |

## 单调性检验（辅助，以 FM γ1 为准）

| 组 | 平均月收益 |
|----|-----------|
| Q1 | +0.0058 |
| Q2 | +0.0120 |
| Q3 | +0.0145 |
| Q4 | +0.0165 |
| Q5 | +0.0251 |

- L/S 价差：+0.0193
- Spearman ρ = 1.000，p = 0.0000
- ✅ 严格单调

> 注：判定以 FM γ1（控 persistence）为准，单调性为辅助展示。


## PASS 结论

控制 persistence 后 SUE 仍显著预测收益（γ1 t-stat ≥ 2），
**under-reaction 存在**，继续 §5 主测试。

*需用 --real 模式做最终确认后才算正式 PASS。*

## 真实数据运行说明

等以下数据就绪后替换输入，运行 `--real` 得到正式判定：

| 数据 | 脚本 | 状态 |
|------|------|------|
| §14 baostock SUE | 14_sue_baostock.py | ⏳ 等 §10（~86h） |
| §15 日线（收益率） | 15_fetch_baostock_daily.py | ⏳ 等 §10 |
| §16 PIT 市值 | 16_pit_marketcap.py | ⏳ 等 §10 |
| §17 ST 历史 | 17_st_history.py | ✅ 完成 |

```bash
# 真数据到位后：
.venv/Scripts/python.exe scripts/koc/03_precheck.py --real
```
