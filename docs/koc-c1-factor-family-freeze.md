# C1 — 因子族冻结清单 / Factor Family Freeze (Westfall-Young)

> **Freeze declaration / 冻结声明**: This document enumerates the COMPLETE family of tradable-signal
> claims ever tested in this project, frozen BEFORE the KoC real-data runs (§4 `--real` has not
> executed as of this commit). The Westfall-Young family-wise correction (Chain C) is computed over
> exactly this closed set. Any new signal must first be registered in
> [koc-c2-decision-tree.md](koc-c2-decision-tree.md) and appended here (with its own git timestamp)
> before being tested. The git commit timestamp of this file is the freeze time.
>
> 冻结时间 = 本文件的 git commit 时间。KoC §4 真实数据运行（`03_precheck.py --real`）截至本提交尚未执行。

## 1. 月频价量信号族（7 个，全部已检验）

来源：`data/expert-registry.json`（24 时点无偏池 frozen-eval-lowpos-v2-baostock，含退市股，
train 6 时点/test 6 时点，bootstrap 10k）。**全样本 0/7 通过；分 regime 仅 2 个 cell verified。**

| 信号 | 定义 | 全样本 test spread | 终态 | WY 收益序列来源 |
|------|------|------------------|------|----------------|
| reversal | 超跌反转（低位回升） | −10.3% CI[−17.2,−3.3] | 全样本反信号；sideways anti | 事件月收益可从冻结池重建 ✓ |
| momentum | 相对强度延续 | −0.9% CI[−7.1,+5.5] | bear verified（+11.2%）；sideways anti | 同上 ✓ |
| macd | 金叉 | +12.2% CI[+5.8,+19.2] | bear verified（+11.8%）；sideways anti | 同上 ✓ |
| volume | 放量突破 | −8.0% CI[−24.4,+8.1] | pending-B（CI 巨宽） | 同上 ✓ |
| ma60 | 价格在 MA60 上方 | −13.3% CI[−18.8,−8.2] | 三 regime 全 anti_signal | 同上 ✓ |
| rsi | 月线 RSI 30/70 | 无信号产出 | insufficient_data（月线极少触发） | 排除：无收益序列可言 |
| bollinger | 月线布林触轨 | 无信号产出 | insufficient_data（月线带宽过宽） | 排除：同上 |

## 1b. 周频 regime 策略族（独立注册表，勿与 §1 混算）

来源：`lib/decision-tree.cjs`（P4 Decision Tree v2，CSI300 regime，**周频**，2026-06-07 验证：
跨频率 + p 值审计 + hold-out 回测）。与 §1 的月频 24tp 注册表是**不同频率、不同样本**的两套结果
——同名信号（如 reversal）在两套中状态可以不同且不矛盾，论文中必须分开陈述。

| Claim | spread | decay | wf | 终态 | WY 收益序列来源 |
|-------|--------|-------|-----|------|----------------|
| reversal\|bear（周频） | +9.2 | +1.05 | 0.57 | HIGH | 周收益聚合到月度网格后纳入 ✓ |
| momentumF\|bear（周频） | +8.8 | +0.64 | 0.71 | HIGH | 同上 ✓ |
| momentumF\|sideways（周频） | +7.1 | +1.02 | 0.54 | MEDIUM（5/5 gate） | 同上 ✓ |
| reversal\|bull / momentumF\|bull | +13.7 / +14.2 | — | 0.50 | CANDIDATE（hold-out 失败，2023 系统性为负） | 纳入（失败 claim 也占族预算） |

## 2. ML 预测器族

| 信号 | 实验规模 | 终态 | WY 映射 |
|------|---------|------|---------|
| LSTM（月频 61d/31d 特征，6 条改进路径） | 7 配置 GPU sweep + v4 roadmap | 全部关闭（见 `docs/lstm_path_final_postmortem.md`） | 排除：预测器内部变体，无统一可交易组合序列；以决策树披露 |
| GRU-WF (bear, daily) | Level-2 边界 | pending_validation，未达 verified | 排除（未形成 claim） |
| LGB (bear/bull, daily) | bull −42.2% / bear +20.6% | train/test 系统性翻号，未 verified | 排除（同上） |
| quant_32d（32 维树模型 ensemble） | 月频 CS_IC +0.177 | pending_validation | **可映射性待确认**：监测序列若可完整重建则纳入；否则降级为披露。该决定必须在跑 WY 之前以追加方式落档本文件 |
| Kronos（K 线 transformer） | 24tp spread +9.7% CI[+5.1,+15.3] | verified (all, monthly) | 可映射 ✓ |

## 3. LLM 信号族

| 信号 | 实验 | 终态 | WY 映射 |
|------|------|------|---------|
| llm_strong（强标签子集） | 24tp | pending_deconservative | 排除：24 个稀疏时点不构成月度序列；决策树披露 |
| debate（多空辩论合成） | 24tp | pending_validation | 排除（同上） |
| baseline sentiment / 一致性 / 投票 ensemble | p4 系列实验 | 未通过 | 排除（同上）；实验清单见 `docs/p4-*.md` |

**LLM 终局裁定**（进入架构而非因子族）：LLM 移出信号链，仅做解释层（`docs/p4-registry-assessment.md`）。

## 4. 其他注册槽位

| 信号 | 终态 | WY 映射 |
|------|------|---------|
| sector_alpha (bull, monthly) | pending_validation | 待其检验时追加序列 |
| resonance (transition, multi) | pending_validation | 同上 |
| money_flow (all, daily) | pending_feature（数据未建） | 未检验，不占预算 |

## 5. SUE/PEAD 主族（链 A，本次论文主检验）

| Claim | 定义 | 状态 | WY 映射 |
|-------|------|------|---------|
| SUE 主检验 | 单季 SUE，分母 mean\|surprise\|（§14 钉死），top/bottom 10%，+5d 建仓，20/60/120d 窗 | **未跑真数据** | 核心：日历时间月度 L/S 序列 ✓ |
| SUE 分母变体 ×2 | std(surprises) / 股价 缩放 | 稳健性，同 SUE 主检验相关预计 0.9+ | 纳入（WY 联合重采样下近似不增预算，将实证自证：含/不含两版） |
| B5 可实现性楔子 | 纸面 vs 可执行组合差 | 设计完成未跑 | 描述性度量，不作为独立 alpha claim 进 family |

## 6. family 统计

- 进入 WY 检验的收益序列：月频价量 5（rsi/bollinger 无信号排除）+ 周频 regime 策略 5（§1b，含 2 个 CANDIDATE 失败 claim）+ kronos + SUE 主 + 分母变体 2 = **约 14 条**（quant_32d 待定，见 §2）
- 以决策树披露但不可映射的尝试：LSTM 全路径、GRU、LGB、LLM 族（llm_strong/debate/sentiment 等）、money_flow —— 全部在 §2-§4 列明，**无任何已检验信号被隐藏**
- 排除理由只有两类：(a) 该实验未产出可交易组合的时间序列（预测器内部变体/稀疏时点实验）；(b) 信号从未触发（insufficient_data）

> 注：30+ 实验 → ~10 条可映射序列的收缩不是筛选，是"claim 级"与"实验级"的区别——
> 同一 claim 的多次实验（如 LSTM 七个配置）在 family 中只占其最终 claim 的位置，
> 且这些 claim 因不可映射而以披露代替检验。审稿人可对照 C2 决策树验证完备性。
