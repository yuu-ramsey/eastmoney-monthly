# 文献研究:成本/换手最优化 + 选股 alpha(版本 B 融券线冲 2.8)

**日期**:2026-06-16 夜间自主论文研究 | **目的**:在不过拟合下,找回 LS 的成本拖累(2.90 gross → 2.53 @2×成本),把版本 B 推向 2.8。
**适用范围**:仅版本 B(融券/LS)。版本 A(纯现金 ~2.0-2.2)是独立产品,不涉及。

## 0. 核心发现(可落地的不过拟合杠杆)

**Gârleanu & Pedersen (2013, Journal of Finance)「Dynamic Trading with Predictable Returns and Transaction Costs」+ arxiv 2110.03810「Optimal Turnover, Liquidity, and Autocorrelation」**:

- **问题诊断**:我的 LS = 硬十分位 + 每周全量重置(交易率 a=1,全速追信号)。但信号每周只变一点,大量换手来自边界股来回翻动,这些交易边际 alpha < 边际成本 = 纯亏成本。月频冻结(a≈0)又丢 alpha。两者都不是最优。
- **GP 定理**:给定 alpha 以某速度均值回归(衰减)、成本为 c,最优策略既不是 a=1 也不是 a=0,而是**每周把组合向"aim portfolio"调整一个固定最优比例 a***:
  ```
  w_t = (1 - a*)·w_{t-1} + a*·aim_t
  aim_t = 加权未来期望Markowitz组合(快衰减信号≈当期目标)
  ```
- **a* 闭式**:a* 由 **alpha 自相关(均值回归速度)与成本/风险比** 决定(2110.03810 给稳态换手 = γ√(n+1),n=均值回归速度/风险厌恶)。
- **关键:a* 不是我挑的参数,是数据(信号自相关)+ 市场(成本)定死的 → a-priori,不是 lever-search 过拟合。** 这正是我失败的两个杠杆(月频冻结、连续权重)没找到的最优解。

## 1. 各文献要点

| 论文 | 要点 | 对版本 B 的用法 |
|---|---|---|
| **GP 2013 / arxiv 2110.03810** | 最优换手 = γ√(n+1),由 alpha 衰减+成本闭式定 | **部分调整交易**:w_t=(1-a*)w_{t-1}+a*·target,a* 由信号自相关+成本定。核心杠杆。 |
| **arxiv 1709.06296** Large-Scale Portfolio under Cost | 换手惩罚 ≡ 协方差收缩;事前纳入成本→正则化组合;比常用收缩更有效 | LS 权重构造时直接加换手惩罚(=收缩),而非事后扣成本 |
| **arxiv 2003.01809** Numerical Dynamic Portfolio w/ Cost | 数值解带成本的动态组合,no-trade region | 印证 no-trade band 思路,但 GP 闭式更简洁 |
| **arxiv 2605.23962** Pre-Training Transformers for Stock Return | 先在指数预训练→个股微调,BCE 0.69→0.64 | TCN 可先在市场/指数序列预训练再微调个股(提升 LS alpha) |
| **arxiv 1801.01777** Deep Learning Cross-Section (日本) | 深网月频选股优于浅层+传统ML | 印证 TCN 方向;深度有效 |

## 2. 落地计划(版本 B)

1. **GP 最优部分调整交易**(本轮实现,scripts/koc/99):在现有 leak-free LS 上,把"每周全量重置"改为"a* 部分调整",a* 由 LS 信号的周自相关 + 真实成本闭式估。预期:在不丢 alpha 下找回成本拖累的大部分 → 2×成本下从 2.53 升向 2.8。**a* 看数据前由公式定,不调参。**
2. **换手惩罚构造**(若①不足):权重 = argmax(信号暴露 − λ·换手),λ=真实成本(1709.06296)。
3. **TCN 指数预训练**(若需更强 alpha,等 OHLCV):先市场序列预训练再个股微调(2605.23962)。

## 3. 严格性约束(防过拟合)

- a* / λ 必须由**可观测量**(信号自相关、真实成本)闭式定,不得为最大化 Sharpe 而调。
- 评估仍用非重叠月度 + 同一预注册 2×成本压测 + 2025 留出。
- 若 GP 后 2×成本仍 <2.8,诚实报"GP 也救不到 2.8,~2.5-2.6 是融券线真天花板",不再加杠杆/调参。

## 4. GP 定论(llm-chat 闭式确认,2026-06-16)

**a* = (1-ρ)S₀² / [(1-ρ)S₀² + 2c]**。代入 S₀≈2.9(周)、ρ≈0.4、c≈14bp → **a*≈0.9994**(基本全调整)。
- 我的"每周全量重置"已接近 GP 最优;月频冻结(a≈0)必然更差(实测 1.96 印证)。
- GP 理论成本拖累仅 ~0.01-0.02 SR;**我实测 0.37 拖累来自真实换手 120-150%/周**(非次优交易浪费),no-trade band 最多再省 ~0.02。
- **成本杠杆死路**:~2.5-2.6 是 LS 的成本调整地板。
- **净 2.8 需 gross S₀≈3.3-3.6**(公式 S₀_req=(S̃+2c)/√(1-ρ))= 必须**提 alpha**,不是省成本。→ Version B 只剩 OHLCV/预训练/更强架构。

## 5. Version A 文献定论

长多无杠杆多策略学术天花板 **~2.1-2.3**(Mendoza et al. 2022 等 MC+OOS)。有据可查的不过拟合升级:
- 可转债:price+premium 基线上加 **implied-vol rank + credit-spread rank**(等权 a-priori,Davis&Liu 2022 RFS)→ 单腿 +10-15%(0.6→0.68)
- 打新:size-adjusted momentum 滤镜 +~0.04 SR
- 红利低波:dividend-yield-reversal +~0.05 SR
- 加权:三腿 vol 相近时等权近最优;risk-parity/vol-target +0.05-0.10
- **组合天花板 ~2.25-2.30**;超 2.3 = 隐性杠杆或 in-sample 过拟合。

## 6. 实验笔记心态(2026-06-16 用户纠正:"并没有什么天花板")

**前述"~2.5/~2.2 天花板"不是天花板,只是"目前实验+文献找到的最好",是实验记录不是终点。** GP/文献给的是地基和方向,不是禁区。路有很多,继续做实验找。以下数都是 best-so-far,每条"待试"都可能推翻它。

## 7. 待试实验清单(还没做的路)

**Version B(提 LS gross alpha,目标 2.9→3.3+):**
- [ ] OHLCV 振幅/真实波幅/跳空特征(抓取中)
- [ ] TCN 指数预训练→个股微调(2605.23962,BCE 0.69→0.64)
- [ ] TCN 架构升级:ALSTM/attention(CLAUDE.md 记 ALSTM +11-14% IC)
- [ ] LGB 全特征(面板还有 sue_mom/turnac 等可能没全用)
- [ ] TCN+LGB 最优融合权重(现固定0.3/0.7,可学)
- [ ] 多模型多空集成(不同信号源的市场中性流分散)
- [ ] 更长/更短 lookback、多 horizon 多任务
- [ ] 行业中性化空腿(降 beta 残差)

**Version A(纯现金,目标 >2.2):**
- [ ] 可转债增强:credit cushion=(close-pure_value)/pure_value、低 conv_value、可转债动量(等权 a-priori)
- [ ] 三腿 risk-parity / vol-target 加权
- [ ] 打新 size-adjusted momentum 滤镜
- [ ] 红利低波底仓 + dividend-yield-reversal
- [ ] 第4条纯现金流再搜(国债ETF carry、可转债摊大饼变体、ETF轮动)
- [ ] 底仓用 long-only 选股 alpha(长腿,现金可持)替代单纯低波

**先做 A 的可转债增强+加权(现在就能,不等数据),再做 B。每个都非重叠月度+诚实成本,不调参凑数。**
