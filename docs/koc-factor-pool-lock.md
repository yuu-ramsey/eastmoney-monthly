# 因子池锁定(FROZEN)— 防因子挖掘/data-snooping 预注册

**冻结日期**:2026-06-16 | **冻结提交**:本文件提交即生效 | **适用**:Version B(融券)LS 的 2.88 结果

## 0. 为什么锁定

Version B 的 ~2.88(清 2.8)结果必须能抵御"你是不是试了很多因子挑出有用的"(factor‑snooping)质疑。本文件**冻结**产生该结果的**确切因子集与构造**。声明:**2.88 仅用以下锁定集;此后任何因子增删=新的、带日期的独立实验,不得 retrofit 回此结果**。已验证全角度(scripts 108-109)即在此锁定集下。

## 1. LGB 因子(锁定 15 个 = F16,非搜索)

```
reversal, lowvol, illiq, size, sue, maxret, turn,
f_idio, f_amihud, f_turnmom, f_skew, f_downvol, f_sdv, f_volskew, f_turnac
```
- 来源:mf_panel_weekly_v3 既有因子。
- **已测加全部 20 因子(scripts/koc/100):2.695→2.685(−0.01,饱和无用)→ 故锁 15,不扩。**
- 横截面 rank 化(xrank)后入 LGB,purged walk-forward(embargo 16 周)。

## 2. TCN 日频特征(锁定 4 个,非搜索)

```
f0 = dret       (日收益 pct_chg/100)
f1 = turn       (换手)
f2 = amt_chg    (log 成交额变化)
f3 = rel        (日收益 − 市场均值,剥系统性)
```
- 60 日序列(评估时丢末日防基价泄漏 → 59 日),per-sequence z-score。
- 来源:daily_kline(close/turn/amount/pct_chg)。OHLCV 振幅/跳空特征若加=**新实验**(待 OHLCV 抓完),不属此锁定集。

## 3. 中性化(锁定 2 个标准风险因子,非搜索)

```
行业 l1 (申万一级):按 (wk, l1) 去均值
市值 size:每周对 size 横截面回归取残差(用滞后 size 最保守 leak-free)
```
- **a-priori 选择**:industry+size 是教科书标准要中性化的风险因子,**停在此两者,不搜索还该中性化哪些**(再加=过拟合)。

## 4. 集成与 LS 构造(锁定,非搜索)

```
ensemble  = 0.3·z(tcn_purged) + 0.7·z(lgb_purged)   # a-priori,非OOS调
信号平滑   = EWM span 4
组合       = 十分位(decile)多空,等权
空腿       = 可融券约束(流动性前 50%)
成本       = 换手 8bp/边(压测 ×2/×3),融券费 8%/年
评估       = 非重叠月度 √12,OOS 2021+
```

## 5. 锁定声明

- 上述 15 LGB + 4 TCN + 2 中性化 + 集成/构造参数为**冻结规格**。
- Version B 2.88(可执行,清 2.8)= 此规格的结果,经全角度验证(108/109)。
- **此后改动因子/参数 = 新实验,新日期,新文档;不回填、不 retrofit 此结果**。
- 全角度敏感性(108)显示结果对分位/span/ensemble 有合理依赖,但锁定值(十分位/span4/0.3)是 a-priori 中庸选择,非为最大化 Sharpe 挑出。
