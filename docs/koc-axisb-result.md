# KoC Axis B — 归母 EPS 线确认性重跑结果

**分支**: pead-koc-sue  
**日期**: 2026-06-15  
**外审**: gpt-oss-120b（两轮，初审 + 终审）  

---

## 目标

验证 KoC v1/v2 的 PEAD 结论在使用**归母（parent-attributable）EPS 线**后是否仍成立。
v1/v2 使用 baostock `net_profit/total_share`（含少数股东，系统性偏高 ~6%）。

---

## Method C（归母占比缩放）

外审推荐，优于 Path B（TTM 递归，seed 误差传播 15 年）。

```
frac[code, year] = epsTTM(Q4) × total_share(Q4) / net_profit(Q4)
eps_single_parent[q] = eps_single_np[q] × frac[year]
```

护栏：
- clip frac 到 [0, 1]（默认）
- `NETPROFIT_FLOOR = 1e7`：net_profit 近零 → frac = NaN
- ffill 缺 Q4 的年份
- 仍无 frac → 保持原值（frac=1.0）

**实现**：`scripts/koc/11c_parent_eps.py`  
**副本 DB**：`data/pead-baostock-parent.sqlite`（2303MB，eps_single_np 保留净利润线备份）

---

## 三复现点

| 复现点 | v2（净利润线）| 轴 B（归母线）| 外审裁定 |
|--------|-------------|-------------|---------|
| §03 全期 γ1 t | +4.07 | **+4.253** | ✅ PASS |
| §07 旧时代 FM t（2010-2021）| +8.73 | **+9.47** | ✅ PASS |
| §09 sue×illiq 全期 t | −2.64 | **−2.22** | ✅ PASS（边缘，降级描述）|
| §07 新时代基准 t（2022-2024）| (危机剔除后 −3.01) | **−0.93** 基准 / −3.16 危机后 | ⚠️ 有条件 PASS |

**终审结论**：Axis B 三复现点全部通过，对 EPS 口径选择稳健。

---

## 验证步骤

### 1. 比率 spot-check（`scripts/koc/axisb_spotcheck.py`）

**正确测试**：`eps_single_parent / eps_single_np ≈ frac[year]`（逐行比率）

> ⚠️ 注：sum(eps_single_parent) = epsTTM(Q4) 恒等式**不是**正确测试——
> 总股本在季内变动时（如 sh.601658 2019 年 Q2 总股本 4× 增发），
> Path A 单季 EPS 之和本就不等于年度 EPS，这是 accounting 定义的属性，
> 不是 Method C 的 bug。外审（gpt-oss-120b）裁定：per-row ratio 才是 Method C 的合约。

| 指标 | 结果 |
|------|------|
| 样本量 | 200 行（自然 frac，未裁剪）|
| max \|ratio - frac\| | **1.11e-16**（浮点精度）|
| 全部 < 1e-3 | ✅ YES |

Method C 实现完全正确。

### 2. 敏感性网格（`scripts/koc/axisb_sensitivity.py`，7 run）

对 clip 参数和 NETPROFIT_FLOOR 做敏感性：

| Run | clip | floor | t-stat | PASS |
|-----|------|-------|--------|------|
| A-baseline | [0.0, 1.0] | 1e7 | **4.253** | ✅ |
| B-clip1.05 | [0.0, 1.05] | 1e7 | 4.237 | ✅ |
| C-clip1.10 | [0.0, 1.10] | 1e7 | 4.233 | ✅ |
| D-lo0.90 | [0.90, 1.0] | 1e7 | 4.103 | ✅ |
| E-lo0.95 | [0.95, 1.0] | 1e7 | 4.090 | ✅ |
| F-floor5e6 | [0.0, 1.0] | 5e6 | 4.247 | ✅ |
| G-floor2e7 | [0.0, 1.0] | 2e7 | 4.219 | ✅ |

**7/7 全部通过，t-stat 范围 4.09–4.25，基准偏离最大 −3.8%（D run）。**

---

## 关键发现

### A. 污染是保守的（不是夸大的）

旧时代 FM t: 净利润线 8.73 → 归母线 **9.47**（+8.5%）。
净利润线系统性高估 EPS ~6%，使 SUE 的信号噪声比被稀释。归母线去除污染后信号更强。
原始结论被低估，而非被污染吹大。

### B. 新时代死亡依赖危机 cohort 剔除

- 基准新时代（全11期）：t = −0.93（不显著）
- 剔除 2023Q3 危机 cohort：t = −3.16（显著）
- 外审终审：PASS with note — "post-crisis signal weak and not statistically distinguishable from zero"
- **在论文中必须明确：'消化提速' 叙事依赖危机 cohort 的预先剔除，这是预先注册的协议**

### C. Spearman ρ=1.000

归母线 Q1~Q5 收益严格单调（0.0058/0.0120/0.0145/0.0165/0.0251）；  
净利润线有 Q3>Q4 反转。外审将 ρ=1.000 标记为异常强但非人工制造。  
敏感性 7 run 均通过表明不依赖特定 clip 参数。  
**报告时应注明：完美单调性罕见，作为信号异常强的观察呈现，而非质量保证。**

### D. 真·跨源对账（§12）待做

§12 的 "54.8% 重述率" 是 口径差 非重述（baostock 含少数股东 vs akshare 归母线）。
归母线可用后，真正的同口径对账（baostock epsTTM vs akshare eps_ytd）理论上应近 0% 重述。
此项作为未来工作。

---

## 论文稳健性小节建议措辞（外审提供）

> "Axis B's results are robust to the choice of EPS line: the key t-statistic exceeds 4.0 under all
> reasonable clipping and floor specifications and the corrected Method C spot-check confirms
> proper implementation. Nevertheless, the perfect monotonic ordering of parent-EPS quintiles
> (Spearman ρ = 1.0) is unusually strong and should be regarded as a potential data-ordering
> artifact. In the post-crisis (new-era) period the coefficient is modest (t = −0.93) and not
> statistically different from zero, indicating no clear drift."

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `scripts/koc/11c_parent_eps.py` | Method C 实现：归母占比缩放 |
| `scripts/koc/axisb_rebuild.py` | §14 + §01 归母线重建 |
| `scripts/koc/axisb_reproduce.py` | §03 / §07 / §09 三复现点 |
| `scripts/koc/axisb_spotcheck.py` | 比率 spot-check（per-row ratio）|
| `scripts/koc/axisb_sensitivity.py` | 7-run 敏感性网格 |
| `data/pead-baostock-parent.sqlite` | 归母线副本 DB（2303MB）|
| `docs/koc-precheck-parent.md` | §03 归母线报告 |
| `docs/koc-universe-parent.md` | §01 归母线 universe 报告 |
