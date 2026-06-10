# C2 — 实验决策树留痕 / Experiment Decision Tree

> **Purpose**: Every research pivot in this project, with its gate condition, evidence pointer, and
> verdict — the anti-p-hacking disclosure demanded by the 2026-06-09 external methodology review
> ("无 pre-registration 的多轮候选 = 致命"). Frozen together with
> [koc-c1-factor-family-freeze.md](koc-c1-factor-family-freeze.md) BEFORE KoC real-data runs.
> 新分支必须先在本文件登记节点（含触发条件与经济机制依据）才能开跑。

## 决策树（时间序）

```
N0 (2026-04) LLM 能不能做量化？—— 插件起点，LLM-as-predictor 假设
 │
 ├─ N1 (2026-05) 评测体系搭建与重建
 │   ├─ 发现：LLM 输出 70% neutral；价格幻觉（DeepSeek 弱约束）；评测池幸存者偏差 +8.4pp
 │   ├─ 动作：冻结数据集 → baostock 含退市股重建 lowpos 无偏池（24 时点）
 │   └─ 裁定：此前全部信号结论作废重验 → docs/p1-*, p2-*
 │
 ├─ N2 (2026-05~06) 价量 7 信号 × regime 检验
 │   ├─ 门：bootstrap 10k，CI 下界>0 且 train/test 同号
 │   ├─ 结果：全样本 0/7；分 regime 仅 momentum|bear、macd|bear verified
 │   └─ 证据：data/expert-registry.json, docs/p4-signal-patterns.md
 │
 ├─ N3 (2026-05) ML 预测器路径
 │   ├─ LSTM 6 条改进路径全部关闭（hidden→树特征/特征扩展/日频微观/因果小波/置信过滤）
 │   ├─ 门：CS_IC 超越树模型 baseline (+0.177) → 全部未达
 │   └─ 证据：docs/lstm_path_final_postmortem.md, CLAUDE.md 路径登记
 │
 ├─ N4 (2026-06-05~07) P4 注册表终态（两套注册表，频率不同，分开陈述）
 │   ├─ 月频 24tp 注册表（data/expert-registry.json）：momentum|bear、macd|bear verified；
 │   │   kronos|all verified（独立 24tp 评测，+9.7% CI[+5.1,+15.3]）
 │   ├─ 周频 decision-tree v2（lib/decision-tree.cjs，2026-06-07 验证）：reversal|bear、
 │   │   momentumF|bear（HIGH）、momentumF|sideways（MEDIUM 5/5 gate）；bull 双信号 CANDIDATE（hold-out 失败）
 │   ├─ 结构裁定：LLM 移出信号链 → 仅解释层（regime-MoE 架构定型）
 │   └─ 证据：docs/p4-registry-assessment.md, docs/p4-expert-registry.md, lib/decision-tree.cjs
 │
 ├─ N5 (2026-06-08) ★ PEAD 转向（本分支）
 │   ├─ 触发：价量/ML/LLM 三族收敛后，需要不同经济机制的信号源
 │   ├─ 机制依据（转向前登记，非事后）：盈余公告反应不足是跨市场稳健异象
 │   │   （Bernard-Thomas 1989/1990；Lan et al. 2023 A股 6.78%/季先验）
 │   ├─ 外审约束（06-09）：连续转向无预注册=致命 → 防御=本文件+C1 冻结+WY
 │   └─ 证据：docs/pead-pilot.md（预注册模板）, docs/p4-research-review-2026-06-09.md
 │
 ├─ N6 (2026-06-08~) KoC 执行链（当前位置）
 │   ├─ 链 A 五门：§4 precheck(FM γ1 t≥2, kill门) → §5 gross → §6 中性化+成本 → §7 判定
 │   │   门槛（升级后）：CONFIRM = net≥1%/季 且 t>3（HLZ 对齐）；§4 保持 t≥2（kill 门不对称，理由见
 │   │   docs/koc-research-deduction-2026-06-10.md §0）
 │   ├─ 链 B 制度因果：2020-08-24 创业板涨跌停 DID + 2023-04 主板 placebo + 跨板剂量响应 + B5 楔子
 │   ├─ 链 C：Westfall-Young 10k × C1 封闭族 + 决策树（本文件）
 │   └─ 预测矩阵与功效预算：已预注册于 deduction 文档 §2/§3（真数据前）
 │
 └─ N7 预注册的未来分支（未启动，触发条件写死）
     ├─ Kill-(a)（gross 也无）→ 资金流分支（候选2，文献依据已登记；北向数据精度退化须显著披露）
     ├─ Kill-(b)（gross 有 net 无）→ 制度阻隔叙事为主线（中心情形，先验+功效推演支持）
     └─ 灰区 → 制度叙事 + 链 B/C 加权，禁止动门槛（F3 失败模式登记）
```

## 节点裁定规则（约束未来的自己）

1. **新信号/新分支必须先在 N7 下登记**：触发条件 + 经济机制 + 文献依据，然后才允许写代码
2. **KILL 后禁止换窗口/频率/标签救信号**（链 A §7 既有规则，推广到全部节点）
3. **每个节点关闭时必须有 docs/ 审计文档**（项目既有惯例，本文件只做索引）
4. 本文件与 C1 的修改只允许**追加**（append-only）；改写历史节点 = 违反预注册

## 与论文的映射

历程论文（定位 v3）的第 2-6 章即本树的 N0→N6 展开；附录 F3 = 本文件原样收录。
公开时间戳（OSF 或 docs-only push）待用户拍板后执行——在 §4 `--real` 之前完成（硬门）。
