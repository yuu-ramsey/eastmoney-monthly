# 东方财富月线 AI 分析助手

> **项目章程**: [.claude-charter.md](.claude-charter.md)（每次任务前必读）

Chrome MV3 扩展，注入到东方财富个股页面（quote.eastmoney.com），
通过 LLM API 分析股票走势。仅本地自用，不发布到 Chrome Web Store。

## Regime-Adaptive MoE 架构（2026-06-05 设计）

### 设计原则

- **无单点故障**：任何组件失效不影响整体
- **LLM 隔离**：LLM 在信号链之外，只做解读
- **机械查表**：regime→权重是硬编码的，不经过任何模型
- **三维专家网格**：专家 = regime × scale × perspective 的交叉点，不是模型名

### 专家注册表（三维 key）

```python
expert_registry = {
    # key = (signal_id, regime, scale)
    # status: verified | anti_signal | pending_validation | pending_feature
    ('momentum', 'bear', 'monthly'):     {'spread': +20.6, 'ci': [10.8, 32.4], 'status': 'verified'},
    ('momentum', 'bull', 'monthly'):     {'spread': -42.2, 'ci': [-94.5, -2.5], 'status': 'anti_signal'},
    ('kronos', 'all', 'monthly'):        {'spread': +9.7, 'ci': [5.1, 15.3], 'status': 'verified'},
    ('reversal', 'sideways', 'monthly'): {'spread': None, 'status': 'pending_validation'},
    ('llm_strong', 'all', 'monthly'):    {'spread': None, 'status': 'pending_deconservative'},
    ('gru_wf', 'bear', 'daily'):         {'spread': None, 'status': 'pending_validation'},
    ('lgb', 'bear', 'daily'):            {'spread': None, 'status': 'pending_validation'},
    ('debate', 'high_vol', 'monthly'):   {'spread': None, 'status': 'pending_validation'},
    ('resonance', 'transition', 'multi'):{'spread': None, 'status': 'pending_validation'},
    ('quant_32d', 'all', 'monthly'):     {'spread': None, 'status': 'pending_validation'},
    ('sector_alpha', 'bull', 'monthly'): {'spread': None, 'status': 'pending_validation'},
    ('money_flow', 'all', 'daily'):      {'spread': None, 'status': 'pending_feature'},
}
```

### Gating 查表（机械，零 LLM）

```python
def get_weights(regime, scale):
    """只激活该 (regime, scale) 下 status=='verified' 的专家"""
    active = {k: v for k, v in expert_registry.items()
              if k[1] in (regime, 'all') and k[2] in (scale, 'multi')
              and v['status'] == 'verified'}
    # 按 spread 量级分配权重，每个 ≤40%，归一化
    return normalize_by_spread(active)
```

### 信号链路中 LLM 角色

| 层 | 有 LLM？ | 角色 |
|----|---------|------|
| Regime Detection | ❌ | 三检测器统计投票 |
| Gating | ❌ | 硬编码查表 |
| 信号聚合 | ❌ | 加权平均 |
| 解读输出 | ✅ | 翻译成人话 |

### 待验证槽位（per-regime 拆分任务）

| 信号 | 当前 | 目标 |
|------|------|------|
| S8 反转 | 24tp 整体 −19.2% | per-regime 拆解，可能在 sideways 下有效 |
| S6 LGB | 涨市 −42.2% / 跌市 +20.6% | 拆到 (lgb, bear, daily) 和 (lgb, bull, daily) |
| S7 动量 | 已有 per-regime | 已拆→验证通过 |
| S5 GRU | Level 2 边界 | 拆到 (gru_wf, bear, daily) 重新验证 |
| S1 LLM | 70% neutral | 拆出 (llm_strong, all, monthly) 仅 strong 标签 |

## 核心架构

- **content.js**：注入 Shadow DOM 隔离的 FAB + 侧边面板，从页面 DOM 抓取
  "大事提醒"事件
- **background.js**（service worker）：抓东财月线/周线/日线 + 资金流 + 调
  LLM API + 缓存
- **lib/llm/**：provider 抽象层，支持 Anthropic 和 DeepSeek
- **lib/agents/**：多智能体架构 Bull/Bear/Predictor/Judge
- **lib/build-prompt.js**：4 种风格 (technical/chanlun/value/comprehensive)
  + 多周期共振 prompt 拼装

## 关键设计决策（不要轻易推翻）

### 路径 Y：决策辅助而非投资建议

prompt 不直接喊"建议买入/卖出"，结论用"如果 X 则验证 Y"的条件式表达。
仅 `decisionMode = true` 时附加"个人决策视角"段落给出明确建议（仅本地
自用，规避合规风险）。

### Cache Key 结构

`analysis:<market>.<code>:<period>:<bucket>:<style>:<mode>:<decision>`

- bucket 按 period 取不同粒度（monthly=YYYY-MM / weekly=YYYY-WW /
  daily=YYYY-MM-DD）
- 不加 provider 维度——避免用户切 provider 时大量重复消耗 token
- 切风格 / 切周期 / 切 decisionMode 都会触发新缓存

### 多 Provider 隔离

- API key 分别存：apiKey:anthropic / apiKey:deepseek
- 模型字段分别存：model:anthropic / model:deepseek
- Anthropic 默认 claude-sonnet-4-6，DeepSeek 默认 deepseek-chat
- popup 切 provider 时 onProviderChange 直接读 storage，**不调
  loadSettings**（避免 select.value 被覆盖回旧值的 bug）

### Chanlun 风格用 FULL 严格性约束（不区分 provider）

之前尝试过给 DeepSeek 用 LITE 版宽松约束，反而输出质量更差（出现
ZG=58/ZD=16 这种全段当中枢的离谱判断）。回滚到 FULL 版统一所有
provider，但 popup 加了警告"chanlun 推荐 Anthropic Claude"。

### DeepSeek 模型字段必填校验

DeepSeek API 收到非法 model 名（如曾经填错的 deepseek-v4-pro）会**静默
回退**到劣质模型，输出会出现价格幻觉（92.21 写成 79.95）。popup.js
的 validateModel() 在 blur 时校验是否在 KNOWN_MODELS 列表内。

## 4 种风格的核心差异

- **technical**：均线 + MACD + 量价 + 陷阱信号检测
- **chanlun**：中枢识别 + 笔/线段 + 三类买卖点 + 背驰判定
- **value**：历史分位 + 长期趋势分期 + 估值阶段定位
- **comprehensive**：技术面 + 价值面共振或背离

每种风格的"分析任务"段落必须独立写，不要为了节省 token 让多种风格共用
同一段 prompt。

## 多智能体辩论模式

- Bull / Bear / Predictor 三个 Agent **并发**调用（Promise.allSettled）
- successCount >= 2 才调 Judge 综合
- Judge 输入 = 三方 Agent 的 partials + 基础事实
- Judge 不重新做技术分析，只做对比 + 评估扎实度 + 综合判断
- Judge 必须在 [偏多/偏空/中性/信号不一致] 中明确选一个
- 多周期共振模式下**不开启**辩论模式（token 成本爆炸）

## Prompt 设计原则

1. 结论先行：每个小节开头一句话点明结论再展开论据
2. 综合结论占 20-30%：放在末尾，是用户最优先阅读部分
3. 历史回顾 ≤ 2 句：每个历史引用必须服务于当前判断，不孤立陈述
4. decisionMode 下价位精确到 2 位小数 + 标注数据来源
5. **严禁** CAPM / DDM / 凯利公式 / Beta / 风险溢价 / 股息折现等
   学术金融模型——A 股个股层面输出虚假精确感

## 测试文件位置

- ``test/*.test.js`` — 纯函数测试
- ``test/agents/*.test.js`` — Agent 层测试

跑测试：``D:/node.js/node.exe --test test/*.test.js test/agents/*.test.js``

## 工作环境

- Windows 系统，PowerShell 环境
- Node.js v24.15.0 安装在 ``D:\node.js\node.exe``
- 项目根目录：``D:\ClaudeProjects\test\eastmoney-monthly-ai``

### 爬虫工具

- **Scrapling v0.4.8**：自适应网页抓取框架，绕过 Cloudflare
  - Python 路径：``D:\ClaudeProjects\eastmoney-monthly-ai\.venv\Scripts\python.exe``
  - 调用方式：隔离在项目 venv 内，通过绝对路径调用
  - 功能：CSS/XPath选择器、自适应解析、反爬绕过、并发爬取
  - GitHub: https://github.com/D4Vinci/Scrapling

## 当前阶段

### LSTM 月线预测优化（2026-05-23 完成）

**最终方案**：月线 61d 特征 + LightGBM/XGBoost/MLP/Ridge Ensemble
- Weighted Ensemble: **CS_IC=+0.177, Hit=59.4%, Top-5% LS=+0.181**
- 46 测试月 (2024+), 2247 只 A 股, 51K 样本
- 年化表现: 2024 IC=+0.195, 2025 IC=+0.186
- 45/46 月 IC 为正 (97.8%)

**所有路径验证结论**：6 条改进路径全部跑完，无一超越基线。月线 61d+树模型集成是当前上限。

**Python 环境**：`.venv`（Python 3.13.11, torch 2.12+cu128, RTX 5070）
**数据**：2247 只 A 股，217K 序列，56 flat features + 47 seq features

### LSTM 学术研究汇编

#### 核心论文（已读已验证）

1. **xLSTM (2405.04517)**：Hochreiter 团队，指数门控 + 矩阵内存，GitHub: NX-AI/xlstm
2. **xLSTM-TS (2408.12408)**：72.8% 方向准确率(EWZ日频)，小波去噪 + xLSTM，GitHub: gonzalopezgil/xlstm-ts
3. **SGP-LSTM (Nature 2024)**：A股4500只截面，Rank IC +1128%，SGP 自动特征构造是最大增量
4. **GBDT+LSTM (2505.23084)**：集成提升 10-15%，验证"LSTM≠树模型替代品，而是互补特征提取器"
5. **LSTM+GCN (PeerJ 2024)**：LSTM hidden state → 下游模型特征列，A50 测试
6. **WD-LSTM (Comput.Econ 2025)**：小波去噪+双层特征选择，因果去噪（只用历史数据）
7. **TFT 置信区间 (IIETA 2025)**：q10/q50/q90 分位数，80%置信区间同方向时才交易

#### 关键结论（来自论文）

- **特征工程 > 模型架构**：SGP-LSTM 的 1128% IC 提升来自特征构造，非模型改进
- **LSTM 定位应切换**：从"直接预测器"→"树模型的特征提取器"（LSTM+GCN 思路）
- **全样本 60% 已是学术顶尖**：论文 Chu(2025) 验证样本外常坍缩到 50%
- **>70% 需置信度过滤**：TFT 策略——只在分位数同向时交易，覆盖率有损但准确率跃升

### 执行路径（按优先级）

#### 已跑过（不再重复）
- [x] Phase A: 分类框架 → 负收益，不如回归
- [x] Phase B+C: Rank Loss + Wavelet + EMA + Walk-Forward → IC 提升但 Hit 仍低
- [x] LSTM GPU sweep (7 configs) → BiLSTM h256-l2 最优，CS_IC +0.126
- [x] v4 Roadmap (3-class + Attention + MC Dropout) → Hit 51.7%, CS_IC +0.114, 三分类无效

#### 下一步执行（排好序，读完论文验证后再跑）

1. **路径 B: LSTM hidden state → LightGBM 特征**（论文 #5 #4 支持）
   - ~~不重训 LSTM，只 forward pass 提取 hidden state (512d)~~ **已跑，失败。LSTM hidden 噪声淹没树模型，CS_IC 下降 12%。关闭。**

2. **路径 A: 扩展特征工程**（论文 #3 支持）
   - ~~55d→80d: ADX/CCI/OBV、K线形态、连涨连跌等~~ **已跑，新增 11 维反而降 IC 17%。FFT 是核心贡献者。关闭。**

3. **路径 E: 日线微观结构 LSTM**（月内量价模式）
   - ~~12月×22天→240时间步→两层LSTM~~ **已跑，Val→Test gap=0.175 严重过拟合，300只股不够。关闭。**

4. **路径 C: 因果小波去噪**（论文 #2 #6 支持）
   - 当前 wdenoise 用的是全局序列（有未来信息泄漏风险）
   - 改为 sliding window causal denoising
   - 预期：修复泄漏可能降低 IC，但更诚实

5. **路径 F: 置信度过滤**（论文 #7 支持）
   - 基于现有 LGB+XGB ensemble（CS_IC=+0.178）
   - 使用 MC Dropout 或 5-seed 投票
   - 不在全样本追求 70%，只在高置信子集达到

### 工作约定（新增）

- **先读论文验证，再写计划，最后跑代码**——不准跳步
- 每个路径跑完必须出对比表（vs 上一步 baseline）
- 特征/标签禁止未来信息泄漏，必须严格时间切分
- 树模型和 LSTM 的 rank correlation 是集成收益的关键指标（< 0.6 = 高收益）
- 全样本 Hit 60% 是合理天花板，不要追求 70%（除非加置信度过滤）

### 数据扩展计划（2026-05-24）

**目标**：日线覆盖从 300 只 → 全部 3265 只 A 股，日+月集成覆盖率 11% → ~100%

**步骤**：
1. [ ] `download_daily_v2.py` — Scrapling 下载 ~3000 只缺失股票日线（进行中）
2. [ ] Retrain Daily LSTM（10-seed, 全量 3000+ 股票）
3. [ ] Daily→Monthly 聚合（覆盖从 11% → ~100%）
4. [ ] Daily+Monthly 重新集成，目标 IC +0.200

### 日线 LSTM 优化（2026-05-23 启动）

**当前基线**：LSTM-7（7层残差，hidden=128，lookback=252天）
- IC 范围：+0.014 ~ +0.062（5个种子），极度不稳定
- 参数量：938K
- Val/test gap 显著

**核心问题**：架构过深+序列过长 → 种子方差巨大（4.4x），不同种子落入不同局部最优

**Qlib 对标**（A股 CSI300, 20种子均值）：
- LSTM (2层, h=64, Alpha360): IC=0.045±0.00 ← 零方差
- ALSTM (2层): IC=0.050±0.00
- HIST: Rank IC=0.067 ← 最强 DL 模型
- LightGBM (Alpha158): IC=0.045

**关键洞察**：Qlib 所有稳定模型都是 **2层 + 小hidden + 短lookback**。

#### 日线执行路径

1. **路径1: 架构瘦身**（最高优先）
   - 7层→2层, hidden 128→64, lookback 252→60天
   - 参数量目标 <200K
   - 5种子跑，目标 std<0.005

2. **路径2: ALSTM 注意力**（论文 #2 支持）
   - Input Attention + Temporal Attention
   - Qlib: ALSTM 比 LSTM IC 提升 11-14%

3. **路径3: 输入特征优化**
   - 采用 Alpha360 风格：6个比率特征（非绝对值）
   - 原则：LSTM看原始时序，树模型看加工指标

4. **路径4: 多种子集成**
   - 10-20种子取均值，直接消除方差

5. **路径5: HIST 概念图**（需行业数据，高级路径）

**预期目标**：稳定 IC 0.050-0.060, 种子 std<0.005

**日线-月线集成**：日线LSTM预测 → 月度聚合 → 月线LGB新特征

**参考文献**：
- Qlib: github.com/microsoft/qlib, arXiv:2009.11189
- ALSTM: Qin et al. (2017) IJCAI, arXiv:1704.02971
- HIST: Xu et al. (2021) arXiv:2110.13716
- TRA: Lin et al. (2021) KDD
- SFM: Zhang et al. (2017) KDD

## 给 Claude Code 的工作约定

### 流程纪律（2026-06-05 确立）

1. **先读文献/文档，再动手**：任何涉及模型架构、特征工程、训练策略的改动，
   必须先查已有的论文笔记（`docs/p3-lstm-literature.md` 等）和项目文档
   （`CLAUDE.md`、`PROGRESS.md`），确认没有被已有结论否定。不要凭记忆。

2. **写代码前先写测试**：涉及 GPU 训练的 Python 脚本，必须先通过小规模 smoke
   test（20 只股票、5 epoch、小 hidden）验证所有模块可用、无语法错误、无
   numpy 兼容性问题、JSON 序列化正常，才能启动全量通宵跑。

3. **防过拟合是强制检查项**：所有训练脚本必须包含：
   - 严格时间切分（train/val/test 按日期，不随机）
   - val-test gap 监控（>0.05 → 标记 overfit）
   - Dropout + weight decay（至少各一个正则化手段）
   - 归一化参数只用 train 前数据计算（防泄漏）

4. **防数据泄漏是强制检查项**：提交前逐项确认：
   - 归一化 mean/std 是否只用了训练期之前的数据
   - 滚动指标（MA/MACD/RSI）是否只用了 cutoff 之前的数据
   - 标签（forward return）的计算是否严格只用 cutoff 之后的数据

### 操作约定

- 改 prompt 文本时只改 lib/build-prompt.js 或 lib/agents/*.js 里的常量，
  不要改函数签名和代码逻辑
- 改 cache key 结构会让所有旧缓存失效，谨慎为之
- 不要为了让测试通过反向修改 src（测试失败说明 src 有问题，停下来等用户
  决策）
- 跑 JS 测试时用 ``D:/node.js/node.exe --test test/*.test.js test/agents/*.test.js``
- 完成任务后必须告诉用户：改动文件清单、测试通过情况、关键设计点
- 用户偏好简洁回复，不需要过度解释

### 通宵实验管线

- **核心库**：`lib/overnight_core.py`（可 import，不触发执行）
- **Smoke test**：`scripts/test_overnight.py`（20 股，先跑这个再跑全量）
- **全量 runner**：`scripts/run_overnight.py`（每晚可跑，增量续跑）
- **日志**：`.eastmoney-ai/overnight_v2/overnight_YYYY-MM-DD.log`
- **结果**：`.eastmoney-ai/overnight_v2/results.jsonl` + `leaderboard-*.md`
