# 东财月线 AI 分析助手 — 架构图 & 功能图

> 生成日期: 2026-05-29
> 最后更新: 2026-05-29（p0 审计修正：补 Native Messaging 桥、研究层连通性重标、score-fusion 权重来源标注）

---

## 一、系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户交互层 (UI Layer)                         │
├───────────────┬──────────────────┬──────────────────────────────────┤
│  popup.html   │   content.js     │        viewer.html               │
│  popup.js     │  (Shadow DOM)    │        viewer.js                 │
│  设置/成本/    │  FAB按钮+侧面板   │       分析查看器                  │
│  历史/调试     │  "大事提醒"抓取   │       (URL参数加载)              │
└───────┬───────┴────────┬─────────┴─────────────┬────────────────────┘
        │ chrome.runtime │ chrome.runtime        │ URL参数
        │ .sendMessage() │ .sendMessage()        │
        ▼                ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Service Worker 编排层 (background.js)              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    三种分析模式                               │   │
│  │  ┌──────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │ 单次分析  │  │  多Agent辩论模式   │  │  多周期共振模式    │   │   │
│  │  │ Single   │  │ Bull+Bear+       │  │ 月线+周线+日线    │   │   │
│  │  │          │  │ Predictor→Judge  │  │ resonance分析     │   │   │
│  │  └──────────┘  └──────────────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Native Messaging 桥 (chrome.runtime.sendNativeMessage):             │
│    ┌─ query_sector_alpha(code,period,lookback) → 行业alpha         │
│    └─ read(mc_dropout/{code}) → LSTM MC Dropout 预计算JSON         │
└──────────────┬──────────────────────────────────────────────────────┘
               │ chrome.runtime.sendNativeMessage("com.eastmoney_ai.sync", ...)
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│            Native Messaging Host (native-host/server.js)              │
│            独立 Node 进程，Chrome 按需启动，stdio 协议                  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 消息路由 (handleMessage):                                      │   │
│  │  query_sector_alpha → import lib/sector/alpha.js → calcSectorAlpha(db)  │
│  │  read(mc_dropout/{code}) → fs.readFileSync(JSON) → 返回LSTM信号  │
│  │  sync/sync_batch → fs.writeFileSync → .eastmoney-ai/storage/   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                   │                  │
│                        动态 import('../lib/db/connection.js')        │
│                                                   ▼                  │
│                    ┌──────────────────────────────┐                  │
│                    │   lib/db/connection.js        │                  │
│                    │   better-sqlite3 (Node原生)   │                  │
│                    └──────────────┬───────────────┘                  │
│                                   │                                  │
└───────────────────────────────────┼──────────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │      SQLite 数据库            │
                    │  .eastmoney-ai/db/            │
                    │  klines-v2.sqlite             │
                    │  月线/周线/日线/60分K线        │
                    │  行业映射 / 复权因子           │
                    └──────────────────────────────┘
        │                              │
        ▼                              ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│   lib/ 核心引擎层     │   │      缓存 & 持久化层           │
│  (静态 import 闭包,   │   │                              │
│   不含 Node 原生模块)  │   │ chrome.storage.local         │
│                      │   │  (分析缓存/历史/设置)         │
│ ┌──────────────────┐ │   │                              │
│ │ build-prompt.js  │ │   └──────────────────────────────┘
│ │ (Prompt拼接引擎) │ │
│ └────────┬─────────┘ │
│          │           │
│ ┌────────┴─────────┐ │
│ │ lib/agents/      │ │
│ │ (多Agent系统)    │ │
│ └────────┬─────────┘ │
│          │           │
│ ┌────────┴─────────┐ │
│ │ lib/indicators/  │ │   ┌──────────────────────────────┐
│ │ (技术指标库)     │ │   │   lib/data-sources/           │
│ │ MA/MACD/RSI/     │ │   │   多源K线获取+降级            │
│ │ KDJ/BOLL/OBV/ATR │ │   │                              │
│ └────────┬─────────┘ │   │ 东财→百度→新浪→腾讯           │
│          │           │   │ (超时+限流降级)               │
│ ┌────────┴─────────┐ │   └──────────────┬───────────────┘
│ │ lib/signals/     │ │                  │
│ │ (信号工厂)       │ │   ┌──────────────┴───────────────┐
│ │ 15个多/空信号    │ │   │ lib/data-validation/         │
│ └────────┬─────────┘ │   │ K线质量校验(字段/时序/自洽)   │
│          │           │   └──────────────────────────────┘
│ ┌────────┴─────────┐ │
│ │ lib/quant-factors│ │
│ │ 5维量化因子      │ │   ┌──────────────────────────────┐
│ │ (纯计算,无外部) │ │   │  lib/eval/ + lib/evaluation/  │
│ └──────────────────┘ │   │  评估框架 + 夜间流水线        │
│                      │   │  (仅 CLI,不在运行时闭包)      │
│ ┌──────────────────┐ │   └──────────────────────────────┘
│ │ lib/self-backtest│ │
│ │ (自我回测校准)   │ │
│ └──────────────────┘ │
│                      │
│ ┌──────────────────┐ │
│ │ lib/score-fusion │ │  ← regime权重硬编码 (score-fusion.js:29-34)
│ │ (LLM+量化融合)   │ │    不读任何配置文件或研究产物
│ └──────────────────┘ │
└──────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LLM Provider 抽象层 (lib/llm/)                   │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐                         │
│  │ Anthropic       │    │ DeepSeek        │                         │
│  │ claude-sonnet-4 │    │ deepseek-chat   │                         │
│  │ (SSE流式+Tool   │    │ (OpenAI兼容     │                         │
│  │  Use+扩展思考)  │    │  无Tool Use)    │                         │
│  └─────────────────┘    └─────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     外部数据源 & API                                  │
│                                                                     │
│  东方财富API       百度股市通        新浪财经          腾讯财经         │
│  push2his.         finance.pae.     hq.sinajs.cn     web.ifzq.      │
│  eastmoney.com     baidu.com                          gtimg.cn      │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                  离线研究层 (Python / Node CLI)                       │
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║  已接入运行时 (经 Native Messaging 读预计算产物)              ║  │
│  ╠═══════════════════════════════════════════════════════════════╣  │
│  ║                                                               ║  │
│  ║  lib/lstm/  深度学习                                          ║  │
│  ║  ┌─────────────────────────────────────────────────────┐     ║  │
│  ║  │ mc_dropout.py (MC Dropout 不确定性量化)             │     ║  │
│  ║  │   ↓ 50次前向传播 → y3_mean/y3_std/overall_confidence │     ║  │
│  ║  │ cli/mc_export_json.py                               │     ║  │
│  ║  │   ↓ 导出 JSON → .eastmoney-ai/storage/mc_dropout/{code}.json│║  │
│  ║  │ native-host/server.js (read handler)                │     ║  │
│  ║  │   ↓ sendNativeMessage('read', 'mc_dropout/{code}')  │     ║  │
│  ║  │ background.js:325-358                               │     ║  │
│  ║  │   ↓ 不确定性门控: high=跳过, low+medium=注入prompt     │     ║  │
│  ║  │ buildLstmSignalBlock() → prompt                     │     ║  │
│  ║  └─────────────────────────────────────────────────────┘     ║  │
│  ║  Native host 不可用或数据未生成时: 静默降级, 不影响正常分析     ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║  独立 R&D (运行时 0 引用, 不产出运行时消费的配置/产物)      ║  │
│  ╠═══════════════════════════════════════════════════════════════╣  │
│  ║                                                               ║  │
│  ║  kronos/  Transformer 预测 (BSQ+自回归)                       ║  │
│  ║  lib/backtest/  回测引擎 (Walk-Forward/行业中性化)             ║  │
│  ║  lib/portfolio/  投资组合优化 (Black-Litterman/风险平价)        ║  │
│  ║  lib/scanner/  批量扫描 (沪深300+自选股, 仅 CLI)              ║  │
│  ║  lib/uncertainty/  不确定性量化 (仅 CLI eval 使用)            ║  │
│  ║                                                               ║  │
│  ║  这些模块的研究结果记录在 docs/*.md 和 PROGRESS.md 中,         ║  │
│  ║  运行时不做任何读取。                                         ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ cli/ CLI 工具链 (Node)                                    │      │
│  │ 数据库初始化 │ 批量评估 │ 夜间批处理 │ eval 脚本             │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ scripts/ Python 研究实验 (~100+)                          │      │
│  │ 特征工程 │ LSTM实验 │ XGBoost集成 │ IC分析 │ 因子研究     │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、功能模块图 (数据流)

```
                        用户点击分析
                             │
                             ▼
              ┌──────────────────────────┐
              │     解析页面URL和代码      │
              │    parse-url.js           │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │     获取K线数据           │
              │  lib/data-sources/        │
              │  多源降级获取              │
              │  ┌─东财(主)               │
              │  ├─百度(备1)              │
              │  ├─新浪(备2)              │
              │  └─腾讯(备3)              │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │     数据质量校验           │
              │  lib/data-validation/      │
              │  字段完整性/时序/自洽       │
              └────────────┬─────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌────────────┐    ┌───────────────┐    ┌───────────────┐
│ 行业alpha   │    │ LSTM信号      │    │  K线数据      │
│ (异步)     │    │ (异步,可选)    │    │  (主路径)     │
│            │    │               │    │               │
│ sendNative │    │ sendNative    │    │               │
│ Message(   │    │ Message(      │    │               │
│  query_    │    │  read,        │    │               │
│  sector_   │    │  mc_dropout/  │    │               │
│  alpha)    │    │  {code})      │    │               │
│    ↓       │    │    ↓          │    │               │
│ native-    │    │ MC Dropout    │    │               │
│ host →     │    │ JSON → 不确   │    │               │
│ lib/sector │    │ 定性门控      │    │               │
│ /alpha.js  │    │ (high=跳过)   │    │               │
│    ↓       │    │    ↓          │    │               │
│ sectorAlpha│    │ lstmSignal    │    │               │
│ Data       │    │ Data          │    │               │
└────┬───────┘    └───────┬───────┘    └───────┬───────┘
     │                    │                    │
     └────────────────────┼────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ 月线方向    │  │ 周线方向    │  │ 日线方向    │
    │ MA60斜率    │  │ MA20斜率    │  │ MA20斜率    │
    │ +MACD      │  │ +MACD      │  │ +MACD      │
    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │   多周期共振分析           │
              │   lib/multi-period/       │
              │   strong/partial/divergent│
              └────────────┬─────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ 技术指标    │  │ 信号检测    │  │ 量化因子    │
    │ MA/MACD/   │  │ 15个信号    │  │ 5维因子     │
    │ RSI/KDJ/   │  │ 多+空工厂   │  │ 趋势/位置/  │
    │ BOLL/OBV   │  │             │  │ 波动/量/    │
    │            │  │             │  │ 一致性      │
    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
              ┌──────────────────────────────────────┐
              │         Prompt 拼装                   │
              │       lib/build-prompt.js             │
              │                                       │
              │  输入: K线表格 + 指标 + 信号           │
              │       + sectorAlphaData (可选)        │
              │       + lstmSignalData   (可选)       │
              │       + resonance (纯函数格式化)       │
              │                                       │
              │  4风格: 技术/缠论/价值/综合            │
              │  3模式: 单次/辩论/共振                 │
              └────────────┬─────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    ┌──────────────┐             ┌──────────────┐
    │  单次分析     │             │  辩论模式     │
    │  直接调LLM    │             │  并发3Agent   │
    │              │             │  ↓           │
    │              │             │  Judge综合    │
    └──────┬───────┘             └──────┬───────┘
           │                            │
           └────────────┬───────────────┘
                        │
                        ▼
              ┌──────────────────────────┐
              │    LLM Provider 调用      │
              │    lib/llm/              │
              │    Anthropic │ DeepSeek  │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │    结构化输出解析         │
              │   parse-structured-output│
              │   提取JSON 验证字段       │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │    评分融合 (可选)        │
              │   lib/score-fusion.js    │
              │   LLM评分 × 量化评分     │
              │                          │
              │   自适应Regime权重        │
              │   (硬编码常量             │
              │   score-fusion.js:29-34) │
              │   - strong_trend: L30%+Q70%│
              │   - sideways:     L92%+Q8%│
              │   - high_vol:     L85%+Q15%│
              │   - mixed:        L50%+Q50%│
              │   不读配置/研究产物       │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │    缓存写入 + 返回        │
              │   chrome.storage.local   │
              │   analysis:market.code:  │
              │   period:bucket:style:   │
              │   mode:decision           │
              └──────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

              离线评估 & 研究流水线

    ┌─────────────────┐     ┌─────────────────┐
    │ 夜间评估流水线    │     │ 研究实验          │
    │ lib/evaluation/ │     │ scripts/         │
    │                 │     │                  │
    │ nightly.js      │     │ 特征工程          │
    │  → collector    │     │ LSTM训练          │
    │  → cost-guard   │     │ XGBoost集成       │
    │  → draft-review │     │ IC分析            │
    │  → refine       │     │ 因子研究          │
    └────────┬────────┘     └────────┬─────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────────────────────────────┐
    │              SQLite 数据库               │
    │  月线/周线/日线K线 │ 行业映射 │ 复权因子  │
    └──────────────────┬──────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌───────────┐ ┌──────────┐ ┌───────────┐
    │  LSTM 训练 │ │ 回测引擎  │ │ 组合优化   │
    │  lib/lstm/ │ │ backtest │ │ portfolio │
    │  [已接入]  │ │ [独立R&D]│ │ [独立R&D] │
    │  经 Native │ │ 运行时0  │ │ 运行时0   │
    │  Messaging │ │ 引用     │ │ 引用      │
    │  注入prompt│ │          │ │           │
    └───────────┘ └──────────┘ └───────────┘
         │
         ▼
    ┌───────────┐
    │ mc_export │
    │ _json.py  │
    │ → JSON    │
    │ → native- │
    │   host    │
    └───────────┘
```

---

## 三、模块关系总表

| 层级 | 模块 | 职责 | 运行时状态 |
|------|------|------|-----------|
| **交互** | `content.js` / `popup.js` / `viewer.js` | 用户UI、设置管理、分析展示 | 运行时入口 |
| **编排** | `background.js` | 消息路由、模式选择、缓存管理、Native Messaging 桥 | 运行时中枢 |
| **分析引擎** | `lib/build-prompt.js` | Prompt拼装（4风格 × 3模式） | 运行时 |
| | `lib/agents/` | 多Agent辩论（Bull/Bear/Predictor/Judge） | 运行时 |
| | `lib/indicators/` | 技术指标计算（MA/MACD/RSI/KDJ/BOLL/OBV） | 运行时 |
| | `lib/signals/` | 15个交易信号检测 | 运行时 |
| | `lib/multi-period/` | 多周期共振分析 | 运行时 |
| | `lib/quant-factors.js` | 5维量化因子（纯计算） | 仅 CLI |
| | `lib/score-fusion.js` | LLM+量化混合评分，regime 权重硬编码 (`:29-34`) | 仅 CLI |
| | `lib/self-backtest.js` | 自我回测校准 | 运行时 |
| **LLM** | `lib/llm/` | Provider抽象（Anthropic/DeepSeek）+ 定价 | 运行时 |
| **数据** | `lib/data-sources/` | 4源K线获取 + 降级 | 运行时 |
| | `lib/data-validation/` | K线质量校验 | 运行时 |
| | `lib/db/` | SQLite持久化 → **经 Native Messaging 代理访问** | 仅 CLI + native-host |
| **桥接** | `native-host/server.js` | stdio 协议，路由 query_sector_alpha / read(mc_dropout) / sync | 独立 Node 进程 |
| | `lib/sector/alpha.js` | 行业alpha计算 → 由 native-host 动态 import | 仅 native-host |
| **评估** | `lib/eval/` + `lib/evaluation/` | 评估框架 + 夜间流水线 | 仅 CLI |
| **研究（已接入）** | `lib/lstm/` → JSON → native-host → background.js | MC Dropout 不确定性量化，不确定性门控过滤 | 离线训练，运行时读产物 |
| **研究（独立 R&D）** | `kronos/` | Transformer 价格预测，含自有 CLI | 运行时 0 引用 |
| | `lib/backtest/` | Walk-Forward 回测 | 运行时 0 引用 |
| | `lib/portfolio/` | Black-Litterman / 风险平价 | 运行时 0 引用 |
| | `lib/scanner/` | 批量股票扫描 | 仅 CLI |
| | `lib/uncertainty/` | 不确定性量化 | 仅 CLI |
| **CLI** | `cli/` | 命令行工具（DB/批量/批处理） | 仅 CLI |
| **脚本** | `scripts/` (~100+) | Python 研究实验 | 离线 |

---

## 四、关键架构决策（p0 审计确认）

### 4.1 Native Messaging 桥

```
background.js ──sendNativeMessage──→ native-host/server.js ──动态import──→ lib/db/connection.js → SQLite
                                     ↑
                                     Chrome 按需启动，stdio 长度前缀协议
                                     消息类型：query_sector_alpha / read / sync / sync_batch / remove
```

- **承载内容**：
  - `query_sector_alpha(code, period, lookback)` → `lib/sector/alpha.js` → `calcSectorAlpha(db)` → 行业超额收益
  - `read('mc_dropout/{code}')` → 读 `.eastmoney-ai/storage/mc_dropout/{code}.json` → LSTM MC Dropout 信号
  - `sync/sync_batch` → 写 `chrome.storage.local` 数据到磁盘文件
- `background.js` **从不直接 import `lib/db/`**，不引入 `better-sqlite3` 到 Service Worker

### 4.2 研究层连通性

| 模块 | 运行时引用？ | 数据通道 |
|------|------------|---------|
| `lib/lstm/` | **是** | `mc_dropout.py` → `mc_export_json.py` → `.eastmoney-ai/storage/mc_dropout/{code}.json` → native-host `read` handler → `background.js:325-358` → 不确定性门控(high=跳过) → `buildLstmSignalBlock()` → prompt |
| `kronos/` | **否** | 仅 `cli_predict.py` 手动调用 |
| `lib/backtest/` | **否** | 独立 Python 脚本 |
| `lib/portfolio/` | **否** | 独立 Python 脚本 |

### 4.3 Score-fusion 权重来源

`lib/score-fusion.js:29-34` — **硬编码常量**，不读任何配置文件或研究产物：

```js
const ADAPTIVE_WEIGHTS = {
  strong_trend: { llm: 0.30, quant: 0.70 },
  sideways:     { llm: 0.92, quant: 0.08 },
  high_vol:     { llm: 0.85, quant: 0.15 },
  mixed:        { llm: 0.50, quant: 0.50 },
};
```

Regime 检测 (`detectStockRegime`) 基于运行时计算的 `quantResult.factors.f2` (价格位置) 和 `f3` (波动率百分位)，纯函数。

### 4.4 静态 import 边界

Service Worker 静态 import 闭包 = `background.js` → lib/ 下 7 个子目录的传递闭包。
`lib/db/`、`lib/lstm/`、`lib/backtest/`、`lib/portfolio/`、`lib/sector/` 均不在闭包内。
`lib/db/` 仅由 native-host 和 CLI 脚本通过**动态 import()** 访问。

守卫测试：`test/import-guard.test.js`。

---

## 五、目录结构

```
eastmoney-monthly-ai/
├── manifest.json              # Chrome MV3 扩展清单
├── background.js              # Service Worker 编排 + Native Messaging 消费者
├── content.js / content.css   # Shadow DOM 注入
├── popup.html / popup.js      # 设置面板
├── viewer.html / viewer.js    # 独立分析查看器
├── native-host/               # Native Messaging Host (Node 进程)
│   ├── server.js              #   stdio 协议, 动态 import lib/db
│   ├── install.js / uninstall.js
│   ├── launcher.bat
│   └── manifest/
├── lib/                       # 核心库
│   ├── llm/                   #   Provider 抽象
│   ├── agents/                #   多Agent系统
│   ├── indicators/            #   技术指标
│   ├── signals/               #   信号工厂
│   ├── multi-period/          #   多周期共振
│   ├── data-sources/          #   多源K线
│   ├── data-validation/       #   数据校验
│   ├── tools/                 #   LLM工具
│   ├── db/                    #   SQLite (仅 CLI + native-host 可达)
│   ├── sector/                #   行业alpha (仅 native-host 可达)
│   ├── eval/                  #   评估框架 (仅 CLI)
│   ├── evaluation/            #   夜间流水线 (仅 CLI)
│   ├── lstm/                  #   深度学习 (离线训练, 产物经 native-host 注入运行时)
│   ├── backtest/              #   回测引擎 (独立 R&D)
│   ├── portfolio/             #   组合优化 (独立 R&D)
│   ├── scanner/               #   批量扫描 (仅 CLI)
│   ├── dashboard/             #   评分仪表盘 (viewer)
│   └── uncertainty/           #   不确定性量化 (仅 CLI)
├── cli/                       # CLI 工具
├── scripts/                   # Python 研究脚本
├── kronos/                    # Transformer 预测 (独立 R&D)
├── data/                      # 静态数据
├── test/                      # 测试 (含 import-guard.test.js)
└── docs/                      # 文档 (含 p0-audit.md)
```
