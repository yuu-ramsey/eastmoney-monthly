# 东方财富月线 AI 分析助手

Chrome MV3 扩展，注入到东方财富个股页面（quote.eastmoney.com），
通过 LLM API 分析股票走势。仅本地自用，不发布到 Chrome Web Store。

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

## 当前阶段

第三阶段 D 多智能体辩论 + 个人决策模式已完工。下一步：
- 盲点 1：Agent 输出语义级回归测试（纯加测试，不动 src）
- 盲点 2：跨级别一致性校验（涉及新功能）
- 路径 3：历史决策反馈循环（需 2 周真实数据后再做）

## 给 Claude Code 的工作约定

- 改 prompt 文本时只改 lib/build-prompt.js 或 lib/agents/*.js 里的常量，
  不要改函数签名和代码逻辑
- 改 cache key 结构会让所有旧缓存失效，谨慎为之
- 不要为了让测试通过反向修改 src（测试失败说明 src 有问题，停下来等用户
  决策）
- 跑测试时用 ``D:/node.js/node.exe --test test/*.test.js test/agents/*.test.js``
- 完成任务后必须告诉用户：改动文件清单、测试通过情况、关键设计点
- 用户偏好简洁回复，不需要过度解释
