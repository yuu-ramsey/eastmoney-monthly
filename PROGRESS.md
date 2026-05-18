> **所有任务必须先读 .claude-charter.md（项目章程）**

## LLM Prompt 工程边际收益（2026-05-18 认知更新）

| 阶段 | 方法 | score Δ | 累计变化 | 评级 |
|------|------|---------|---------|------|
| 9 | 指标层 + 禁止自算 | +0.011 | 0.079 | 小改进 |
| 10 | 信号 + strong 触发 | **+0.108** | 0.187 | **质变** |
| 11 | 共振约束 | -0.036 | 0.151 | 回退 |
| 12 | 行业 alpha | -0.003 | ~0.15 | 噪声 |
| 13 | ASC confidence | +0.0018 | 0.198 | 噪声 |
| 15 | 多 Agent | +0.0013 | 0.198 | 噪声 |

**结论**：阶段 10 之后，纯 prompt 工程边际收益递减到接近零。
与 BizFinBench (arxiv 2505.19457) 揭示的 LLM 金融预测 40-50% 准确率天花板一致。

**下一步突破方向**：
- 不在 prompt 上继续投入
- 引入完全不同的模型类型 (LSTM/Transformer 时序)
- 用 backtest 验证真实交易价值，不是单期 eval score

## 工作流约定（自测优先）

从本约定确立之日起，所有阶段任务遵守以下流程：

### 实施阶段
1. 按用户给的 spec 实施改动
2. 同步写测试覆盖所有新增/修改的逻辑
3. 如涉及 UI（popup/content.js），用 jsdom 模拟 DOM 写 e2e 测试，覆盖关键交互路径
4. 如涉及外部 API（LLM/东财），用 mock 测试，不要在测试里发真实网络请求

### 自测阶段
1. 跑全套测试，失败自己修，直到全过
2. 用 grep / 静态检查工具检测：
   - 是否引入未捕获的异常路径
   - 是否泄漏 API key 到 console.log
   - 是否破坏现有功能（测试数应该只增不减）
3. 如改动涉及 popup 或 content.js，跑一次 dry-run（mock provider 返回固定文本），确认渲染流程不报错

### 汇报阶段
完成后给用户一份"实测结果摘要"，格式如下：
- 改动文件清单（表格）
- 新增测试数 + 总测试数（原 X → 新 Y）
- 关键设计决策的解释（为什么这样做）
- 已知未覆盖的边界情况（如有）
- 不需要用户手动验证的明确声明

### 用户保留的权力
- 用户随机抽查权：对重大改动用户可指定 1-2 个场景手测
- 用户否决权：发现测试存在但实际有 bug，用户可要求回滚
- 设计歧义仲裁：遇到 spec 模糊时，实施方暂停并询问用户，不要自己拍板

### 实施方禁止行为
- 禁止跳过测试编写直接提交
- 禁止用"trivial change"为由跳过自测
- 禁止在测试里 hardcode 当前实现细节（应该测行为不测实现）
- 禁止自动修改 prompt 模板内容（用户拍板）
- 禁止超出 spec 范围做"顺手优化"（独立改动单独提）

### 测试不通过的处理
如自测发现问题但无法在合理时间内修复：
1. 不要硬交付
2. 把问题清单写在汇报里
3. 标注"阻塞项"等待用户决策
4. 用户可选择：回滚 / 接受带 bug 提交 / 让实施方继续修

---

# 项目进度

## 阶段总览

| 阶段 | 功能 | 状态 | 测试数 |
|------|------|------|--------|
| 1.1 | 多 prompt 模板（4 维度） | 完成 | 11 |
| 1.2 | 多轮对话 | 完成 | 6 |
| 1.3 | 大盘对照（沪深 300） | 完成 | 已含入 build-prompt 测试 |
| 2.1 | 分析历史本地保存 | 完成 | 21 |
| 2.2 | LLM Provider 抽象 | 完成 | 16 |
| 3 | Tool Use（Anthropic 专属） | 完成 | 27 |
| 3.5 | 自我回测 decision log | 完成 | 14 |
| 4 | 推理过程可视化 + 调试面板 | 完成 | 12 |
| 6 | 自学习闭环（夜间复盘 + 审核迭代） | 完成 | 28 |
| 6.5 | 自动化定时分析 + 主动检索 | 完成 | 8 |
| 5 | 综合评分仪表盘 | 完成 | 13 |
| 7.1 | 数据源多源降级 | 完成 | 5 |
| 7.2 | Prompt 评估集 | 完成 | 9 |
| D4 | 行业 Cross-Section 分析 | 完成 | 14 |
| 8 | 本地多周期 K 线数据库 + 源锁定 | 完成 | 31 |
| 9 | 数据健康检查 + 自实现指标层 | 完成 | 23 |
| 10 | 结构化信号识别 + prompt 极端标签 | 完成 | 14 |
| **合计** | | | **440** |

## 阶段 1：多 prompt 模板 + 多轮对话 + 大盘对照

### 1.1 多 prompt 模板

- `lib/prompt-templates.js`：4 个模板（technical/trend/valuation/sentiment）
- 每个模板嵌入 6 条硬约束（数字依据、反方观点、操作建议区间、数据窗口标注、禁止表外数据、区分已收盘 K 线）
- `lib/build-prompt.js`：`buildPromptByTemplate()` 接收 `templateKey` 参数
- popup：分析维度 `<select>` 单选，存 `template` 到 storage

### 1.2 多轮对话

- background.js：`FOLLOW_UP` 消息路由，history 数组直接拼到 Claude messages
- content.js：分析面板底部追问输入框 + 发送 + 清空按钮
- conversationHistory 本地维护，每次追问 append Q&A 并重新渲染
- 对话超 20 条时侧边栏显示 warning

### 1.3 大盘对照

- background.js：并行 fetch 沪深 300 K 线（secid=1.000300），失败不阻塞主流程
- build-prompt.js：`buildIndexBlock()` 生成对比段落（涨跌幅 + 跑赢/跑输百分点）

## 阶段 2：分析历史 + Provider 抽象

### 2.1 分析历史

- `lib/history.js`：纯函数模块（generateHistoryId/trimHistory/historyToMarkdown/formatHistoryDate/checkCapacity）
- background.js：6 个历史相关消息路由（SAVE/GET/DELETE/CLEAR/EXPORT）
- popup：Tab 切换 [设置]/[历史]，卡片列表 + 展开查看 + 导出单条/全部 + 删除/清空
- 容量管理：trimHistory(100 条) + persistHistory 体积检查(9MB)

### 2.2 LLM Provider 抽象

- `lib/llm/anthropic.js`：Anthropic Claude API 适配器
- `lib/llm/deepseek.js`：DeepSeek API 适配器（OpenAI 兼容格式）
- `lib/llm/index.js`：`getProvider(id)` + `listProviders()`
- popup：提供商切换 `<select>`，API key 分别存（apiKey:anthropic / apiKey:deepseek）

## 阶段 3：Tool Use

### 工具

- `lib/tools/get-financials.js`：PE/PB/市值/行业（push2.eastmoney.com/api/qt/stock/get）
- `lib/tools/get-money-flow.js`：近 N 月主力资金流（push2his.eastmoney.com/api/qt/stock/fflow/daykline/get）

### 集成

- `lib/llm/anthropic.js`：while 循环 tool_use，5 轮上限，usage 累加
- `lib/build-prompt.js`：仅 Anthropic provider 时附加 `buildToolInstructions(secid)`
- `background.js`：仅 `provider === 'anthropic'` 时传 tools 数组

### 防护

- 工具 fetch 8s AbortController 超时
- handler 异常/未知工具不中断循环，返回错误文本给 LLM
- 每轮 console.log 记录工具名 + 参数

## 阶段 3.5：自我回测 decision log

### 核心模块

- `lib/self-backtest.js`：3 个导出函数
  - `runHistoricalAnalysis()`：在历史 cutoff 点调 LLM 拿判断（强制用轻量模型省成本）
  - `calculateActualReturn()`：纯数值计算实际涨跌 + 沪深300 alpha
  - `buildSelfCalibrationBlock()`：生成"历史自我校准"markdown 段落

### 集成

- `background.js`：handleAnalyze 单次分析路径，K 线 ≥36 根时选 1-2 个回测时点（48 根选 2，否则选 1）
- 回测判断缓存 30 天（`backtest:<code>:<template>:<cutoff>:<provider>`），actual return 不缓存
- `build-prompt.js`：`extraContext.backtestBlock` 渲染到上下文
- popup：自我回测开关（默认开），存 `enableSelfBacktest`

### 防护

- 仅 single 模式启用（debate 跳过）
- klines < 36 跳过
- 回测 LLM 强制用 Sonnet/chat
- 任何步骤失败 → console.warn，主流程不中断
- 校准段末尾明示"可能存在偏差"

## 阶段 4：推理过程可视化 + Prompt 调试面板

### 4.1 流式展示

- `lib/llm/anthropic.js`：拆分 `nonStreamCall` + `streamCall` 双路径
  - `streamCall`：SSE 解析（message_start/content_block_start/content_block_delta/message_delta/message_stop）
  - 支持 thinking_delta / text_delta / tool_use 事件
  - `onProgress` 回调发射实时事件
  - extended thinking：仅 Opus + `enableThinking=true` 时启用（`thinking.budget_tokens=8000`）
  - 有 `onProgress` 走流式，无回调走原有 while 循环（向后兼容）
- `background.js`：`onProgress` → `chrome.tabs.sendMessage(STREAM_PROGRESS)` → content.js
  - `chrome.alarms` 保持 SW 唤醒（0.5 分钟周期），流式完成后清除
  - `sender.tab.id` 路由确保消息只到触发 tab
- `content.js`：思考流 UI（thinking-stream 区域，含 header + body）
  - thinking：灰色斜体 / text：正常字色 / tool_start：蓝色 / tool_result：绿色更新
  - 分析完成自动折叠，点击 toggle 重新展开

### 4.2 调试面板

- `background.js`：`enableDebugLog=true` 时写入 `debug:lastAnalysis`（仅保留最近 1 条）
  - 字段：timestamp/code/name/template/provider/model/settings/fullPrompt/toolCalls/rawResponse/usage/cost/durationMs
- popup：新增"调试"tab
  - 区域 1：完整 Prompt（折叠 + 复制按钮）
  - 区域 2：工具调用日志（参数 + 返回 + 耗时）
  - 区域 3：LLM 原始输出（折叠 + 复制）
  - 区域 4：Token 用量明细
- popup 设置：`enableThinking` + `enableDebugLog` 两个 checkbox
- 预设系统：快速/深度/辩论/自定义 4 模式一键切换

## 阶段 6：自学习闭环（DeepSeek 夜间复盘 + Claude 用户审核迭代）

### 核心模块

- `lib/evaluation/collector.js`：extractJudgment / evaluateOneAnalysis / evaluateBatch（纯计算）
- `lib/evaluation/cost-guard.js`：预算管理（月/日限额 + 超额抛 BudgetExceededError）
- `lib/evaluation/draft-review.js`：computeStats / pickFailureCases / generateDraftReview（DeepSeek 生成草稿）
- `lib/evaluation/refine.js`：parseUserReview / refineWithClaude（用户审核 + Claude 精修）
- `lib/evaluation/nightly.js`：runNightlyJob 主流程
- `cli/index.js`：ema CLI 命令（nightly / review / budget）

### 成本管控
- 预算默认 ¥50/月，已按用户要求调为 ¥20/月
- 夜间强制 deepseek-chat（不可覆盖）
- 每日 ¥3 硬上限，月度 ¥20 硬上限
- Claude 精修不受严格限制，但计费

### 工作流
夜间 → 纯计算评估 → 触发条件（≥50条+7天 OR 14天保底）→ DeepSeek 生成草稿 → 用户审核勾选 → Claude Opus 生成精修方案 → 用户手动执行改动

## 阶段 6.5：自动化定时分析 + 主动检索

### 核心模块

- `lib/scanner/watchlist.js`：自选股管理（add/remove/list/import，max 50）
- `lib/scanner/hs300.js`：沪深300成分股拉取 + 月缓存（降级策略）
- `lib/scanner/batch-scan.js`：批量扫描器（分批并发10只/批，预算守门，失败跳过）
- `lib/scanner/daily-report.js`：机会股日报（按信号强度排序，top10看多+top10看空）
- `lib/scanner/scheduler.js`：调度路由 + 安全开关

### 调度规则
- 周日第1、3周（HS300周）→ 沪深300 + 自选股 + 日报
- 周日第2、4周（自选股周）→ 仅自选股 + 日报
- 其他天 → evaluation收集 + 复盘触发检查
- 月剩余预算 < ¥5 时不跑HS300
- emergencyStop=true 立即停止所有自动任务

### 成本控制
- 强制 deepseek-chat（不可覆盖）
- 单次约 ¥0.02，沪深300全程约 ¥6
- 预算默认 ¥50/月，日限额 ¥3

### 调度配置
- Windows: `.eastmoney-ai/scripts/setup-windows.ps1`（任务计划程序）
- Linux/Mac: `.eastmoney-ai/scripts/setup-cron.sh`（crontab）
- CLI: `ema scheduler pause/resume/config`

## 阶段 5：综合评分仪表盘

### 核心模块
- `lib/prompt-templates.js`：HARD_CONSTRAINTS 第 7 条（结构化 JSON 输出）
- `lib/dashboard/parse-score.js`：parseScoreBlock / validateScoreData / computeWeightedScore
- `content.js`：dashboard-card HTML + renderDashboard 渲染逻辑 + 降级容错
- `content.css`：仪表盘卡片样式（大分数/信号/label/levels/meta）

### 结构化 JSON 字段
score(0-100) / signal(strong_bull~strong_bear) / confidence(high/medium/low) / key_levels(support/resistance/stop_loss) / trend / position_percentile / one_line_summary

### 容错
- JSON 缺失/格式错/字段超范围 → dashboard 显示 "?" + 警告，不崩溃
- 分析正文正常展示，仪表盘降级不影响主流程

### 集成联动
- collector.js：evaluation 优先用 scoreData.signal（优于 extractJudgment 正则）
- daily-report.js：排序优先用 scoreData.score（优于关键词评分）
- batch-scan.js：扫描结果附带 scoreData

## 阶段 7.1：数据源多源降级

### 核心模块
- `lib/data-sources/eastmoney.js`：东财主源（从 background.js 迁移，字段最全）
- `lib/data-sources/sina.js`：新浪备源（缺 amount/turnoverRate，振幅和涨跌幅自行计算）
- `lib/data-sources/tencent.js`：腾讯末源（同上，需自行计算）
- `lib/data-sources/dispatcher.js`：降级调度（东财→新浪→腾讯）+ 每小时降级上限 20 次 + 日志记录

### 真实 API 验证
- eastmoney 1.5s — 283 根月线，11 字段（全）
- sina 2.8s — 10 根月线，6 字段（缺 amount/change/turnoverRate）
- tencent 0.9s — 10 根月线，6 字段（同上）

### manifest.json
新增 host_permissions：quotes.sina.cn、web.ifzq.gtimg.cn、push2.eastmoney.com

## 阶段 7.2：Prompt 评估集 (Evaluation Set)

### 核心模块
- `lib/eval/seed-stocks.json`：40 只种子股票（6 类各 5-8 只）
- `lib/eval/dataset-builder.js`：buildDataset — 拉K线→选cutoff→算groundTruth
- `lib/eval/runner.js`：runEvaluation + scorePrediction（完全匹配1.0/方向对0.5/neutral0.3/方向反-0.5到-1.0）
- `lib/eval/report.js`：generateEvalReport + compareRuns（版本对比）
- `lib/eval/prompt-versions/v1-baseline.js`：当前 prompt 快照

### 评估工作流
build dataset → run eval (DeepSeek only) → generate report → compare versions

### groundTruth 规则
alpha>10%→strong_bull / >3%→bull / |alpha|≤3%→neutral / <-3%→bear / <-10%→strong_bear

### 成本
40只×4testPoints×4templates=640次×¥0.02≈¥13/次完整eval

### CLI
ema eval build | ema eval run | ema eval report | ema eval compare v1 v2 | ema eval snapshot --label v2

## 阶段 D4：行业 Cross-Section 分析

### 核心模块
- `data/industry-map.json`：申万一级28行业 × 403只A股映射
- `lib/industry-map.js`：加载/查询/覆盖统计
- `lib/cross-section.js`：analyzeIndustry / analyzeAll / enrichWithCrossSection

### 功能
- 行业内排名：绝对score → 相对排名（前X%）
- 行业轮动信号：强弱top5对子
- 行业强度标签：strong/neutral/weak

### CLI
ema cross-section <code> | ema industries | ema rotate | ema industries refresh

## 已知偏差（有意跳过 / spec 合理偏离）

### Ollama provider — 已废弃

spec 阶段 2.2 要求实现 Ollama 本地 provider，实际未实现。原因：
- Chrome MV3 service worker 环境下 localhost CORS 限制
- 本地模型在 A 股技术分析场景下能力不足以支撑严谨判断
- 维持两个 provider（Anthropic + DeepSeek）已覆盖主要使用场景

### get-industry-peers — 跳过

spec 阶段 3.2 标注"可选，优先级最低"，决定不做。后续如有需求可独立加。

### 20 条对话 warning 位置 — 合理改进

spec 要求"在 popup 顶部 warning 提示"，实际实现在 content.js 侧边栏。
原因：用户操作焦点在侧边栏，popup 需要手动点开才能看到。
此偏差为合理改进，后续不再纠正。

## 路线图

### 当前工作

- [x] Phase 1：多 prompt 模板 + 多轮对话 + 大盘对照
- [x] Phase 2：分析历史 + LLM Provider 抽象
- [x] Phase 3：Tool Use（Anthropic 专属）
- [x] Phase 3.5：自我回测 decision log
- [x] Phase 4：推理过程可视化 + Prompt 调试面板
- [x] Phase 6：自学习闭环（夜间复盘 + 审核迭代）
- [x] Phase 6.5：自动化定时分析 + 主动检索
- [x] Phase 5：综合评分仪表盘
- [x] Phase 7.1：数据源多源降级
- [x] Phase 7.2：Prompt 评估集
- [x] Phase 8：本地全市场多周期 K 线数据库（SQLite, 394 测试）

### 下一步

- [ ] Phase 8 实际建库：`ema db init --scope hs300`（建沪深 300 月+周+日线）
- [ ] Phase 8 60min：`ema db init --scope hs300 --periods 60min`（东财恢复后）
- [ ] 盲点 1：Agent 输出语义级回归测试（纯加测试，不动 src）
- [ ] 盲点 2：跨级别一致性校验增强（新功能）

### 未来阶段

#### 阶段 9：多周期 prompt 模板（阶段 8 建库完成后）
现有 4 模板均为月线视角，建好多周期数据库后需要：
- prompt 中"近 N 根月线"改为根据周期动态（N 根 {periodLabel}）
- 位置百分位的时间窗口按周期调整（monthly 5 年 / weekly 2 年 / daily 6 月 / 60min 1 月）
- 操作建议时间尺度按周期适配
- HARD_CONSTRAINTS 6 条保留，但措辞按周期调整

#### 阶段 10：多周期联动判断（阶段 9 后）
- 同时跑 4 周期分析，综合 scoreData
- 输出"多周期共振强度"（0-100）
- 共振规则：
  - 月+周+日 全偏多 → 高置信度入场
  - 月偏多 + 日超买 → 等回调
  - 月偏空 + 60min 反弹 → 反弹陷阱

## 阶段 8：本地全市场多周期 K 线数据库

### 已完成

- [x] SQLite 数据库（`klines-v2.sqlite`, 271 MB）
- [x] 沪深 300 真实成分股（新浪 API, 300 只含 32 创业板）
- [x] 百度作主源（18 字段含换手率, 前复权, 零限流）
- [x] 月/周/日 三周期全量历史（最早 1991-04）
- [x] 单股源锁定（source 列 + 跨源写入抛错）
- [x] 本地读 0-3ms（vs 在线 ~1000ms, 加速 300x+）
- [x] klines-repo 返回格式与 dispatcher 完全一致
- [x] db-init 断点续传 + 进度持久化
- [x] 401 测试全过

### 数据规格

| 周期 | 记录数 | 平均/股 | 日期范围 |
|------|--------|---------|---------|
| 月线 | 62,475 | 208 | 1991-04 ~ 2026-05 |
| 周线 | 262,056 | 874 | 1991-02 ~ 2026-05 |
| 日线 | 1,240,025 | 4,133 | 1991-01 ~ 2026-05 |

### 已知遗留

- [ ] 60min 周期表已建但未填充（待东财恢复验证 klt=60）
- [ ] 全市场 5000 只未建（当前仅 hs300）
- [ ] 复权 adjust_events 表已建但未填充
- [ ] db-update 增量更新仅框架
- [ ] v2-baseline eval 待完成（修完 3 个 bug 后重跑中）

## 阶段 9：数据健康检查 + 自实现指标层

### 已完成
- [x] 强防御基础设施（.npmrc ignore-scripts, 版本锁定, OSV 检查脚本）
- [x] 15 个通达信兼容指标（SMA/EMA/MACD/RSI/KDJ/WR/CCI/Boll/ATR/OBV/MFI/Stochastic）
- [x] 指标已知值对账（茅台 600519 MA5 手工验证偏差 < 0.01）
- [x] 数据健康检查层（5 项检查 A~E, severity 分级）
- [x] prompt 指标表格 + HARD_CONSTRAINTS #8（禁止自算指标）
- [x] v3-indicators eval: 0% LLM 自算指标, score 0.079

### v3 eval 结果
- 640/640 调用成功, ¥12.68, 38.7min
- LLM 自算指标: 0%（HARD_CONSTRAINTS #8 完美生效）
- bear bias: 43%, strong_bull 预测: 0
- 根因诊断: LLM 极端标签回避（190 strong_bull GT，0 次预测 strong_bull）

## 阶段 10：结构化信号识别层

### 已完成
- [x] lib/signals/atoms.js — 6 个原子函数（cross/exist/count/hhv/llv/every）
- [x] lib/signals/factory.js — 18 个买卖信号（MACD 金叉/KDJ 金叉/超跌/突破/均线排列等）
- [x] lib/signals/summary.js — 信号清单生成 + 信号指引（触发 strong_bull/strong_bear 判断）
- [x] HARD_CONSTRAINTS #9（极端标签指引）+ #10（signal 一致性）
- [x] maxTokens=4000 固定 + 截断检测 + token 预算监控
- [x] v4-signals eval: score 0.187 (v3 的 2.4x), strong_bull 首次非零

### v4 eval 结果（审计修正版 — 全量分母）

> **2026-05-18 审计修正**：原表 v4=0.187 为排除 65 条 parse_failed 后的数字（107.4/575）。
> 以下表格统一用全量分母（/640），不可直接与旧表对比。
>
> **审计结论：PROGRESS.md v4 行存在分母作弊。** 详见 `scripts/audit-v4-reparse.js` 和 `scripts/audit-v4-deep.js`。

| 指标 | v1 | v2-fixed | v3 | v4-signals | v5-resonance | v6-sector |
|------|----|---------|-----|-----------|-------------|----------|
| 全量加权 score | ? | ? | 0.079 | **0.1678** | **0.1302** | **0.1064** |
| 样本数 | ? | ? | 640 | 640 | 640 | **280** ⚠️ |
| strong_bull % | — | 0% | 0% | **16.3%** | **12.5%** | **22.9%** |
| bull % | — | — | — | 22.0% | 15.8% | 22.9% |
| bear % | — | 100% | 43% | **31.7%** | **31.1%** | **31.8%** |
| strong_bear % | — | — | — | 9.8% | 10.6% | 7.5% |
| neutral % | — | — | — | 10.0% | 16.4% | 15.0% |
| parse_failed | — | — | 0? | **65/640** | **87/640** | **0/280** |
| 成本 | ¥10.24 | ¥11.14 | ¥12.68 | ¥12.62 | ? | ? |

⚠️ v6-sector 仅 280 样本（18 只股票），不可与 v4/v5 直接对比。

### 已知遗留
- strong_bear 假阳性较高（信号工厂看空信号可能给了错误指引）
- strong_bull 假阳性 38%（过于激进的 strong 标签）
- 信号清单仅月线，未覆盖周/日线多周期共振

### 架构债（已还）

- [x] storage 抽象层（lib/platform/storage.js）— Node/浏览器双环境
- [x] Native Messaging 自动同步 — Chrome → Node 数据桥
- [x] CLI nightly/scheduled 命令真实跑通（不再空壳）
- [x] get-financials 字段单位修正（f43/f162/f167 ÷100）
- [x] dispatcher fallback bug 修复（logDegrade 参数错位）
- [x] 数据源调整 — 百度作主源(18字段/前复权/稳定)，同源锁定防跨源混用

### 数据源约束

**K 线源顺序**：百度 → 新浪 → 腾讯 → 东财（兜底）
- 百度 API 返回前复权数据，价格绝对值与其他源存在偏移（复权基准日不同）
- 偏移不影响技术分析（MA/MACD/支撑阻力），但不能跨源混用价格数据
- **单股源锁定**：SQLite `source` 列记录每条 K 线的数据源，同 code+period 禁止混用
- 改源需清库重建（或 `--force` 覆盖）

**建库时**：`ema db init --source baidu` 全程锁定百度，失败跳过不降级

## 数据信任修复（2026-05-18）

### 修复 1：PROGRESS.md 入 git
- PROGRESS.md 首次 git add + commit
- 旧 v4 数据表替换为审计修正版（全量分母）

### 修复 2：eval 结果 schema 加 version 字段
- `lib/eval/runner.js` runEvaluation 每条记录写 `parserVersion`、`evalRunnerVersion`、`timestamp`
- 未来重解析可识别 schema 兼容性

### 修复 3：分母透明化
- 新增 `lib/eval/compute-score.js`：`computeScoreFull()` 同时输出 full/excl_pf
- 报告默认用 full 分母，excl_pf 仅作辅助参考

## Frozen Eval Dataset（2026-05-18）

### 背景
Phase 11/12 实验发现每次用 `--limit=` 取前 N 只股票，baseline 漂移严重（0.1966→0.0806→0.0981），不同股票子集不可比。

### 解决方案
创建 `data/frozen-eval-dataset-v1.json`，从 Phase 12 Run A (640 样本, 40 只股票) 提取。

**Absolute Baseline**：
- Run ID: `runA-no-sector-2026-05-18-05-12-20`
- 配置: no sector, no resonance, old #12 constraint, deepseek-chat, maxTokens=4000
- **score = 0.1966**（全量分母, 640 样本）
- GT 分布: strong_bull=45, strong_bear=44, bull=22, bear=26, neutral=23

**规则**：后续所有 eval score 对比必须 vs 此 baseline。禁止用 subset baseline。
加载入口: `lib/eval/load-frozen-dataset.js` → `loadFrozenDataset({ version:'v1', subsetStocks:N, seed:42 })`。
Subset 用可复现随机抽样（mulberry32 PRNG），替代旧的 `--limit=` 取前 N 逻辑。

## Phase 11 & 12 量化验证（2026-05-18）

### 方法论

298 只 HS300 股票 × 2018-2025 月度数据，离线纯计算，零 LLM 成本。严格 walk-forward。

### Phase 12: Sector Alpha IC 矩阵

| lookback \ holding | 1m | 3m | 6m | 12m |
|---|---|---|---|---|
| 3m | -0.0075 | -0.0013 | -0.0121 | -0.0015 |
| 6m | -0.0008 | -0.0069 | -0.0103 | -0.0086 |
| 12m | +0.0075 | +0.0056 | +0.0004 | -0.0242 |
| 24m | -0.0017 | -0.0174 | -0.0409 | **-0.0738** |

**结论**：16 个 cell 全部 |IC| < 0.10。Hit Rate 47.8-51.4%。Long-Short Sharpe 全部为负。
**Phase 12 关闭**。代码保留 `lib/sector/`，prompt 注入路径禁用（ENABLE_SECTOR_ALPHA=false）。
详见 `docs/phase12-postmortem.md`。

### Phase 11: 多周期共振预测力

共振信号: strong_bull/bear = 三周期全同向, mild_bull/bear = 二周期同向。

扣除 HS300 等权基准 alpha:

| signal \ holding | 1m | 3m | 6m | 12m | n |
|---|---|---|---|---|---|
| strong_bull | 0.59% | 2.61% | 4.31% | 9.25% | 1,114 |
| mild_bull | 0.87% | 2.95% | 6.10% | 10.70% | 2,721 |
| **strong_bear** | **1.53%** | **3.48%** | **7.67%** | **11.15%** | 1,114 |
| mild_bear | 2.55% | 18.43% | 40.26% | 36.65% | 2,939 |

Long-Short (strong_bull − strong_bear): 全部为负，\|Sharpe\| ≈ 0.10。

**结论**：三周期共振是边缘反向信号（strong_bear 未来收益系统性高于 strong_bull）。
LLM v5 score 下降原因：HARD_CONSTRAINTS #11 强制跟随共振方向，但共振是反向指标。
详见 `docs/phase11-postmortem.md`。

### 对比

| 指标 | Phase 12 sector alpha | Phase 11 resonance |
|------|----------------------|-------------------|
| 最强 IC/Sharpe | IC=-0.074 | Sharpe=-0.195 (1m) |
| 方向 | 反向 | **反向** |
| 状态 | 关闭 | 保留代码，禁用注入 |
| LLM 注入 | 已验证无效（-0.03 Δ） | 已验证有害（v5 < v4） |
| LSTM 候选 | 24m lookback (IC=-0.074) | 反向因子 (Sharpe=0.10) |

### 战略转向

1. **LLM prompt 注入路径暂停**：Phase 11 和 12 的量化信号均 |IC/Sharpe| < 0.20，不适合作为 LLM 硬约束
2. **Phase 17 LSTM**：共振反向信号（Sharpe=0.10）和 sector alpha 24m（IC=-0.07）可作为特征
3. **反向约束实验**：可选路径，Phase 11 改 #11 为反向逻辑后重新 eval（预算 ¥7-15）

### 改动文件

- `cli/analyze-sector-predictive-power.js` — Phase 12 IC 矩阵计算
- `cli/analyze-resonance-predictive-power.js` — Phase 11 共振预测力计算
- `cli/verify-resonance-matrix.js` — 共振矩阵可信度验证
- `docs/phase*.md` — Phase 11 & 12 事后总结
- `lib/prompt-templates.js` — #12 约束文本已改回中性措辞

## Phase 13 ASC（2026-05-18）

### 结果

| 指标 | Frozen Baseline | Phase 13 | Phase 13 high-conf |
|------|----------------|---------|-------------------|
| score | 0.1966 | 0.1984 | **0.225** (+14.5%) |
| n | 640 | 640 | 60 (9.4%) |
| strong_bull | 16.7% | — | — |
| confidence 分布 | — | high=9.4% medium=86% low=2.5% | — |

LLM 自评 confidence 与准确率正相关（high=0.225 > medium=0.199 > unparsed=0），#13 约束生效。
high-conf 子集 score 0.225 是本阶段最大单次提升，但仅覆盖 9.4% 预测。

### 改动

- `lib/uncertainty/asc.js` — confidence 校准模块
- `cli/eval-phase13-asc.js` — ASC eval runner
- `lib/prompt-templates.js` — HARD_CONSTRAINTS #13（confidence 诚实法则）
- 代码保留，默认启用 #13（对总体 score 无影响，high-conf 有增益）

## Phase 15 Multi-Agent（2026-05-18）

### 结果

Dry-run 48 样本（3 只股票），5 agent 架构（Bull/Bear/Technical/Sector/Judge）：
- score 0.1979 vs baseline 0.1966, **Δ=+0.0013（噪声）**
- Judge 输出 0 strong 信号，0/48 extreme predictions
- 成本 ¥0.05/样本, 5× baseline, 性价比不成立

### 诊断发现

- Bull/Bear 独立性通过（不同数据+不同论点）
- Judge 不是简单"取中间值"——确实评估每条论点的数据支撑强度
- Judge 理性放弃当 Bull/Bear 势均力敌时→正确行为但不转化为 score 提升
- Sector Agent 在 Judge 决策中引用率最低，行业 alpha 数据对当前架构贡献有限

### 处置

- 代码保留 `lib/agents/{bull,bear,technical,sector,judge}-agent.js`, `lib/agents/phase15-runner.js`
- **ENABLE_MULTI_AGENT=false** 默认关闭（同 Phase 11 模式）
- 不跑全量 ¥70 eval
- 诊断脚本: `scripts/diagnose-phase15.js`

## Phase 17 LSTM v1 Baseline（2026-05-19）

### 架构

LSTM(input=21, hidden=64, 1 layer, dropout=0.2) → FC(2 heads: y3, y6)
Train: 2015-2021 (15,778 seqs) / Val: 2022-2023 (5,723) / Test: 2024-2026 (4,340)

### 结果

| 指标 | Val | Test | Δ |
|------|-----|------|-----|
| IC y3 | 0.095 (p≈0) | **0.025** (p=0.10) | -0.070 |
| IC y6 | 0.074 | -0.024 | -0.097 |
| Sharpe y3 (ann) | 0.267 | 0.144 | -0.123 |

### 失败原因

1. **Regime change**：2015-2021 牛市规律不适用 2024+ 市场
2. **Val overfit**：monitor IC y3 导致选择了 Val 最优但 Test 失效的 checkpoint
3. **单层 LSTM** 太浅，无法学习复杂的跨截面排序模式
4. **固定 split** 训练：需 walk-forward retraining（Qlib 标准做法）

### 处置

- 代码保留 `lib/lstm/`，checkpoint 留作 Phase 19 信号源
- Test 集已使用，标记为污染，不可再用作单模型评估
- 24-26 数据可作为**组合策略 backtest 评估**（不同任务，不同纪律）
- 文档: `lib/lstm/eval_test.py`, `lib/lstm/eval_val.py`

## Phase 19 Walk-Forward Backtesting（设计）

### 信号源（5 个，已量化验证）

| # | 信号 | 最强指标 | 方向 | 覆盖 |
|---|------|---------|------|------|
| 1 | LSTM v1 pred | IC y3=0.095 (Val) | 正向 | 100% |
| 2 | Resonance reverse | Sharpe=0.195 (1m) | 反向 | 100% |
| 3 | Sector α 24m/12m | IC=0.074 | 反向 | 100% |
| 4 | Phase 13 ASC high-conf | score=0.225 | 正向 | 9.4% |
| 5 | MACD/RSI/KDJ composite | Phase 10 signal | 正向 | 100% |

### 三层架构

```
Signal Layer (lib/signals/registry.js)
  → 统一接口 fn(code, asOfDate) → score ∈ [-1, +1]

Aggregation Layer (lib/backtest/aggregator.js)
  → 等权 / IC-IR 加权 / Rank IC 集成

Portfolio Layer (lib/backtest/portfolio.js)
  → Top-K (10-30), 等权/风险平价, 月度调仓
  → 单股 ≤15%, 行业 ≤25%

Backtest Engine (lib/backtest/engine.js)
  → Walk-forward strict, 滑点 0.3-0.5%, 手续费万2.5
  → In-sample: 2015-2023 (调超参)
  → Live test: 2024-2026 (冻结)
```

### 评估目标

- Sharpe > 1.0, Max DD < 25%
- vs HS300 等权 alpha + IR

### 决策点（等用户确认）

1. 信号集：5 个全用 or 选 subset?
2. 聚合：等权起步 or 直接 IC-IR 加权?
3. 框架：自写 lightweight or vectorbt?
4. LSTM retrain：每月滚动重训 or 静态 Phase 17 checkpoint?

### 改动文件

- `lib/lstm/` — 数据 pipeline + 模型 + 训练 + 评估脚本
- `lib/lstm/requirements.txt` — torch/numpy/pandas/scipy
| 输入特征 | ✅ 就绪 | 15 通达信指标 (Phase 9) + sector alpha (Phase 12) + 共振因子 (Phase 11) |
| 标签 | ✅ 就绪 | 6 月前瞻 alpha (groundTruth) |

### 特征清单

- **价格类**(5): close, open/high/low range, amplitude, change_percent
- **均线类**(4): MA5/MA20/MA60/MA120 位置 + 斜率
- **动量类**(5): MACD_DIF/DEA/HIST, RSI14, KDJ_K/D/J
- **波动类**(2): BOLL upper/lower 距离, ATR
- **量价类**(2): volume ratio, turnover_rate percentile
- **行业类**(1): sector alpha (12m lookback)
- **共振类**(1): resonance signal (-1=strong_bear, 0=neutral, +1=strong_bull)
- **位置类**(1): price percentile (5yr window)

总计 **21 个特征**，覆盖技术面+行业+共振。

### 架构建议

- 输入: (lookback=24 月, 21 features) × batch
- 输出: 6 月前瞻 alpha (regression) 或 5-class signal (classification)
- 模型: LSTM (baseline) → Transformer encoder (如果 LSTM 收敛)
- 评估: Phase 11/12 同款 IC + Hit Rate 矩阵, vs vanilla LLM baseline 0.1966

### 现有代码

项目内无 LSTM 代码残留。需从零搭建。

