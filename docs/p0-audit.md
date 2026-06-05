# P0 架构审计报告

> 审计日期: 2026-05-29 | 范围: 只读, 不修改任何代码 | 审计员: Claude Code

---

## 1. 运行时可达模块集合 vs 仅 CLI/research 集合

**结论: lib/ 树确实混放了运行时代码和研究代码, 但 Service Worker 的 import 图没有误引入 Node 原生模块 —— 运行时可达集合远小于完整 lib/。**

### 运行时可传递到达的 lib/* 子目录

从 `manifest.json` 声明的入口出发, 追踪 `background.js` (service_worker) 的静态 import 链:

**manifest.json:15-17**
```json
"background": {
  "service_worker": "background.js",
  "type": "module"
}
```

**background.js:7-20 静态 import**
```js
import { parseStockUrl } from './lib/parse-url.js';
import { parseKlines } from './lib/parse-klines.js';
import { computeMA } from './lib/compute-ma.js';
import { computeMACD } from './lib/compute-macd.js';
import { buildPrompt, buildPromptByTemplate, buildMultiPeriodPrompt } from './lib/build-prompt.js';
import { getProvider } from './lib/llm/index.js';
import { estimateCost } from './lib/llm/pricing.js';
import { runDebate } from './lib/agents/runner.js';
import { extractStructuredOutput } from './lib/parse-structured-output.js';
import { checkCrossLevelConsistency } from './lib/cross-level-check.js';
import { HISTORY_KEY, ... } from './lib/history.js';
import { getFinancialsTool } from './lib/tools/get-financials.js';
import { getMoneyFlowTool } from './lib/tools/get-money-flow.js';
import { runHistoricalAnalysis, ... } from './lib/self-backtest.js';
```

**lib/build-prompt.js:5-9 静态 import (传递)**
```js
import { buildTemplatePrompt, ... } from './prompt-templates.js';
import { calcNormalizedReturns } from './multi-period/normalized-return.js';
import { calculateAll, tailIndicators } from './indicators/calculate.js';
import { checkKlines } from './data-validation/validate-klines.js';
import { generateSignalSummary, ... } from './signals/summary.js';
```

**lib/build-prompt.js:370 动态 import (仅传递方向 → 共振摘要格式化函数)**
```js
const { formatResonanceSummary, buildResonanceConstraint } = await import('./multi-period/resonance.js');
```

**运行时可达集合:**
```
lib/llm/                (3 文件: index.js, anthropic.js, deepseek.js, pricing.js)
lib/agents/             (11 文件: runner, bull, bear, predictor, judge, base, phase15-*,
                         technical-agent, sector-agent, judge-agent)
lib/multi-period/       (normalized-return.js, resonance.js — 仅纯函数)
lib/indicators/         (calculate.js → core.js, trend.js, momentum.js, volatility.js, volume.js)
lib/signals/            (summary.js → atoms.js, config.js, factory.js)
lib/data-validation/    (validate-klines.js)
lib/tools/              (get-financials.js, get-money-flow.js)
lib/ 顶层               (parse-url, parse-klines, compute-ma, compute-macd, build-prompt,
                         prompt-templates, parse-structured-output, cross-level-check,
                         history, self-backtest, score-fusion, quant-factors,
                         cross-section, industry-map)
```

**仅 CLI/research 集合 (未出现在运行时 import 图中):**
```
lib/db/                 (connection.js 引 better-sqlite3 — Node 原生模块)
lib/eval/               (评估框架, 仅 CLI 调用)
lib/evaluation/         (夜间评估流水线, 仅 CLI 调用)
lib/lstm/               (Python 深度学习, 仅 CLI 调用)
lib/backtest/           (Python 回测引擎)
lib/portfolio/          (Python 组合优化)
lib/sector/             (行业 alpha — 通过 Native Messaging 代理)
lib/scanner/            (批量扫描, 仅 CLI 调用)
lib/dashboard/          (评分仪表盘, 仅 viewer 使用)
lib/uncertainty/        (仅在 CLI eval 中使用)
```

佐证: `content.js` 和 `popup.js` 均为零 import 的独立脚本, 不做传递分析。

---

## 2. 运行时违规依赖点

**结论: 0 个违规。所有 Node-only/原生模块引用均在仅 CLI/research 路径中, 不在 Service Worker 的 import 闭包内。**

### 已验证的 Node 原生模块使用点 (全部不在运行时路径)

| 文件:行 | 模块 | 风险 | 判定 |
|---------|------|------|------|
| `lib/db/connection.js:4` | `better-sqlite3` | Node 原生 C++ 模块 | 安全 — 仅 CLI + native-host 可达 |
| `lib/db/connection.js:5` | `node:fs` | Node 核心模块 | 安全 — 同上 |
| `lib/db/connection.js:6` | `node:path` | Node 核心模块 | 安全 — 同上 |
| `lib/db/connection.js:7` | `node:url` | Node 核心模块 | 安全 — 同上 |
| `native-host/server.js:5-7` | `node:fs`, `node:path`, `node:url` | Node 原生 | 安全 — native-host 是独立 Node 进程 |

### 关键防线: `lib/multi-period/resonance.js` 的动态 import 隔离

**lib/multi-period/resonance.js:12**
```js
// 动态导入避免 service worker 加载 Node 原生模块 (better-sqlite3)
const { getKlines } = await import('../db/klines-repo.js');
```

`getResonanceAsOf()` 函数内部才动态 import `lib/db/klines-repo.js` → `lib/db/connection.js` → `better-sqlite3`。

`build-prompt.js` 虽然动态 import `resonance.js`, 但只解构了 `formatResonanceSummary` 和 `buildResonanceConstraint` 两个纯函数, 从不调用 `getResonanceAsOf()`。

**lib/build-prompt.js:370**
```js
const { formatResonanceSummary, buildResonanceConstraint } = await import('./multi-period/resonance.js');
```

验证: 全局 grep `getResonanceAsOf` 的调用点:

```
lib/multi-period/resonance.js:10   — 函数定义
cli/eval-v6-sector.js:451          — getResonanceAsOf({ getKlines }, ...)
cli/eval-v6-sector.js:456          — getResonanceAsOf(tp.stockCode, tp.cutoffDate)
```

**仅 CLI 脚本 `cli/eval-v6-sector.js` 调用 `getResonanceAsOf`**。Service Worker 路径不会触发 SQLite import。

### crypto 判定

grep 未发现 runtime 路径中有 `require('crypto')` 或 `from 'crypto'`。各模块使用的 Web Crypto API (`crypto.subtle`, `crypto.randomUUID`) 均为 Chrome Service Worker 内置标准 API。

---

## 3. SQLite 访问路径

**结论: 扩展运行时通过 Native Messaging 代理访问 SQLite, 不直接 import `better-sqlite3`。`lib/db/` 完全不在 Service Worker import 图中。**

### 访问链路

```
background.js (Service Worker)
  │
  │ chrome.runtime.sendNativeMessage(NATIVE_HOST, { type: 'query_sector_alpha', ... })
  │ chrome.runtime.sendNativeMessage(NATIVE_HOST, { type: 'read', key: 'mc_dropout/600519' })
  ▼
native-host/server.js (独立 Node 进程, Chrome 按需启动)
  │
  │ await import('../lib/db/connection.js')
  │ const db = getDb();
  │ db.prepare(...)
  ▼
.eastmoney-ai/db/klines-v2.sqlite
```

### 证据

**background.js:309-321 — 行业 alpha 查询**
```js
const alphaResp = await chrome.runtime.sendNativeMessage(NATIVE_HOST, {
  type: 'query_sector_alpha',
  code, period, lookback: 12,
});
if (alphaResp && alphaResp.type === 'sector_alpha' && alphaResp.data) {
  sectorAlphaData = alphaResp.data;
}
```

**background.js:325-358 — MC Dropout LSTM 信号读取**
```js
const mcResp = await chrome.runtime.sendNativeMessage(NATIVE_HOST, {
  type: 'read',
  key: `mc_dropout/${code}`,
});
if (mcResp && mcResp.type === 'read_result' && mcResp.data) {
  // 注入 lstmSignalData 到 prompt
}
```

**native-host/server.js:197-199 — DB 接入点**
```js
const { getDb } = await import('../lib/db/connection.js');
const { calcSectorAlpha } = await import('../lib/sector/alpha.js');
const db = getDb();
```

**native-host/manifest/eastmoney-ai-sync.json — 注册**
```json
{
  "name": "com.eastmoney_ai.sync",
  "description": "东方财富 AI 分析数据同步",
  "path": "LAUNCHER_PATH_PLACEHOLDER",
  "type": "stdio",
  "allowed_origins": ["chrome-extension://EXTENSION_ID_PLACEHOLDER/"]
}
```

### 直接使用 SQLite 的 CLI 入口 (对比)

```
cli/index.js            — ema sync/db 命令
cli/db-init.js          — 建库
cli/db-update.js        — 增量更新
cli/db-status.js        — 状态查询
cli/eval-*.js           — 所有 eval 脚本直接 import lib/db/connection.js
scripts/*.js            — 数据构建脚本
```

---

## 4. 编排与生命周期

**结论: 辩论中间态全存内存变量, 无断点续跑, 有 keepalive 定时器防 Service Worker 休眠。**

### 辩论编排 (lib/agents/runner.js)

**完整代码 (64 行) — 关键逻辑:**

```js
export async function runDebate(ctx, opts) {
  const startTime = Date.now();

  // 第一阶段：三方并发, 结果存内存变量
  const [bull, bear, predictor] = await Promise.allSettled([
    bullAgent.run(ctx, opts),   // → partials.bull
    bearAgent.run(ctx, opts),   // → partials.bear
    predictorAgent.run(ctx, opts), // → partials.predictor
  ]);

  const partials = {
    bull:      bull.status === 'fulfilled' ? bull.value : null,
    bear:      bear.status === 'fulfilled' ? bear.value : null,
    predictor: predictor.status === 'fulfilled' ? predictor.value : null,
  };

  // 第二阶段：Judge 综合（至少 2 个 Agent 成功才调）
  let judge = null;
  const successCount = Object.values(partials).filter((p) => p !== null).length;
  if (successCount >= 2) {
    judge = await judgeAgent.run({ ...ctx, partials }, opts);
  }

  return { partials, errors, judge, totalCost, totalDurationMs };
}
```

- **中间态存储**: 纯 JS 内存变量 `partials`。不写 `chrome.storage.local`。
- **断点续跑**: 无。Service Worker 被杀死后, 所有中间态丢失, 需从头重跑。
- **容错**: `Promise.allSettled` 保证单个 Agent 失败不阻塞整体。

### 生命周期管理 (background.js)

**background.js:395 — keepalive 定时器**
```js
chrome.alarms.create(alarmName, { periodInMinutes: 0.5 });
// 每 30 秒触发, 防止 Service Worker 在长分析期间休眠
```

**background.js:984 — storage 变更监听**
```js
chrome.storage.onChanged.addListener((changes, areaName) => { ... });
```

**manifest.json:6 — 权限声明**
```json
"permissions": ["storage", "alarms"]
```

- 分析结果写 `chrome.storage.local` (带 cache key)。
- 分析进度通过 `chrome.storage.local` 传递 `{'pending:code': {...}}` 状态, content script 轮询此 key 显示进度 UI。
- 无 `onSuspend` / `onSuspendCanceled` 钩子 —— 未实现优雅挂起保存。

---

## 5. 研究层 → 运行时消费

**结论: LSTM 信号是唯一被运行时消费的研究产物 (通过 Native Messaging 读预计算 JSON)。Kronos/backtest/portfolio 为孤岛, 0 运行时引用。Score-fusion 的 regime 权重为硬编码, 不读任何外部文件。**

### 5.1 LSTM — 真实消费路径

数据流:
```
lib/lstm/mc_dropout.py (模型推理)
  → cli/mc_export_json.py (导出 JSON → .eastmoney-ai/storage/mc_dropout/{code}.json)
  → native-host/server.js (read handler, 读 JSON 文件)
  → chrome.runtime.sendNativeMessage → background.js:325-358
  → lib/build-prompt.js → buildLstmSignalBlock() (注入 prompt)
```

**background.js:334 — 不确定性过滤**
```js
if (ulevel === 'high') {
  console.log(`[analyze] MC Dropout high uncertainty for ${code}, 跳过 LSTM 信号`);
  lstmSignalData = null;
} else {
  lstmSignalData = {
    lstm_signal: d.signal,
    lstm_signal_raw: d.signal_raw,
    overall_confidence: d.overall_confidence,
    // ...
  };
}
```

**lib/prompt-templates.js:97 — buildLstmSignalBlock()**
用 `lstmSignalData` 构建包含 MC Dropout 不确定性指标的 Markdown 表格。

当 native host 不可用或数据未预计算时,**静默降级** (catch 块为空), 不影响正常分析。

### 5.2 Kronos — 孤岛

**佐证**: 全局 grep `kronos` 在 `background.js`、`lib/build-prompt.js`、`lib/score-fusion.js`、`lib/quant-factors.js` 中均无匹配。
- `kronos/` 模块有自己的 CLI (`cli_predict.py`) 和编排器, 但不产出运行时消费的文件。
- `background.js` 不读取任何 kronos 输出。

### 5.3 Backtest / Portfolio — 孤岛

**佐证**: grep `black.?litterman|risk.?parity` 在 background.js 中无匹配。
- `lib/backtest/engine.py` 和 `lib/portfolio/optimizer.py` 均为独立 Python 脚本, 不产生运行时配置文件。
- 回测结果存在 `docs/` 和 `PROGRESS.md` 中, 运行时不做任何读取。

### 5.4 Score-fusion regime 权重 — 硬编码

**lib/score-fusion.js:29-34**
```js
const ADAPTIVE_WEIGHTS = {
  strong_trend: { llm: 0.30, quant: 0.70 },
  sideways:     { llm: 0.92, quant: 0.08 },
  high_vol:     { llm: 0.85, quant: 0.15 },
  mixed:        { llm: 0.50, quant: 0.50 },
};
```

- 不读 JSON 配置, 不读研究产物, 不读 chrome.storage。
- Regime 检测 (`detectStockRegime`) 完全基于运行时计算的 `quantResult.factors` 中的 f2 (价格位置) 和 f3 (波动率百分位), 纯函数无副作用。

---

## 6. 待确认 (需要验证)

| 编号 | 事项 | 原因 | 验证方法 |
|------|------|------|---------|
| U1 | `lib/sector/alpha.js` 是否能被 native-host 的 `handleQuerySectorAlpha` 正确调用 | 该函数接受 `better-sqlite3` 实例作为参数, native-host 传入 `getDb()`, 但该函数未在 Service Worker 环境中测试过 | 在已部署环境中跑一次实际分析, 检查 console 是否有 `[analyze] sectorAlpha` 日志 |
| U2 | MC Dropout JSON 数据是否存在 | `background.js:325-358` 尝试从 `mc_dropout/{code}` 读 JSON, 但 `cli/mc_export_json.py` 是否已运行过未知 | 检查 `.eastmoney-ai/storage/mc_dropout/` 目录是否存在 |
| U3 | Native host manifest 是否已安装 | `native-host/manifest/eastmoney-ai-sync.json` 含占位符 `LAUNCHER_PATH_PLACEHOLDER` 和 `EXTENSION_ID_PLACEHOLDER`, 需 `native-host/install.js` 填入实际路径后注册到 Chrome | 检查 `chrome://extensions` → 扩展详情 → Service Worker console 是否报 `Native host not found` |
| U4 | `lib/agents/phase15-runner.js` 是否被任何运行时路径调用 | 搜索仅发现 `cli/eval-phase15.js` 引用, 但 `background.js` 只 import `runDebate` 来自 `lib/agents/runner.js`, 不 import Phase 15 runner | 确认 background.js 中只有旧 debate runner |
| U5 | `viewer.html/viewer.js` 是否在 manifest 注册 | `manifest.json` 中未声明 viewer 为 web_accessible_resources 或 action 入口, 但文件存在 | 确认 viewer 是否通过其他方式打开 (可能直接文件 URL) |
| U6 | 构建工具缺失 → 多 JS 文件如何被浏览器加载 | `package.json` 无 build script, 无 webpack/esbuild/rollup 配置, 但 `background.js` 使用 ES module `import` 将大量 lib/ 文件合并 | Chrome MV3 service_worker 的 `"type": "module"` 声明可原生支持 ES import, 无需打包。这意味着所有 import 的 lib/*.js 都必须出现在扩展包中 |

---

## 总结

| 审计项 | 结论 |
|--------|------|
| lib/ 树混放 | **是**, 但运行时 import 闭包不包含 Node 原生模块 |
| 运行时误用 Node 模块 | **否** — 0 个违规, `lib/db/` 完全隔离在仅 CLI/native-host 路径 |
| SQLite 访问路径 | **Native Messaging 代理**, 运行时非直接访问 |
| 辩论中间态 | **纯内存**, 无断点续跑, 有 keepalive 保活 |
| LSTM → 运行时 | **是** — 通过 Native Messaging 读预计算 JSON |
| Kronos/Backtest/Portfolio → 运行时 | **孤岛** — 0 引用 |
| Score-fusion 权重 | **硬编码常量**, 非读配置/研究产物 |
