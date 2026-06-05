# P0 Architecture Audit Report

> Audit date: 2026-05-29 | Scope: read-only, no code modifications | Auditor: Claude Code

---

## 1. Runtime Reachable Module Set vs CLI/Research-Only Set

**Conclusion: The lib/ tree indeed mixes runtime code and research code, but the Service Worker's import graph has not mistakenly introduced Node native modules — the runtime reachable set is far smaller than the full lib/.**

### Runtime Transitively Reachable lib/* Subdirectories

Starting from `manifest.json` declared entry points, tracing `background.js` (service_worker) static import chain:

**manifest.json:15-17**
```json
"background": {
  "service_worker": "background.js",
  "type": "module"
}
```

**background.js:7-20 static imports**
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

**lib/build-prompt.js:5-9 static imports (transitive)**
```js
import { buildTemplatePrompt, ... } from './prompt-templates.js';
import { calcNormalizedReturns } from './multi-period/normalized-return.js';
import { calculateAll, tailIndicators } from './indicators/calculate.js';
import { checkKlines } from './data-validation/validate-klines.js';
import { generateSignalSummary, ... } from './signals/summary.js';
```

**lib/build-prompt.js:370 dynamic import (only transitive direction -> resonance summary formatting function)**
```js
const { formatResonanceSummary, buildResonanceConstraint } = await import('./multi-period/resonance.js');
```

**Runtime reachable set:**
```
lib/llm/                (3 files: index.js, anthropic.js, deepseek.js, pricing.js)
lib/agents/             (11 files: runner, bull, bear, predictor, judge, base, phase15-*,
                         technical-agent, sector-agent, judge-agent)
lib/multi-period/       (normalized-return.js, resonance.js — pure functions only)
lib/indicators/         (calculate.js -> core.js, trend.js, momentum.js, volatility.js, volume.js)
lib/signals/            (summary.js -> atoms.js, config.js, factory.js)
lib/data-validation/    (validate-klines.js)
lib/tools/              (get-financials.js, get-money-flow.js)
lib/ top-level          (parse-url, parse-klines, compute-ma, compute-macd, build-prompt,
                         prompt-templates, parse-structured-output, cross-level-check,
                         history, self-backtest, score-fusion, quant-factors,
                         cross-section, industry-map)
```

**CLI/Research-only set (not present in runtime import graph):**
```
lib/db/                 (connection.js references better-sqlite3 — Node native module)
lib/eval/               (Evaluation framework, CLI only)
lib/evaluation/         (Nightly evaluation pipeline, CLI only)
lib/lstm/               (Python deep learning, CLI only)
lib/backtest/           (Python backtest engine)
lib/portfolio/          (Python portfolio optimization)
lib/sector/             (Sector alpha — accessed via Native Messaging proxy)
lib/scanner/            (Batch scanning, CLI only)
lib/dashboard/          (Scoring dashboard, viewer only)
lib/uncertainty/        (Only used in CLI eval)
```

Corroboration: `content.js` and `popup.js` are both zero-import standalone scripts; no transitive analysis needed.

---

## 2. Runtime Violation Dependency Points

**Conclusion: 0 violations. All Node-only/native module references are in CLI/research-only paths, not in the Service Worker import closure.**

### Verified Node Native Module Usage Points (all NOT in runtime path)

| File:Line | Module | Risk | Verdict |
|---------|------|------|------|
| `lib/db/connection.js:4` | `better-sqlite3` | Node native C++ module | Safe — only reachable from CLI + native-host |
| `lib/db/connection.js:5` | `node:fs` | Node core module | Safe — same as above |
| `lib/db/connection.js:6` | `node:path` | Node core module | Safe — same as above |
| `lib/db/connection.js:7` | `node:url` | Node core module | Safe — same as above |
| `native-host/server.js:5-7` | `node:fs`, `node:path`, `node:url` | Node native | Safe — native-host is independent Node process |

### Key Defense: Dynamic import isolation in `lib/multi-period/resonance.js`

**lib/multi-period/resonance.js:12**
```js
// Dynamic import to avoid service worker loading Node native module (better-sqlite3)
const { getKlines } = await import('../db/klines-repo.js');
```

`getResonanceAsOf()` function internally dynamic-imports `lib/db/klines-repo.js` -> `lib/db/connection.js` -> `better-sqlite3`.

Although `build-prompt.js` dynamically imports `resonance.js`, it only destructures `formatResonanceSummary` and `buildResonanceConstraint`, two pure functions, and never calls `getResonanceAsOf()`.

**lib/build-prompt.js:370**
```js
const { formatResonanceSummary, buildResonanceConstraint } = await import('./multi-period/resonance.js');
```

Verification: global grep for `getResonanceAsOf` call sites:

```
lib/multi-period/resonance.js:10   — function definition
cli/eval-v6-sector.js:451          — getResonanceAsOf({ getKlines }, ...)
cli/eval-v6-sector.js:456          — getResonanceAsOf(tp.stockCode, tp.cutoffDate)
```

**Only CLI script `cli/eval-v6-sector.js` calls `getResonanceAsOf`**. Service Worker path will not trigger SQLite import.

### crypto Determination

grep found no `require('crypto')` or `from 'crypto'` in runtime paths. All modules use Web Crypto API (`crypto.subtle`, `crypto.randomUUID`), which are Chrome Service Worker built-in standard APIs.

---

## 3. SQLite Access Path

**Conclusion: Extension runtime accesses SQLite via Native Messaging proxy; never directly imports `better-sqlite3`. `lib/db/` is completely absent from the Service Worker import graph.**

### Access Chain

```
background.js (Service Worker)
  |
  | chrome.runtime.sendNativeMessage(NATIVE_HOST, { type: 'query_sector_alpha', ... })
  | chrome.runtime.sendNativeMessage(NATIVE_HOST, { type: 'read', key: 'mc_dropout/600519' })
  v
native-host/server.js (Independent Node process, Chrome launches on demand)
  |
  | await import('../lib/db/connection.js')
  | const db = getDb();
  | db.prepare(...)
  v
.eastmoney-ai/db/klines-v2.sqlite
```

### Evidence

**background.js:309-321 — Sector alpha query**
```js
const alphaResp = await chrome.runtime.sendNativeMessage(NATIVE_HOST, {
  type: 'query_sector_alpha',
  code, period, lookback: 12,
});
if (alphaResp && alphaResp.type === 'sector_alpha' && alphaResp.data) {
  sectorAlphaData = alphaResp.data;
}
```

**background.js:325-358 — MC Dropout LSTM signal read**
```js
const mcResp = await chrome.runtime.sendNativeMessage(NATIVE_HOST, {
  type: 'read',
  key: `mc_dropout/${code}`,
});
if (mcResp && mcResp.type === 'read_result' && mcResp.data) {
  // inject lstmSignalData into prompt
}
```

**native-host/server.js:197-199 — DB access point**
```js
const { getDb } = await import('../lib/db/connection.js');
const { calcSectorAlpha } = await import('../lib/sector/alpha.js');
const db = getDb();
```

**native-host/manifest/eastmoney-ai-sync.json — Registration**
```json
{
  "name": "com.eastmoney_ai.sync",
  "description": "Eastmoney AI Analysis Data Sync",
  "path": "LAUNCHER_PATH_PLACEHOLDER",
  "type": "stdio",
  "allowed_origins": ["chrome-extension://EXTENSION_ID_PLACEHOLDER/"]
}
```

### CLI Entries Using SQLite Directly (for comparison)

```
cli/index.js            — ema sync/db commands
cli/db-init.js          — DB init
cli/db-update.js        — Incremental update
cli/db-status.js        — Status query
cli/eval-*.js           — All eval scripts directly import lib/db/connection.js
scripts/*.js            — Data build scripts
```

---

## 4. Orchestration & Lifecycle

**Conclusion: Debate intermediate state is stored entirely in memory variables; no checkpoint resume; keepalive timer prevents Service Worker from sleeping.**

### Debate Orchestration (lib/agents/runner.js)

**Full code (64 lines) — key logic:**

```js
export async function runDebate(ctx, opts) {
  const startTime = Date.now();

  // Phase 1: three concurrent calls, results stored in memory variables
  const [bull, bear, predictor] = await Promise.allSettled([
    bullAgent.run(ctx, opts),   // -> partials.bull
    bearAgent.run(ctx, opts),   // -> partials.bear
    predictorAgent.run(ctx, opts), // -> partials.predictor
  ]);

  const partials = {
    bull:      bull.status === 'fulfilled' ? bull.value : null,
    bear:      bear.status === 'fulfilled' ? bear.value : null,
    predictor: predictor.status === 'fulfilled' ? predictor.value : null,
  };

  // Phase 2: Judge synthesis (only when at least 2 Agents succeed)
  let judge = null;
  const successCount = Object.values(partials).filter((p) => p !== null).length;
  if (successCount >= 2) {
    judge = await judgeAgent.run({ ...ctx, partials }, opts);
  }

  return { partials, errors, judge, totalCost, totalDurationMs };
}
```

- **Intermediate state storage**: Pure JS memory variable `partials`. Does not write to `chrome.storage.local`.
- **Checkpoint resume**: None. When Service Worker is killed, all intermediate state is lost; must restart from scratch.
- **Fault tolerance**: `Promise.allSettled` ensures single Agent failure does not block overall flow.

### Lifecycle Management (background.js)

**background.js:395 — keepalive timer**
```js
chrome.alarms.create(alarmName, { periodInMinutes: 0.5 });
// Triggers every 30 seconds to prevent Service Worker from sleeping during long analysis
```

**background.js:984 — storage change listener**
```js
chrome.storage.onChanged.addListener((changes, areaName) => { ... });
```

**manifest.json:6 — Permission declaration**
```json
"permissions": ["storage", "alarms"]
```

- Analysis results written to `chrome.storage.local` (with cache key).
- Analysis progress passed via `chrome.storage.local` with `{'pending:code': {...}}` state; content script polls this key to display progress UI.
- No `onSuspend` / `onSuspendCanceled` hooks — graceful suspend save not implemented.

---

## 5. Research Layer -> Runtime Consumption

**Conclusion: LSTM signal is the only research product consumed at runtime (via Native Messaging reading pre-computed JSON). Kronos/backtest/portfolio are islands with 0 runtime references. Score-fusion regime weights are hardcoded, reading no external files.**

### 5.1 LSTM — Real Consumption Path

Data flow:
```
lib/lstm/mc_dropout.py (model inference)
  -> cli/mc_export_json.py (export JSON -> .eastmoney-ai/storage/mc_dropout/{code}.json)
  -> native-host/server.js (read handler, reads JSON file)
  -> chrome.runtime.sendNativeMessage -> background.js:325-358
  -> lib/build-prompt.js -> buildLstmSignalBlock() (inject into prompt)
```

**background.js:334 — Uncertainty filtering**
```js
if (ulevel === 'high') {
  console.log(`[analyze] MC Dropout high uncertainty for ${code}, skip LSTM signal`);
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
Uses `lstmSignalData` to build a Markdown table containing MC Dropout uncertainty metrics.

When native host is unavailable or data is not pre-computed, **silent degradation** (empty catch block), does not affect normal analysis.

### 5.2 Kronos — Island

**Corroboration**: Global grep for `kronos` in `background.js`, `lib/build-prompt.js`, `lib/score-fusion.js`, `lib/quant-factors.js` all return no matches.
- `kronos/` module has its own CLI (`cli_predict.py`) and orchestrator, but does not produce files consumed at runtime.
- `background.js` does not read any kronos output.

### 5.3 Backtest / Portfolio — Islands

**Corroboration**: grep `black.?litterman|risk.?parity` in background.js returns no matches.
- `lib/backtest/engine.py` and `lib/portfolio/optimizer.py` are independent Python scripts that do not produce runtime configuration files.
- Backtest results are stored in `docs/` and `PROGRESS.md`; runtime makes no reads.

### 5.4 Score-fusion Regime Weights — Hardcoded

**lib/score-fusion.js:29-34**
```js
const ADAPTIVE_WEIGHTS = {
  strong_trend: { llm: 0.30, quant: 0.70 },
  sideways:     { llm: 0.92, quant: 0.08 },
  high_vol:     { llm: 0.85, quant: 0.15 },
  mixed:        { llm: 0.50, quant: 0.50 },
};
```

- Does not read JSON config, does not read research outputs, does not read chrome.storage.
- Regime detection (`detectStockRegime`) is entirely based on runtime-computed `quantResult.factors` f2 (price position) and f3 (volatility percentile), pure function with no side effects.

---

## 6. Pending Confirmation (needs verification)

| # | Item | Reason | Verification Method |
|------|------|------|---------|
| U1 | Whether `lib/sector/alpha.js` can be correctly called by native-host's `handleQuerySectorAlpha` | The function accepts a `better-sqlite3` instance as parameter; native-host passes `getDb()`, but the function has not been tested in Service Worker environment | Run an actual analysis in deployed environment, check console for `[analyze] sectorAlpha` log |
| U2 | Whether MC Dropout JSON data exists | `background.js:325-358` attempts to read JSON from `mc_dropout/{code}`, but whether `cli/mc_export_json.py` has been run is unknown | Check if `.eastmoney-ai/storage/mc_dropout/` directory exists |
| U3 | Whether Native host manifest is installed | `native-host/manifest/eastmoney-ai-sync.json` contains placeholders `LAUNCHER_PATH_PLACEHOLDER` and `EXTENSION_ID_PLACEHOLDER`; needs `native-host/install.js` to fill actual paths and register with Chrome | Check `chrome://extensions` -> extension details -> Service Worker console for `Native host not found` errors |
| U4 | Whether `lib/agents/phase15-runner.js` is called by any runtime path | Search only found `cli/eval-phase15.js` reference; but `background.js` only imports `runDebate` from `lib/agents/runner.js`, not Phase 15 runner | Confirm background.js has only old debate runner |
| U5 | Whether `viewer.html/viewer.js` is registered in manifest | `manifest.json` does not declare viewer as web_accessible_resources or action entry, but files exist | Confirm whether viewer is opened via other means (possibly direct file URL) |
| U6 | Build tool missing -> how multiple JS files are loaded by browser | `package.json` has no build script, no webpack/esbuild/rollup config, but `background.js` uses ES module `import` merging many lib/ files | Chrome MV3 service_worker `"type": "module"` declaration natively supports ES import without bundling. This means all imported lib/*.js must be present in the extension package |

---

## Summary

| Audit Item | Conclusion |
|--------|------|
| lib/ tree mixing | **Yes**, but runtime import closure does not contain Node native modules |
| Runtime misuse of Node modules | **No** — 0 violations, `lib/db/` fully isolated in CLI/native-host paths only |
| SQLite access path | **Native Messaging proxy**, runtime no direct access |
| Debate intermediate state | **Pure memory**, no checkpoint resume, keepalive prevents sleep |
| LSTM -> runtime | **Yes** — via Native Messaging reading pre-computed JSON |
| Kronos/Backtest/Portfolio -> runtime | **Islands** — 0 references |
| Score-fusion weights | **Hardcoded constants**, not reading config/research outputs |
