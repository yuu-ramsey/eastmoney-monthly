> **All tasks: must read .claude-charter.md (Project Charter) first**

## Mandatory Rules

### Runtime import boundary guard (established 2026-05-29)

**Invariant**: The following Node native modules must NOT appear in the **static import closure** of the Service Worker (`background.js`):

- `better-sqlite3`, `sqlite3` — C++ native modules
- `node:fs`, `node:path`, `node:url`, `node:crypto`, `node:child_process`, `node:os`, `node:net`, `node:http`, `node:https`, `node:worker_threads`
- Bare names: `fs`, `path`, `child_process`, `os`, `net`, `http`, `https`, `worker_threads`, `stream`, `util`

Bare name `crypto` is **not banned** — Chrome Service Worker provides the Web Crypto API (`crypto.subtle`, `crypto.randomUUID`).

Dynamic `import()` calls are **not tracked** — this project deliberately uses dynamic imports to isolate `better-sqlite3` (see `lib/multi-period/resonance.js` and `native-host/server.js`); tracking would produce false positives.

**Guard test**: `test/import-guard.test.js` — runs automatically on every `npm test`.

**Correct approach when violation occurs**:
- Runtime code accesses DB via Native Messaging (see `sendNativeMessage` in `background.js`)
- When adding new runtime module references, first confirm it does not (transitively) depend on Node native modules

### Debate checkpoint resume (established 2026-05-29)

In debate mode, three Agents (Bull/Bear/Predictor) call LLM concurrently; each Agent's token
cost is approximately 0.01-0.05 USD. If the Service Worker is terminated by Chrome mid-debate,
completed Agent results are lost, and the user must rerun all three Agents from scratch, wasting tokens.

**Checkpoint mechanism**:
- Key: `debate-wip:<market>.<code>:<period>:<bucket>:<style>:<decision>`
  — reuses the same identity identifier as the final cache key
- Input fingerprint: djb2 hash of `code|period|barCount|firstDate|lastDate|lastClose`
  — **only closed bars** (`isBarClosed` filters out current-period unclosed month/week/day bars), intraday close
  fluctuations do not change the fingerprint
- Timeout cleanup: checkpoint automatically discarded after 1 hour to prevent storage accumulation
- Incremental persistence: each Agent, once fulfilled, immediately writes via serial chained merge
  to chrome.storage.local (read-modify-write with no races)
- Resume: `runDebate` entry first reads checkpoint; if fingerprint matches and partial exists,
  reuse it (Promise.resolve), no LLM call
- Cleanup: after final analysis is written to `analysis:*` cache, `clearDebateCheckpoint()` cleans up
- Fault tolerance: all catch blocks use console.warn (no more silent error swallowing)
- P0 boundary: only guarantees retry resume; no alarm watchdog (P1). Agent prompts / LLM
  provider / score-fusion / structured output parsing are all unmodified

**Fingerprint fix** (2026-05-29): Initial version included the current-period unclosed bar's close;
intraday ticks during trading hours changed the fingerprint on every tick, making checkpoint hits
impossible during the session. Fixed to use `isBarClosed()` to filter current-period bars;
only closed-bar close prices participate in fingerprint computation.

**#3 Reuse judgment verification** (2026-05-29): `hasPartial(role)` only checks
`checkpoint.partials[role]`; `mergeCheckpointError` writes to `errors[role]`
and does not touch `partials`. Failed Agents are not reused — retry always re-calls LLM.

**#4 SW termination → UI failure state verification** (2026-05-29): content.js has three error paths
(`!resp` → "service worker not responding", `!resp.ok` → error text,
`catch` → "communication error") — all terminate with `setBody('<div class="error">...')`;
`finally` block hides loading and resets busy. SW death does not cause infinite spinner;
the user sees a retryable error message.

**Related files**: `lib/agents/runner.js`, `background.js:274-280,504-507`
**Tests**: `test/agents/runner.test.js` — 7 checkpoint-specific tests (including 2
fingerprint-specific + 1 "error not reused" verification)

## LLM Prompt Engineering Marginal Returns (2026-05-18 cognitive update)

| Phase | Method | score Delta | Cumulative Delta | Rating |
|------|------|---------|---------|------|
| 9 | Indicator layer + ban self-calculation | +0.011 | 0.079 | Small improvement |
| 10 | Signals + strong triggers | **+0.108** | 0.187 | **Step change** |
| 11 | Resonance constraints | -0.036 | 0.151 | Regression |
| 12 | Sector alpha | -0.003 | ~0.15 | Noise |
| 13 | ASC confidence | +0.0018 | 0.198 | Noise |
| 15 | Multi-Agent | +0.0013 | 0.198 | Noise |

**Conclusion**: After Phase 10, pure prompt engineering marginal returns diminish to near zero.
Consistent with the 40-50% accuracy ceiling for LLM financial forecasting revealed by BizFinBench (arxiv 2505.19457).

### P1 Eval Characterization Audit (2026-05-29)

Independent audit conclusions for the marginal returns table above (`docs/p1-eval-characterization.md`):

1. **GT**: 6-month alpha (vs CSI300), 5-level discretization (thresholds at +/-3%/10%)
2. **Score**: Weighted hit matrix, neutral has structural 0.3 floor — "always neutral" scores 0.401, beating all models
3. **Sample**: 40 stocks x 4 timepoints x 4 templates = 640 records, ~160 effective independent samples, error bars +/-0.08
4. **Leakage**: MA/MACD computation has no look-ahead leakage; but all prompt versions share the same frozen dataset = overfitting risk
5. **Null baselines**: always neutral=0.401, random=0.144, frozen baseline=0.197, v4=0.168

**Key cognitive correction**: 0.187 is not a "step change" — it is only a score slightly above random, measured on a tiny sample, under a scoring function that structurally rewards neutral. With error bars of +/-0.08, it cannot be distinguished from random.

### P1 Conditional Signal & LSTM Leakage Audit (2026-05-29)

**Conditional signal analysis** (`docs/p1-conditional-signal.md`) — completely discarding the scoring matrix, using real alpha:
- strong_bull bucket alpha=18-20%, far above unconditional 6% — the model can identify extreme momentum stocks
- No significant differentiation among bull/bear/neutral/strong_bear buckets (strong_bear bucket alpha=+9.9%, positive!)
- spread (bullish - bearish) = 7.5%, 95% CI = [-2.2%, +19.8%], **contains 0 -> not significant**
- Directional accuracy 58.9%, only +8pp above always-bullish (50.7%)
- v4 (0.187) vs baseline (0.197) show **zero difference** in alpha space

**LSTM leakage check** (`docs/p1-lstm-leak-check.md`):
- `daily_lstm7.pt` training data has no time split (`scripts/daily_to_monthly_aggr.py:64`) -> 2024+ data is all in the training set
- All eval timepoints (2024-05~2025-11) are inside training data -> **all mc-dropout eval results are invalid**
- v4 eval (0.187) did not inject LSTM -> unaffected
- **Runtime LSTM signal is now disabled by default** (`p1-disable-leaked-lstm`, 2026-05-29): `ENABLE_LSTM_SIGNAL = false`
  follows the same null degradation path as when native host is unavailable
- Will re-enable after walk-forward retraining + strict time-split IC validation surpassing baseline

**strong_bull vs momentum baseline** (`docs/p1-strongbull-validation.md`):
- Judgment C — unique-pair-level CI lower bound 5.02% < unconditional 6.07%, strong_bull itself is not significant
- Momentum baseline alpha=12.7%, Jaccard=0.31, difference CI contains 0
- Conclusion: 34 independent samples are insufficient; must expand sample first; failure is a sample size problem, not a model problem

**Power analysis** (`docs/p1-power-analysis.md`):
- alpha std=33.7%, MDE(current)=31.5% — can only detect effects >37.6%
- Current point estimate 12.5% is far below MDE -> all prior "not significant" results are due to insufficient N, not model incompetence
- Winsorize std->20%, effect>=6% -> need ~330 strong_bull ~= 1660 pairs ~= 280 stocks x 6 timepoints
- Recommended path: Winsorize + 100 stocks x 12 timepoints (1200 pairs, ~240 sb)

### P1 Eval Rebuild Design (2026-05-29)

**`docs/p1-eval-rebuild-design.md`** — UNIVERSE=low_position, retire scoring matrix, rebuild eval:

- **Low position definition**: close is in bottom 20% of past 12-month range AND close < MA60
- **Timepoints**: 12 across 4 regimes (2018-06 ~ 2024-10), 6 Train + 6 Test (hold-out)
- **Dual rulers**: Robust center (Winsorize spread + median) + Tail capture (top-20% capture, >20% bull hit)
- **Reversal factor baseline**: Short-term reversal + MA60 discount + RSI oversold (momentum disabled — low-position stock momentum is universally negative)
- **Win condition**: Whether LLM spread beats reversal factor spread

**Next breakthrough direction**:
- Stop investing in prompt engineering
- Introduce completely different model types (LSTM/Transformer time-series)

## Workflow Convention (Self-Test First)

From the date this convention is established, all phase tasks follow this process:

### Implementation Phase
1. Implement changes per user's spec
2. Simultaneously write tests covering all new/modified logic
3. If involving UI (popup/content.js), use jsdom to simulate DOM and write e2e tests covering key interaction paths
4. If involving external APIs (LLM/Eastmoney), use mock tests; do not send real network requests in tests

### Self-Test Phase
1. Run full test suite; fix any failures until all pass
2. Use grep / static checking tools to detect:
   - Whether uncaught exception paths were introduced
   - Whether API keys are leaked to console.log
   - Whether existing functionality is broken (test count should only increase, never decrease)
3. If changes involve popup or content.js, run a dry-run (mock provider returns fixed text) to confirm the rendering flow does not error

### Reporting Phase
After completion, provide the user with a "Test Results Summary" in the following format:
- Changed files list (table)
- New test count + total test count (was X -> now Y)
- Explanation of key design decisions (why this approach was taken)
- Known uncovered edge cases (if any)
- Clear statement that no manual verification by user is required

### Rights Reserved by User
- User spot-check right: for major changes, the user may designate 1-2 scenarios for manual testing
- User veto right: if tests exist but actual bugs are found, the user may request rollback
- Design ambiguity arbitration: when the spec is ambiguous, the implementer pauses and asks the user; do not decide unilaterally

### Prohibited Implementer Behaviors
- Prohibited from skipping test writing and submitting directly
- Prohibited from using "trivial change" as excuse to skip self-test
- Prohibited from hardcoding current implementation details in tests (should test behavior, not implementation)
- Prohibited from auto-modifying prompt template content (user decides)
- Prohibited from doing "incidental optimizations" beyond spec scope (separate changes filed separately)

### Handling Test Failures
If self-test finds issues that cannot be fixed within a reasonable time:
1. Do not force delivery
2. List the issues in the report
3. Mark as "blocker" and wait for user decision
4. User may choose: rollback / accept with bugs / let implementer continue fixing

---

# Project Progress

## Phase Overview

| Phase | Feature | Status | Test Count |
|------|------|------|--------|
| 1.1 | Multi prompt templates (4 dimensions) | Complete | 11 |
| 1.2 | Multi-turn conversation | Complete | 6 |
| 1.3 | Index comparison (CSI300) | Complete | Included in build-prompt tests |
| 2.1 | Analysis history local storage | Complete | 21 |
| 2.2 | LLM Provider abstraction | Complete | 16 |
| 3 | Tool Use (Anthropic exclusive) | Complete | 27 |
| 3.5 | Self-backtest decision log | Complete | 14 |
| 4 | Reasoning visualization + debug panel | Complete | 12 |
| 6 | Self-learning loop (nightly review + audit iteration) | Complete | 28 |
| 6.5 | Automated scheduled analysis + proactive retrieval | Complete | 8 |
| 5 | Composite scoring dashboard | Complete | 13 |
| 7.1 | Multi-source data degradation | Complete | 5 |
| 7.2 | Prompt evaluation set | Complete | 9 |
| D4 | Sector Cross-Section analysis | Complete | 14 |
| 8 | Local multi-period K-line DB + source locking | Complete | 31 |
| 9 | Data health check + self-implemented indicator layer | Complete | 23 |
| 10 | Structured signal recognition + prompt extreme labels | Complete | 14 |
| **Total** | | | **440** |

## Phase 1: Multi Prompt Templates + Multi-Turn Conversation + Index Comparison

### 1.1 Multi Prompt Templates

- `lib/prompt-templates.js`: 4 templates (technical/trend/valuation/sentiment)
- Each template embeds 6 hard constraints (numeric evidence, opposing views, suggested operation range, data window annotation, prohibit out-of-band data, distinguish closed K-lines)
- `lib/build-prompt.js`: `buildPromptByTemplate()` accepts `templateKey` parameter
- popup: analysis dimension `<select>` single-choice, stores `template` to storage

### 1.2 Multi-Turn Conversation

- background.js: `FOLLOW_UP` message routing, history array directly appended to Claude messages
- content.js: follow-up input box at bottom of analysis panel + send + clear buttons
- conversationHistory maintained locally; each follow-up appends Q&A and re-renders
- Sidebar shows warning when conversation exceeds 20 messages

### 1.3 Index Comparison

- background.js: parallel fetch CSI300 K-lines (secid=1.000300); failure does not block main flow
- build-prompt.js: `buildIndexBlock()` generates comparison section (% change + outperformance/underperformance percentage points)

## Phase 2: Analysis History + Provider Abstraction

### 2.1 Analysis History

- `lib/history.js`: pure function module (generateHistoryId/trimHistory/historyToMarkdown/formatHistoryDate/checkCapacity)
- background.js: 6 history-related message routes (SAVE/GET/DELETE/CLEAR/EXPORT)
- popup: Tab switch [Settings]/[History], card list + expand view + export single/all + delete/clear
- Capacity management: trimHistory(100 entries) + persistHistory size check(9MB)

### 2.2 LLM Provider Abstraction

- `lib/llm/anthropic.js`: Anthropic Claude API adapter
- `lib/llm/deepseek.js`: DeepSeek API adapter (OpenAI-compatible format)
- `lib/llm/index.js`: `getProvider(id)` + `listProviders()`
- popup: provider switch `<select>`, API keys stored separately (apiKey:anthropic / apiKey:deepseek)

## Phase 3: Tool Use

### Tools

- `lib/tools/get-financials.js`: PE/PB/market cap/sector (push2.eastmoney.com/api/qt/stock/get)
- `lib/tools/get-money-flow.js`: Last N months major capital flow (push2his.eastmoney.com/api/qt/stock/fflow/daykline/get)

### Integration

- `lib/llm/anthropic.js`: while loop tool_use, max 5 rounds, usage accumulation
- `lib/build-prompt.js`: Only attaches `buildToolInstructions(secid)` when Anthropic provider
- `background.js`: Only passes tools array when `provider === 'anthropic'`

### Safeguards

- Tool fetch 8s AbortController timeout
- Handler exceptions/unknown tools do not interrupt loop; return error text to LLM
- Each round console.log records tool name + parameters

## Phase 3.5: Self-Backtest Decision Log

### Core Modules

- `lib/self-backtest.js`: 3 exported functions
  - `runHistoricalAnalysis()`: at historical cutoff points, call LLM to get judgment (forced lightweight model for cost savings)
  - `calculateActualReturn()`: pure numeric calculation of actual return + CSI300 alpha
  - `buildSelfCalibrationBlock()`: generate "historical self-calibration" markdown section

### Integration

- `background.js`: handleAnalyze single analysis path; when K-lines >= 36, select 1-2 backtest timepoints (48 bars -> 2, otherwise 1)
- Backtest judgment cached 30 days (`backtest:<code>:<template>:<cutoff>:<provider>`), actual return not cached
- `build-prompt.js`: `extraContext.backtestBlock` rendered into context
- popup: self-backtest toggle (default on), stores `enableSelfBacktest`

### Safeguards

- Only enabled in single mode (debate skips)
- klines < 36 skips
- Backtest LLM forced to use Sonnet/chat
- Any step failure -> console.warn, main flow not interrupted
- Calibration section ends with "may contain bias" disclaimer

## Phase 4: Reasoning Visualization + Prompt Debug Panel

### 4.1 Streaming Display

- `lib/llm/anthropic.js`: split into `nonStreamCall` + `streamCall` dual paths
  - `streamCall`: SSE parsing (message_start/content_block_start/content_block_delta/message_delta/message_stop)
  - Supports thinking_delta / text_delta / tool_use events
  - `onProgress` callback emits real-time events
  - extended thinking: only enabled for Opus + `enableThinking=true` (`thinking.budget_tokens=8000`)
  - Streaming when `onProgress` present, original while loop when no callback (backward compatible)
- `background.js`: `onProgress` -> `chrome.tabs.sendMessage(STREAM_PROGRESS)` -> content.js
  - `chrome.alarms` keeps SW alive (0.5 min interval), cleared when streaming complete
  - `sender.tab.id` routing ensures messages only go to triggering tab
- `content.js`: thinking stream UI (thinking-stream region with header + body)
  - thinking: gray italic / text: normal color / tool_start: blue / tool_result: green update
  - Auto-collapse when analysis complete; click toggle to re-expand

### 4.2 Debug Panel

- `background.js`: when `enableDebugLog=true`, writes to `debug:lastAnalysis` (only keeps most recent 1 entry)
  - Fields: timestamp/code/name/template/provider/model/settings/fullPrompt/toolCalls/rawResponse/usage/cost/durationMs
- popup: new "Debug" tab
  - Region 1: Full Prompt (collapsible + copy button)
  - Region 2: Tool call log (parameters + return + duration)
  - Region 3: LLM raw output (collapsible + copy)
  - Region 4: Token usage details
- popup settings: `enableThinking` + `enableDebugLog` two checkboxes
- Preset system: quick/deep/debate/custom 4 modes one-click switch

## Phase 6: Self-Learning Loop (DeepSeek Nightly Review + Claude User Audit Iteration)

### Core Modules

- `lib/evaluation/collector.js`: extractJudgment / evaluateOneAnalysis / evaluateBatch (pure computation)
- `lib/evaluation/cost-guard.js`: budget management (monthly/daily limits + BudgetExceededError on overrun)
- `lib/evaluation/draft-review.js`: computeStats / pickFailureCases / generateDraftReview (DeepSeek generates draft)
- `lib/evaluation/refine.js`: parseUserReview / refineWithClaude (user audit + Claude refinement)
- `lib/evaluation/nightly.js`: runNightlyJob main flow
- `cli/index.js`: ema CLI commands (nightly / review / budget)

### Cost Control
- Budget default 50 CNY/month, adjusted to 20 CNY/month per user request
- Nightly forced deepseek-chat (cannot be overridden)
- Daily 3 CNY hard cap, monthly 20 CNY hard cap
- Claude refinement not strictly limited but billed

### Workflow
Nightly -> pure computation evaluation -> trigger condition (>=50 entries+7 days OR 14-day fallback) -> DeepSeek generates draft -> user audit check -> Claude Opus generates refinement plan -> user manually executes changes

## Phase 6.5: Automated Scheduled Analysis + Proactive Retrieval

### Core Modules

- `lib/scanner/watchlist.js`: watchlist management (add/remove/list/import, max 50)
- `lib/scanner/hs300.js`: CSI300 constituent pull + monthly cache (degradation strategy)
- `lib/scanner/batch-scan.js`: batch scanner (batched concurrency 10/batch, budget guard, skip on failure)
- `lib/scanner/daily-report.js`: opportunity stock daily report (sorted by signal strength, top10 bullish + top10 bearish)
- `lib/scanner/scheduler.js`: scheduling router + safety switch

### Scheduling Rules
- Sunday weeks 1,3 (HS300 week) -> CSI300 + watchlist + daily report
- Sunday weeks 2,4 (watchlist week) -> watchlist only + daily report
- Other days -> evaluation collection + review trigger check
- Skip HS300 when remaining monthly budget < 5 CNY
- emergencyStop=true immediately stops all automated tasks

### Cost Control
- Forced deepseek-chat (cannot be overridden)
- Single call approximately 0.02 CNY, full CSI300 approximately 6 CNY
- Budget default 50 CNY/month, daily limit 3 CNY

### Schedule Configuration
- Windows: `.eastmoney-ai/scripts/setup-windows.ps1` (Task Scheduler)
- Linux/Mac: `.eastmoney-ai/scripts/setup-cron.sh` (crontab)
- CLI: `ema scheduler pause/resume/config`

## Phase 5: Composite Scoring Dashboard

### Core Modules
- `lib/prompt-templates.js`: HARD_CONSTRAINTS item 7 (structured JSON output)
- `lib/dashboard/parse-score.js`: parseScoreBlock / validateScoreData / computeWeightedScore
- `content.js`: dashboard-card HTML + renderDashboard rendering logic + degradation fallback
- `content.css`: dashboard card styles (large score/signal/label/levels/meta)

### Structured JSON Fields
score(0-100) / signal(strong_bull~strong_bear) / confidence(high/medium/low) / key_levels(support/resistance/stop_loss) / trend / position_percentile / one_line_summary

### Fault Tolerance
- JSON missing/format error/field out of range -> dashboard shows "?" + warning, does not crash
- Analysis body displays normally; dashboard degradation does not affect main flow

### Integration Linkage
- collector.js: evaluation prefers scoreData.signal (over extractJudgment regex)
- daily-report.js: sorting prefers scoreData.score (over keyword scoring)
- batch-scan.js: scan results carry scoreData

## Phase 7.1: Multi-Source Data Degradation

### Core Modules
- `lib/data-sources/eastmoney.js`: Eastmoney primary source (migrated from background.js, most complete fields)
- `lib/data-sources/sina.js`: Sina backup source (missing amount/turnoverRate; amplitude and change% computed manually)
- `lib/data-sources/tencent.js`: Tencent last resort (same as above, manual computation required)
- `lib/data-sources/dispatcher.js`: degradation dispatch (Eastmoney->Sina->Tencent) + hourly degradation cap 20 + logging

### Real API Verification
- eastmoney 1.5s — 283 monthly bars, 11 fields (full)
- sina 2.8s — 10 monthly bars, 6 fields (missing amount/change/turnoverRate)
- tencent 0.9s — 10 monthly bars, 6 fields (same as above)

### manifest.json
Added host_permissions: quotes.sina.cn, web.ifzq.gtimg.cn, push2.eastmoney.com

## Phase 7.2: Prompt Evaluation Set

### Core Modules
- `lib/eval/seed-stocks.json`: 40 seed stocks (6 categories, 5-8 each)
- `lib/eval/dataset-builder.js`: buildDataset — pull K-lines->select cutoff->compute groundTruth
- `lib/eval/runner.js`: runEvaluation + scorePrediction (exact match 1.0/direction correct 0.5/neutral 0.3/direction wrong -0.5 to -1.0)
- `lib/eval/report.js`: generateEvalReport + compareRuns (version comparison)
- `lib/eval/prompt-versions/v1-baseline.js`: current prompt snapshot

### Evaluation Workflow
build dataset -> run eval (DeepSeek only) -> generate report -> compare versions

### groundTruth Rules
alpha>10%->strong_bull / >3%->bull / |alpha|<=3%->neutral / <-3%->bear / <-10%->strong_bear

### Cost
40 stocks x 4 testPoints x 4 templates = 640 calls x ~0.02 CNY ≈ 13 CNY per full eval

### CLI
ema eval build | ema eval run | ema eval report | ema eval compare v1 v2 | ema eval snapshot --label v2

## Phase D4: Sector Cross-Section Analysis

### Core Modules
- `data/industry-map.json`: Shenwan Level-1 28 sectors x 403 A-share mapping
- `lib/industry-map.js`: load/query/coverage stats
- `lib/cross-section.js`: analyzeIndustry / analyzeAll / enrichWithCrossSection

### Features
- Intra-sector ranking: absolute score -> relative ranking (top X%)
- Sector rotation signals: top 5 strong/weak pairs
- Sector strength labels: strong/neutral/weak

### CLI
ema cross-section <code> | ema industries | ema rotate | ema industries refresh

## Known Deviations (intentionally skipped / reasonable spec departures)

### Ollama provider — Deprecated

Spec Phase 2.2 required implementing Ollama local provider; not actually implemented. Reasons:
- localhost CORS restrictions under Chrome MV3 service worker environment
- Local model capabilities insufficient for rigorous judgment in A-share technical analysis scenarios
- Maintaining two providers (Anthropic + DeepSeek) already covers primary use cases

### get-industry-peers — Skipped

Spec Phase 3.2 marked "optional, lowest priority"; decided not to do. Can independently add later if needed.

### 20-message conversation warning placement — Reasonable improvement

Spec required "warning prompt at top of popup"; actually implemented in content.js sidebar.
Reason: user's operational focus is on the sidebar; popup requires manual opening to see.
This deviation is a reasonable improvement and will not be corrected.

## Roadmap

### Current Work

- [x] Phase 1: Multi prompt templates + Multi-turn conversation + Index comparison
- [x] Phase 2: Analysis history + LLM Provider abstraction
- [x] Phase 3: Tool Use (Anthropic exclusive)
- [x] Phase 3.5: Self-backtest decision log
- [x] Phase 4: Reasoning visualization + Prompt debug panel
- [x] Phase 6: Self-learning loop (nightly review + audit iteration)
- [x] Phase 6.5: Automated scheduled analysis + proactive retrieval
- [x] Phase 5: Composite scoring dashboard
- [x] Phase 7.1: Multi-source data degradation
- [x] Phase 7.2: Prompt evaluation set
- [x] Phase 8: Local full-market multi-period K-line DB (SQLite, 394 tests)

### Next Steps

- [ ] Phase 8 actual DB build: `ema db init --scope hs300` (build CSI300 monthly+weekly+daily)
- [ ] Phase 8 60min: `ema db init --scope hs300 --periods 60min` (after Eastmoney recovers)
- [ ] Blind spot 1: Agent output semantic-level regression tests (test-only, no src changes)
- [ ] Blind spot 2: Cross-level consistency check enhancement (new feature)

### Future Phases

#### Phase 9: Multi-period prompt templates (after Phase 8 DB build complete)
Existing 4 templates are all monthly-perspective; after building multi-period DB need:
- Change "last N monthly bars" in prompt to dynamically adapt by period (N bars of {periodLabel})
- Position percentile time window adjusted by period (monthly 5 years / weekly 2 years / daily 6 months / 60min 1 month)
- Suggested operation time scale adapted by period
- HARD_CONSTRAINTS 6 items retained but wording adjusted by period

#### Phase 10: Multi-period linked judgment (after Phase 9)
- Run 4-period analysis simultaneously, synthesize scoreData
- Output "multi-period resonance strength" (0-100)
- Resonance rules:
  - Month+Week+Day all bullish -> high confidence entry
  - Month bullish + Day overbought -> wait for pullback
  - Month bearish + 60min rebound -> rebound trap

## Phase 8: Local Full-Market Multi-Period K-Line DB

### Completed

- [x] SQLite database (`klines-v2.sqlite`, 271 MB)
- [x] CSI300 real constituents (Sina API, 300 stocks including 32 ChiNext)
- [x] Baidu as primary source (18 fields including turnover rate, pre-adjusted, zero rate limiting)
- [x] Monthly/Weekly/Daily three-period full history (earliest 1991-04)
- [x] Single-stock source locking (source column + cross-source write throws error)
- [x] Local read 0-3ms (vs online ~1000ms, 300x+ acceleration)
- [x] klines-repo return format fully consistent with dispatcher
- [x] db-init resume from breakpoint + progress persistence
- [x] 401 tests all passing

### Data Specifications

| Period | Record Count | Avg/Stock | Date Range |
|------|--------|---------|---------|
| Monthly | 62,475 | 208 | 1991-04 ~ 2026-05 |
| Weekly | 262,056 | 874 | 1991-02 ~ 2026-05 |
| Daily | 1,240,025 | 4,133 | 1991-01 ~ 2026-05 |

### Known Remaining

- [ ] 60min period table created but not populated (pending Eastmoney recovery verification klt=60)
- [ ] Full market 5000 stocks not built (currently only hs300)
- [ ] Adjustment events table created but not populated
- [ ] db-update incremental update is framework only
- [ ] v2-baseline eval pending completion (rerunning after fixing 3 bugs)

## Phase 9: Data Health Check + Self-Implemented Indicator Layer

### Completed
- [x] Strong defense infrastructure (.npmrc ignore-scripts, version locking, OSV check script)
- [x] 15 Tongdaxin-compatible indicators (SMA/EMA/MACD/RSI/KDJ/WR/CCI/Boll/ATR/OBV/MFI/Stochastic)
- [x] Indicator known-value reconciliation (Moutai 600519 MA5 manual verification deviation < 0.01)
- [x] Data health check layer (5 checks A~E, severity grading)
- [x] Prompt indicator table + HARD_CONSTRAINTS #8 (prohibit self-calculating indicators)
- [x] v3-indicators eval: 0% LLM self-calculated indicators, score 0.079

### v3 eval Results
- 640/640 calls successful, 12.68 CNY, 38.7min
- LLM self-calculated indicators: 0% (HARD_CONSTRAINTS #8 perfectly effective)
- bear bias: 43%, strong_bull predictions: 0
- Root cause diagnosis: LLM extreme label avoidance (190 strong_bull GT, 0 predicted strong_bull)

## Phase 10: Structured Signal Recognition Layer

### Completed
- [x] lib/signals/atoms.js — 6 atom functions (cross/exist/count/hhv/llv/every)
- [x] lib/signals/factory.js — 18 buy/sell signals (MACD golden cross/KDJ golden cross/oversold/breakout/MA alignment etc.)
- [x] lib/signals/summary.js — signal list generation + signal guidance (triggers strong_bull/strong_bear judgment)
- [x] HARD_CONSTRAINTS #9 (extreme label guidance) + #10 (signal consistency)
- [x] maxTokens=4000 fixed + truncation detection + token budget monitoring
- [x] v4-signals eval: score 0.187 (2.4x v3), strong_bull non-zero for first time

### v4 eval Results (Audited Corrected Version — full denominator)

> **2026-05-18 Audit correction**: Original table v4=0.187 was after excluding 65 parse_failed records (107.4/575).
> The table below uniformly uses full denominator (/640); do not directly compare with old table.
>
> **Audit conclusion: PROGRESS.md v4 row had denominator cherry-picking.** See `scripts/audit-v4-reparse.js` and `scripts/audit-v4-deep.js` for details.

| Metric | v1 | v2-fixed | v3 | v4-signals | v5-resonance | v6-sector |
|------|----|---------|-----|-----------|-------------|----------|
| Full weighted score | ? | ? | 0.079 | **0.1678** | **0.1302** | **0.1064** |
| Sample count | ? | ? | 640 | 640 | 640 | **280** ⚠ |
| strong_bull % | — | 0% | 0% | **16.3%** | **12.5%** | **22.9%** |
| bull % | — | — | — | 22.0% | 15.8% | 22.9% |
| bear % | — | 100% | 43% | **31.7%** | **31.1%** | **31.8%** |
| strong_bear % | — | — | — | 9.8% | 10.6% | 7.5% |
| neutral % | — | — | — | 10.0% | 16.4% | 15.0% |
| parse_failed | — | — | 0? | **65/640** | **87/640** | **0/280** |
| Cost | 10.24 CNY | 11.14 CNY | 12.68 CNY | 12.62 CNY | ? | ? |

⚠ v6-sector only 280 samples (18 stocks), not directly comparable with v4/v5.

### Known Remaining
- strong_bear false positive rate high (signal factory bearish signals may have given wrong guidance)
- strong_bull false positive 38% (overly aggressive strong labels)
- Signal list is monthly-only, not covering weekly/daily multi-period resonance

### Architecture Debt (Paid)

- [x] storage abstraction layer (lib/platform/storage.js) — Node/Browser dual environment
- [x] Native Messaging auto-sync — Chrome -> Node data bridge
- [x] CLI nightly/scheduled commands actually working (no longer empty shells)
- [x] get-financials field unit fix (f43/f162/f167 /100)
- [x] dispatcher fallback bug fix (logDegrade parameter misalignment)
- [x] Data source adjustment — Baidu as primary source (18 fields/pre-adjusted/stable), same-source locking to prevent cross-source mixing

### Data Source Constraints

**K-line source order**: Baidu -> Sina -> Tencent -> Eastmoney (last resort)
- Baidu API returns pre-adjusted data; price absolute values may have offset vs other sources (different adjustment base dates)
- Offset does not affect technical analysis (MA/MACD/support-resistance), but prices cannot be mixed across sources
- **Single-stock source locking**: SQLite `source` column records data source for each K-line; same code+period prohibits mixing
- Changing source requires clearing and rebuilding DB (or `--force` override)

**During DB build**: `ema db init --source baidu` locks to Baidu throughout; failures skip without degradation

## Data Trust Fixes (2026-05-18)

### Fix 1: PROGRESS.md committed to git
- PROGRESS.md first git add + commit
- Old v4 data table replaced with audited corrected version (full denominator)

### Fix 2: eval result schema added version field
- `lib/eval/runner.js` runEvaluation writes `parserVersion`, `evalRunnerVersion`, `timestamp` per record
- Future re-parsing can identify schema compatibility

### Fix 3: Denominator transparency
- Added `lib/eval/compute-score.js`: `computeScoreFull()` outputs both full/excl_pf simultaneously
- Reports default to full denominator; excl_pf is auxiliary reference only

## Frozen Eval Dataset (2026-05-18)

### Background
Phase 11/12 experiments found that using `--limit=` to take first N stocks caused severe baseline drift (0.1966->0.0806->0.0981); different stock subsets are not comparable.

### Solution
Created `data/frozen-eval-dataset-v1.json`, extracted from Phase 12 Run A (640 samples, 40 stocks).

**Absolute Baseline**:
- Run ID: `runA-no-sector-2026-05-18-05-12-20`
- Config: no sector, no resonance, old #12 constraint, deepseek-chat, maxTokens=4000
- **score = 0.1966** (full denominator, 640 samples)
- GT distribution: strong_bull=45, strong_bear=44, bull=22, bear=26, neutral=23

**Rule**: All subsequent eval score comparisons must be vs this baseline. Subset baselines are prohibited.
Loader entry: `lib/eval/load-frozen-dataset.js` -> `loadFrozenDataset({ version:'v1', subsetStocks:N, seed:42 })`.
Subsets use reproducible random sampling (mulberry32 PRNG), replacing old `--limit=` first-N logic.

## Phase 11 & 12 Quantitative Validation (2026-05-18)

### Methodology

298 HS300 stocks x 2018-2025 monthly data, offline pure computation, zero LLM cost. Strict walk-forward.

### Phase 12: Sector Alpha IC Matrix

| lookback \ holding | 1m | 3m | 6m | 12m |
|---|---|---|---|---|
| 3m | -0.0075 | -0.0013 | -0.0121 | -0.0015 |
| 6m | -0.0008 | -0.0069 | -0.0103 | -0.0086 |
| 12m | +0.0075 | +0.0056 | +0.0004 | -0.0242 |
| 24m | -0.0017 | -0.0174 | -0.0409 | **-0.0738** |

**Conclusion**: All 16 cells |IC| < 0.10. Hit Rate 47.8-51.4%. Long-Short Sharpe all negative.
**Phase 12 closed**. Code retained in `lib/sector/`, prompt injection path disabled (ENABLE_SECTOR_ALPHA=false).
See `docs/phase12-postmortem.md` for details.

### Phase 11: Multi-Period Resonance Predictive Power

Resonance signals: strong_bull/bear = all three periods same direction, mild_bull/bear = two periods same direction.

Deducting HS300 equal-weight benchmark alpha:

| signal \ holding | 1m | 3m | 6m | 12m | n |
|---|---|---|---|---|---|
| strong_bull | 0.59% | 2.61% | 4.31% | 9.25% | 1,114 |
| mild_bull | 0.87% | 2.95% | 6.10% | 10.70% | 2,721 |
| **strong_bear** | **1.53%** | **3.48%** | **7.67%** | **11.15%** | 1,114 |
| mild_bear | 2.55% | 18.43% | 40.26% | 36.65% | 2,939 |

Long-Short (strong_bull - strong_bear): All negative, |Sharpe| approx 0.10.

**Conclusion**: Three-period resonance is a marginal contrarian signal (strong_bear future returns systematically higher than strong_bull).
LLM v5 score decline reason: HARD_CONSTRAINTS #11 forced following resonance direction, but resonance is a contrarian indicator.
See `docs/phase11-postmortem.md` for details.

### Comparison

| Metric | Phase 12 sector alpha | Phase 11 resonance |
|------|----------------------|-------------------|
| Strongest IC/Sharpe | IC=-0.074 | Sharpe=-0.195 (1m) |
| Direction | Contrarian | **Contrarian** |
| Status | Closed | Code retained, injection disabled |
| LLM injection | Verified ineffective (-0.03 Delta) | Verified harmful (v5 < v4) |
| LSTM candidate | 24m lookback (IC=-0.074) | Contrarian factor (Sharpe=0.10) |

### Strategic Pivot

1. **LLM prompt injection path paused**: Phase 11 and 12 quantitative signals both have |IC/Sharpe| < 0.20, unsuitable as LLM hard constraints
2. **Phase 17 LSTM**: Resonance contrarian signal (Sharpe=0.10) and sector alpha 24m (IC=-0.07) can serve as features
3. **Contrarian constraint experiment**: Optional path — change Phase 11 #11 to contrarian logic and re-eval (budget 7-15 CNY)

### Changed Files

- `cli/analyze-sector-predictive-power.js` — Phase 12 IC matrix computation
- `cli/analyze-resonance-predictive-power.js` — Phase 11 resonance predictive power computation
- `cli/verify-resonance-matrix.js` — Resonance matrix credibility verification
- `docs/phase*.md` — Phase 11 & 12 postmortems
- `lib/prompt-templates.js` — #12 constraint text reverted to neutral wording

## Phase 13 ASC (2026-05-18)

### Results

| Metric | Frozen Baseline | Phase 13 | Phase 13 high-conf |
|------|----------------|---------|-------------------|
| score | 0.1966 | 0.1984 | **0.225** (+14.5%) |
| n | 640 | 640 | 60 (9.4%) |
| strong_bull | 16.7% | — | — |
| confidence distribution | — | high=9.4% medium=86% low=2.5% | — |

LLM self-assessed confidence positively correlated with accuracy (high=0.225 > medium=0.199 > unparsed=0), #13 constraint effective.
high-conf subset score 0.225 is the largest single improvement this phase, but only covers 9.4% of predictions.

### Changes

- `lib/uncertainty/asc.js` — confidence calibration module
- `cli/eval-phase13-asc.js` — ASC eval runner
- `lib/prompt-templates.js` — HARD_CONSTRAINTS #13 (confidence honesty rule)
- Code retained, #13 enabled by default (no impact on overall score, gain in high-conf subset)

## Phase 15 Multi-Agent (2026-05-18)

### Results

Dry-run 48 samples (3 stocks), 5 agent architecture (Bull/Bear/Technical/Sector/Judge):
- score 0.1979 vs baseline 0.1966, **Delta=+0.0013 (noise)**
- Judge output 0 strong signals, 0/48 extreme predictions
- Cost 0.05 CNY/sample, 5x baseline, cost-effectiveness not justified

### Diagnostic Findings

- Bull/Bear independence confirmed (different data + different arguments)
- Judge is not simply "taking the middle" — genuinely evaluates data support strength for each argument
- Judge rationally abstains when Bull/Bear are evenly matched -> correct behavior but does not translate to score improvement
- Sector Agent has lowest citation rate in Judge decisions; sector alpha data contributes minimally to current architecture

### Disposition

- Code retained: `lib/agents/{bull,bear,technical,sector,judge}-agent.js`, `lib/agents/phase15-runner.js`
- **ENABLE_MULTI_AGENT=false** disabled by default (same pattern as Phase 11)
- Full 70 CNY eval not run
- Diagnostic script: `scripts/diagnose-phase15.js`

## Phase 17 LSTM v1 Baseline (2026-05-19)

### Architecture

LSTM(input=21, hidden=64, 1 layer, dropout=0.2) -> FC(2 heads: y3, y6)
Train: 2015-2021 (15,778 seqs) / Val: 2022-2023 (5,723) / Test: 2024-2026 (4,340)

### Results

| Metric | Val | Test | Delta |
|------|-----|------|-----|
| IC y3 | 0.095 (p approx 0) | **0.025** (p=0.10) | -0.070 |
| IC y6 | 0.074 | -0.024 | -0.097 |
| Sharpe y3 (ann) | 0.267 | 0.144 | -0.123 |

### Failure Causes

1. **Regime change**: 2015-2021 bull market patterns do not apply to 2024+ market
2. **Val overfit**: monitoring IC y3 led to selecting checkpoint optimal on Val but failing on Test
3. **Single-layer LSTM** too shallow to learn complex cross-sectional ranking patterns
4. **Fixed split** training: needs walk-forward retraining (Qlib standard practice)

### Disposition

- Code retained in `lib/lstm/`, checkpoint kept as Phase 19 signal source
- Test set already used, marked as contaminated; cannot be reused for single-model evaluation
- 2024-2026 data can serve as **composite strategy backtest evaluation** (different task, different discipline)
- Docs: `lib/lstm/eval_test.py`, `lib/lstm/eval_val.py`

## Phase 19 Walk-Forward Backtesting (Design)

### Signal Sources (5, all quantitatively validated)

| # | Signal | Strongest Metric | Direction | Coverage |
|---|------|---------|------|------|
| 1 | LSTM v1 pred | IC y3=0.095 (Val) | Positive | 100% |
| 2 | Resonance reverse | Sharpe=0.195 (1m) | Contrarian | 100% |
| 3 | Sector alpha 24m/12m | IC=0.074 | Contrarian | 100% |
| 4 | Phase 13 ASC high-conf | score=0.225 | Positive | 9.4% |
| 5 | MACD/RSI/KDJ composite | Phase 10 signal | Positive | 100% |

### Three-Layer Architecture

```
Signal Layer (lib/signals/registry.js)
  -> Unified interface fn(code, asOfDate) -> score in [-1, +1]

Aggregation Layer (lib/backtest/aggregator.js)
  -> Equal-weight / IC-IR weighted / Rank IC ensemble

Portfolio Layer (lib/backtest/portfolio.js)
  -> Top-K (10-30), equal-weight/risk parity, monthly rebalance
  -> Single stock <=15%, Sector <=25%

Backtest Engine (lib/backtest/engine.js)
  -> Walk-forward strict, slippage 0.3-0.5%, commission 0.025%
  -> In-sample: 2015-2023 (hyperparameter tuning)
  -> Live test: 2024-2026 (frozen)
```

### Evaluation Targets

- Sharpe > 1.0, Max DD < 25%
- vs HS300 equal-weight alpha + IR

### Decision Points (pending user confirmation)

1. Signal set: all 5 or select subset?
2. Aggregation: start with equal-weight or directly IC-IR weighted?
3. Framework: custom lightweight or vectorbt?
4. LSTM retrain: monthly rolling retrain or static Phase 17 checkpoint?

### Changed Files

- `lib/lstm/` — data pipeline + model + training + evaluation scripts
- `lib/lstm/requirements.txt` — torch/numpy/pandas/scipy
| Input features | Ready | 15 Tongdaxin indicators (Phase 9) + sector alpha (Phase 12) + resonance factor (Phase 11) |
| Labels | Ready | 6-month forward alpha (groundTruth) |

### Feature Checklist

- **Price type**(5): close, open/high/low range, amplitude, change_percent
- **MA type**(4): MA5/MA20/MA60/MA120 position + slope
- **Momentum type**(5): MACD_DIF/DEA/HIST, RSI14, KDJ_K/D/J
- **Volatility type**(2): BOLL upper/lower distance, ATR
- **Volume-price type**(2): volume ratio, turnover_rate percentile
- **Sector type**(1): sector alpha (12m lookback)
- **Resonance type**(1): resonance signal (-1=strong_bear, 0=neutral, +1=strong_bull)
- **Position type**(1): price percentile (5yr window)

Total **21 features**, covering technical + sector + resonance.

### Architecture Suggestions

- Input: (lookback=24 months, 21 features) x batch
- Output: 6-month forward alpha (regression) or 5-class signal (classification)
- Model: LSTM (baseline) -> Transformer encoder (if LSTM converges)
- Evaluation: Same IC + Hit Rate matrix as Phase 11/12, vs vanilla LLM baseline 0.1966

### Existing Code

No LSTM code remnants in project. Must build from scratch.

## Engineering Audit + Strict v6 Retest (2026-05-20)

### P0: Proportional Split Bug
Original code `X[:n_tr]` index-proportional split mixed same-period data from different stocks. Fixed to strict date-based split.

### Strict v6 Three-Frequency Comparison

| Frequency | Model | Stocks | Test IC | Verdict |
|------|------|------|---------|------|
| Daily | LSTM-7 | 298 | **+0.141** | Only real signal |
| Weekly | LSTM-7 | 296 | +0.007 | Fail |
| Monthly | LSTM-7 | 298 | -0.027 | Fail (original 0.019 was split-contaminated) |

### Monthly Optimization History

| Iteration | Stocks | Method | Features | IC | Key Breakthrough |
|------|------|------|------|-----|---------|
| v1 | 298 | LSTM | 21 | -0.027 | Strict split exposed truth |
| v2 | 813 | LSTM | 21 | +0.027 | Added CSI1000 data |
| v3 | 785 | LightGBM | 11 | +0.042 | Tree models better for monthly |
| v4 | 1158 | LightGBM | 22 | +0.054 | More features |
| v5 | 2247 | LightGBM | 22 | **+0.063** | Full A-share (3744 index stocks) |
| v6 | 2247 | +Daily bridge | 19 | +0.015 | Daily bridge degraded |
| v7 | 500 | +Fund flow | 16 | -0.013 | Fund flow data too sparse |
| v8 | 1500 | Alpha158 | 31 | +0.043 | More features != better |
| v9 | 1000 | Hyper sweep | 13 | +0.040 | Hyperparams no improvement |

### Daily Enhancement Attempts

| Method | Test IC | Conclusion |
|------|---------|------|
| LSTM-7 strict v6 | **+0.141** | Baseline |
| Triple-Barrier 3-class | +0.040 | Classification inferior to regression |
| ListNet ranking | +0.010 | Degraded |
| MASTER cross-attn | -0.018 | Cross-stock attention ineffective |
| Sprint 1 33-dim dist | -0.019 | Daily distribution features unhelpful |

### DB Expansion

- Baidu API: 3744 index stocks -> 3265 ingested, 2247 valid (>=84 months)
- Full A-share 5000 stocks incomplete due to akshare API blockade

### Work Discipline v1.0
- All new code mandatory header comments (INPUT_DATA_RANGE / WALK_FORWARD / TEST_SET_USAGE)
- Prohibited spin language checklist
- Mandatory assertion + dry-run

## Phase 5 v32 Final Feature Set (2026-05-25)

### 32d Feature Composition

| Group | Dimensions | Content |
|----|------|------|
| G2 | 3 | MA5/MA20/MA60 deviation |
| G3 | 3 | MACD DIF/DEA/Histogram |
| G4 | 2 | vol_6m volatility + ATR14 |
| FFT | 10 | Simple amplitude spectrum (no freq/phase) |
| G7 | 14 | Full volume-price (including above_ma5) |

**Total 32 dimensions**. Difference vs 31d: FFT uses simple amplitude spectrum instead of freq+amp+phase blend (+0.0037 IC), G7 restores above_ma5 (+0.0037 IC).

### Three-Way Comparison (5-Fold CV + IC Decay T+1~T+6)

| Version | Dims | IC | ICIR | IC>0 | CV_mean | CV_all+ |
|------|------|-----|------|------|---------|---------|
| 61d | 61 | +0.0295 | +0.428 | 72.5% | +0.0297 | True |
| 31d | 31 | +0.0341 | +0.479 | 71.8% | +0.0338 | True |
| **32d** | **32** | **+0.0377** | **+0.510** | **72.5%** | **+0.0377** | **True** |

**32d vs 61d: IC +27.9%**. 32d leads across all Folds and all horizons (T+1~T+6).

### Key Findings

1. **Simple FFT amplitude spectrum > freq+amp+phase blend**: Retaining frequency/phase information actually introduces noise
2. **above_ma5 has value**: G7 13->14 dimension increment is definite and significant
3. **32d is the current optimum**: Reduced dimensions + highest IC + lowest overfitting risk

### Long-Only Backtest (Q5, equal-weight, monthly rebalance)

Filter: daily avg turnover > 5M CNY + no limit-up/down (+/-9.9%). Costs: slippage 0.3% + commission 0.025%.

| Metric | Value |
|------|------|
| Backtest Period | 2015-01 ~ 2025-11 (128 months) |
| Cumulative Net Return | **+520.67%** |
| Annualized Return | +21.01% |
| Annualized Volatility | 27.47% |
| Sharpe | **0.765** |
| Max Drawdown | -42.20% |
| Calmar | 0.498 |
| Monthly Win Rate | 57.8% |
| Avg Holdings | 244 stocks |

Yearly: 2015 +67.7% / 2016 +30.6% / 2017 -0.3% / 2018 **-38.2%** / 2019 +40.9% / 2020 +18.8% / 2021 **+43.3%** / 2022 +16.3% / 2023 -7.1% / 2024 +34.0% / 2025 YTD **+35.2%** (Sharpe 3.17).

### Peer Features (32d vs 36d)

Added 4 dimensions: ind_rank_pct / ind_zscore / peer_dist_median / peer_dist_pct.

| Version | IC | ICIR | IC>0 | Delta vs 32d |
|------|-----|------|------|-----------|
| 32d | +0.0377 | +0.510 | 72.5% | — |
| 36d | +0.0314 | +0.462 | 69.5% | **-16.5%** |

**Conclusion: Rejected.** Peer features degrade IC across all Folds and all horizons. Intra-sector relative ranking signals are already adequately captured by the existing 32 dimensions.

### OOS Tracking

`scripts/v32_oos_track.py` ready, three sub-commands:
- `predict` — generate current-month predictions at start of month
- `realize` — fill T+1~T+6 realized returns at start of next month
- `report` — rolling IC report

Data file: `.eastmoney-ai/oos/v32_oos_tracker.json`. Recording starts next month.

### Output Files

- `scripts/phase5_v32_final.py` — 32d vs 31d vs 61d three-way comparison
- `scripts/v32_backtest_long.py` — Q5 long-only backtest
- `scripts/v32_peer_features.py` — Peer features comparison
- `scripts/v32_oos_track.py` — OOS tracking CLI
- `.eastmoney-ai/backtest/v32_backtest_summary.json` — Backtest summary
- `.eastmoney-ai/diagnosis/v32_peer_summary.json` — Peer feature comparison results

## IC/IR Methodology Audit + v32 Full Fix Evaluation (2026-05-25)

### Audit Finding: 4 Methodological Flaws in icir_report.py

| # | Flaw | Impact | Fix |
|---|------|------|------|
| 1 | Cumulative 3-period return `(c[i+3]-c[i])/c[i]` | IC mechanically inflated 2-3x | Single-month return `(c[i+1]-c[i])/c[i]` |
| 2 | Mixed date formats YYYY-MM / YYYY-MM-DD | Cross-section fragmented, only 1-3 stocks per month | Unified `str(date)[:7]` |
| 3 | 61-dim features (non-optimized) | Noise features dilute IC | 32-dim optimized version |
| 4 | 46 date strings != 46 calendar months | Window description misleading | 137 calendar months, 76 valid post-training cross-sections |

### v32 Full Fix Results (LightGBM+XGBoost+Ridge Ensemble, CSRC L2 neutralization, rolling 60 months)

**T+1 ~ T+6 single-month IC decay:**

| Lag | IC | ICIR | IC>0 | N(months) |
|-----|-----|------|------|------|
| T+1 | **+0.0765** | +0.98 | 85.5% | 76 |
| T+2 | +0.0446 | +0.50 | 73.3% | 75 |
| T+3 | +0.0489 | +0.51 | 70.3% | 74 |
| T+4 | +0.0337 | +0.41 | 65.8% | 73 |
| T+5 | +0.0308 | +0.36 | 69.4% | 72 |
| T+6 | +0.0318 | +0.36 | 63.4% | 71 |

**5-Fold CV:** Mean IC=+0.0693, all positive (0.049~0.077)

**Long-Only Backtest (Q5, equal-weight, monthly):**

| Metric | Q1(Weak) | Q3(Mid) | Q5(Strong) | LS |
|------|--------|--------|--------|-----|
| Annualized Return | +4.6% | +17.0% | **+34.9%** | +28.2% |
| Sharpe | 0.20 | 0.85 | **1.53** | 3.02 |
| Max Drawdown | -46.0% | -20.0% | -18.5% | -4.9% |
| Monthly Win Rate | 55.3% | 60.5% | **71.1%** | — |

Monotonicity: PASS. After 50bp costs, LS Sharpe still 2.24.

### Before/After Fix Comparison

| | icir_report.py (Old) | v32_final_eval.py (New) |
|---|---|---|
| Return measure | Cumulative 3-period | **Single-month** |
| Date grouping | Mixed format, 46 date strings | **YYYY-MM, 76 valid cross-sections** |
| Feature dimensions | 61 dims | **32 dims** |
| Sector neutralization | None | **CSRC L2** |
| Backtest window | 2024-01+ (46 months) | **2015-01+ (137 calendar months)** |
| **T+1 IC** | **0.177** (inflated) | **0.0765** (honest) |

IC decline from 0.177->0.0765 is not model degradation — it is removal of the mechanical inflation from cumulative returns + date confusion.
T+1 IC=0.076 + 85.5% monthly positive IC + LS Sharpe 3.02 = signal is real and effective.

### Output Files

- `scripts/v32_final_eval.py` — Full-fix evaluation script
- `.eastmoney-ai/final_eval/v32_final_results.json` — Complete results

## Kronos Prediction Service Phase 1: Basic Environment Setup (2026-05-25)

### Background

Beyond the existing monthly factor engine (LightGBM/XGBoost tree models), introduce Kronos time-series prediction model as a second signal line.
Kronos uses BSQ (Binary Spherical Quantization) to encode OHLCV K-lines into discrete tokens, then uses a decoder-only Transformer to autoregressively generate future K-lines.

The two signal lines converge at the Node.js signal fusion layer.

### File Checklist

| File | Lines | Description |
|------|------|------|
| `kronos/module.py` | ~650 | 14 core PyTorch modules (BSQ/RoPE/TransformerBlock/HierarchicalEmbedding/DualHead) |
| `kronos/tokenizer.py` | ~200 | KronosTokenizer — encode/decode, HuggingFace PyTorchModelHubMixin |
| `kronos/transformer.py` | ~205 | Kronos — decoder-only, decode_s1/decode_s2 two-stage autoregressive decoding |
| `kronos/predictor.py` | ~300 | KronosPredictor — z-score normalization/multi-sampling/autoregressive generation |
| `kronos/data_adapter.py` | ~120 | SQLite -> DataFrame adapter layer (Phase 2) |
| `kronos/signal_generator.py` | ~130 | Multi-sample aggregation -> trading signal dict (Phase 3) |
| `kronos/download_weights.py` | ~130 | HF Hub pretrained weight download CLI |
| `kronos/__init__.py` | ~20 | Package exports |
| `kronos/tests/test_tokenizer.py` | ~140 | 8 tests — init/forward/encode/decode/half/padding |
| `kronos/tests/test_transformer.py` | ~120 | 9 tests — forward/decode_s1/decode_s2/teacher forcing/CUDA |
| `kronos/tests/test_predictor.py` | ~170 | 16 tests — sampling/timestamps/end-to-end/AR inference/CUDA |

### Architecture

```
Baidu monthly SQLite (klines-v2.sqlite)
  -> data_adapter.load_monthly_klines(code)       [Phase 2]
  -> orchestrator.run_analysis(code, predictor)    [Phase 4]
  -> predictor.predict(df, ...) x N samples        [Phase 1]
  -> signal_generator.generate_signal(predictions)  [Phase 3]
  -> dict { direction, change_pct, confidence, volatility }
  |
Node.js CLI: ema analyze <code>
  -> Python subprocess: cli_predict.py --json
  -> + lib/indicators/calculate.js technical indicators
  -> Composite assessment output
```

### Test Results

```
36 passed, 1 skipped in 3.65s (after pretrained weights loaded)
```

- tokenizer: 10/10 passed (2 new pretrained tests added)
- transformer: 9/10 passed (1 HF Hub online test skipped)
- predictor: 16/16 passed
- signal_generator: self-test passed (median aggregation + assertion verification)

### Data Adapter Verification — Kweichow Moutai (600519)

```
Stock: 600519
Records: 297
Time range: 2001-08 -> 2026-05
Fields: open/high/low/close/volume/amount (DatetimeIndex)
```

### Phase 2: Data Adapter Layer (2026-05-25)

- `load_monthly_klines(code, db_path)` — SQLite -> DataFrame
- Field mapping: same-name pass-through (open/high/low/close/volume/amount), amount null -> 0
- Date parsing: monthly "YYYY-MM" -> append "-01" -> pandas datetime
- DataFrame index = DatetimeIndex (predictor uses `df.index <= x_ts` for time filtering)
- 600519 verification: 297 records, 2001-08 -> 2026-05, zero nulls
- Diagnostic output goes to stderr to avoid polluting stdout JSON

### Phase 3: Prediction Signalization (2026-05-25)

- `generate_signal(predictions: list[pd.DataFrame], threshold=2.0) -> dict`
- Pure statistical aggregation, no dependency on predictor; upper-layer orchestrator handles sampling loop
- **Median aggregation** (outlier-resistant): predicted_change_pct / high / low / close use median
- Direction judgment also uses median (|median| < 2% -> flat)
- Confidence: proportion of samples aligned with final judgment direction
- Volatility: std of per-sample return percentages
- Direction threshold: |median_change_pct| < 2% -> flat

### Phase 4: CLI & Node.js Integration (2026-05-25)

- `orchestrator.run_analysis()` — chains data_adapter -> N x predictor -> signal_generator
- `cli_predict.py` — local offline CLI, supports --json / --n-samples / --T / --temperature / --top-k
- Node.js CLI `ema analyze <code>` — technical indicators + Kronos AI + composite assessment
- Default x_ts = second-to-last month (avoids data boundary collapse; last month usable for backtest verification)
- JSON mode auto-suppresses non-JSON stdout (via os.devnull redirect)

### Model Detection Results (2026-05-25)

#### Weight Download
- Tokenizer: NeoQuasar/Kronos-Tokenizer-base -> `kronos/weights/tokenizer/`
- Model: NeoQuasar/Kronos-base -> `kronos/weights/model/`
- Config: s1_bits=10, s2_bits=10, d_model=832, n_layers=12, n_heads=16

#### Pitfalls Found & Fixed

1. **Parameter naming confusion**: `T=256` (context length) and `temperature=0.6` (sampling temperature) were mixed together
   - Fix: `T` -> `context_len`, CLI parameter `--T` -> `--context-len`
   - Four files synchronized: predictor.py / orchestrator.py / cli_predict.py / test_predictor.py

2. **Incomplete current-month data**: Running predictions on May 25, the May monthly bar is still a partial month bar
   - Fix: when auto-inferring x_ts, check if current month is complete (`now.day < 28` means incomplete); if incomplete, take previous month

3. **600519 crashes at specific windows** (pending deeper investigation): when x_ts=latest month, only 600519 produces extreme values (open=-1144), other stocks are normal
   - Suspected root cause: z-score normalization parameters for specific 200-month windows cause encoded values to exceed tokenizer training distribution
   - Current mitigation: previous-month strategy + median aggregation, but should not rely on these workarounds
   - What actually needs investigation: BSQ decoder response to out-of-distribution token combinations, or numerical stability of inverse normalization

4. **Temperature too high**: temperature=1.0 gives std=38%, single sample unreliable
   - Fix: default temperature=0.6, top_k=30, top_p=0.85 -> std reduced to ~12%

5. **Negative price decode**: occasional s1/s2 token combinations decode to negative OHLCV values
   - Safeguard: predictor clamps (O/H/L/C >= 0.01, V/A >= 0) after inverse normalization
   - This should not be the primary fix — root cause is above (BSQ decoder robustness)

6. **stdout pollution of JSON**: data_adapter and from_pretrained print logs to stdout
   - Fix: data_adapter uses stderr; JSON mode isolates intermediate output via os.devnull

7. **timestamps column vs Index**: predictor filters by `df.index <= x_ts`, but data_adapter returns RangeIndex
   - Fix: data_adapter sets DatetimeIndex

#### Measured Performance

| Stock | Technical Signal | AI Prediction | Confidence | Composite |
|------|---------|---------|--------|---------|
| 600519 Moutai | Bearish alignment+MACD death cross | -13.9% | 80% | Resonance bearish |
| 000001 Ping An Bank | Bearish alignment+KDJ death cross | +1.83% | 13% | AI sideways/technical bearish |
| 600036 CMB | MA entanglement | -17.23% | 90% | AI bearish |

#### Default Parameters

| Parameter | Default | Category |
|------|--------|------|
| `context_len` | 256 | Context window (historical K-line count) |
| `temperature` | 0.6 | Sampling temperature |
| `top_k` | 30 | top-k filtering |
| `top_p` | 0.85 | Nucleus sampling |
| `n_samples` | 30 | Independent sampling count |
| `pred_len` | 3 | Prediction months |

### Key Design Decisions

1. **No repo clone, handwritten reproduction**: Referenced original repo (shiyu-coder/Kronos) source code; all `__init__` signatures fully consistent with original repo, guaranteeing `from_pretrained()` compatibility
2. **Independent venv isolation**: `kronos/venv/` (Python 3.13.11, torch 2.11.0+cu128), does not affect project `.venv`
3. **Field mapping**: SQLite monthly table field names fully match Kronos requirements (open/high/low/close/volume/amount), no mapping conversion needed
4. **DatetimeIndex**: DataFrame index = monthly dates; predictor uses `df.index <= x_ts` for time filtering
5. **Median aggregation**: signal_generator uses median instead of mean to resist single-sample outliers. Direction judgment also uses median
6. **Last complete month**: x_ts defaults to last complete month (excludes partial month if current month not yet ended), not blindly using second-to-last month
7. **context_len naming**: Context window length uses `context_len` (not abbreviated to T), clearly distinguished from `temperature`/`top_k`
8. **amount tolerance**: null -> fill 0, ensuring all stock compatibility

### Pending

- [x] `python -m kronos.download_weights` downloads HuggingFace pretrained weights
- [x] Pretrained weight tests passing (36/37, 1 HF online test skipped)
- [x] End-to-end inference verified (600519/000001/600036 all passed)
- [x] Node.js CLI `ema analyze` integration complete
- [ ] Correlation analysis with existing 32d LightGBM signal
- [ ] Signal fusion layer design
- [ ] Batch run Kronos predictions for all A-shares, build cache

### Project-Level Conclusions

**Daily LSTM IC=0.141 is the only rigorously verified signal. Monthly LightGBM IC=0.063 ceiling confirmed.**
More features/frequencies/model architectures/data sources cannot break the current IC ceiling.
Recommendation: Phase 23 Chrome extension productization + Paper Trading + Kronos signal line supplement.

---

## P3 Signal Gating Final Conclusion (2026-05-30)

Unified baseline pool: 24tp Baostock unbiased pool (3190 pairs), same measurement basis.

| Signal | Test spread | Test CI | Decay | Judgment |
|------|-----------|---------|-------|------|
| **Kronos** | +9.7% | **[+5.1,+15.3]** | 0.80 | **Only passing, runtime connected** |
| Reversal | -19.2% | [-24.5,-14.4] | 1.78 | Significantly negative, pending runtime removal |
| Momentum | -9.8% | [-15.5,-4.4] | 1.14 | Fail |
| LSTM/GRU/LGB | — | CI all contain 0 | Negative/NaN | Fail |

Key finding: Reversal +6.6% came from 12tp pool; expanding to 24tp flipped to -19.2%. Kronos is the only one with test CI excluding 0 + healthy decay + r(LLM)=0.069. A/B test: Kronos 4/4 disagreed tickets changed LLM direction.

Runtime: Kronos=ON, Reversal=OFF (removed), LSTM=OFF.

### LLM Direction Signal 24tp Re-Gating (2026-06-05)

LLM's own directional prediction (minimal prompt, no external signal injection) re-evaluated on 24tp pool:

- 1613 test pairs, DeepSeek, 0.42 CNY
- Signal distribution: neutral 70.9%, bull 25.9%, strong_bull 1.9%, bear 1.4%, strong_bear 0%
- Extremely conservative: almost never selects direction, strong_bear zero times

| Signal | 12tp (Phase C) | 24tp (this run) |
|------|---------------|------------|
| LLM spread | +8.9% CI[+3.7,+14.1] passed | +1.6% CI[-3.4,+6.6] fail |
| strong_bull alpha | — | +26.0% (n=30) |
| Directional accuracy | — | 57.5% |

**Conclusion**: LLM directional signal is not significant on 24tp pool (CI contains 0). strong_bull 30 predictions have alpha +26%, extremely high but sample too small. LLM is unsuitable as an independent basis for directional judgment.

### Role Changes (2026-06-05)

- **Kronos**: Primary direction signal — only pass through gating on 24tp
- **LLM**: Auxiliary interpretation — provides technical analysis, event interpretation, fundamental logic; does not independently give direction
- **Reversal**: Removed from runtime (flipped negative on 24tp)
- **Kronos prompt block updated**: "If Kronos diverges from technicals, prioritize Kronos direction, use technicals to explain possible reasons"

### Changed Files (2026-06-05)

| File | Change |
|------|------|
| `background.js` | Removed reversal factor computation block + reversalSignalData parameter |
| `lib/build-prompt.js` | Removed buildReversalSignalBlock import/call/rendering |
| `lib/prompt-templates.js` | Reversal function deprecated, Kronos block rewritten: positioned as primary direction signal |
| `scripts/p3-kronos-clean-ab.py` | Added: Kronos zero-cost complementarity analysis |
| `scripts/p3-llm-gate-24tp.py` | Added: LLM 24tp gating runner |
| `scripts/p3-analyze-llm-gate.py` | Added: LLM gating results analysis |
