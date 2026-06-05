# P0a Debate Checkpoint Resume (with fingerprint fix)

> Branch: `p0a-debate-checkpoint` -> `p0a-fix-fingerprint` | Date: 2026-05-29 | Implemented based on p0-audit findings

---

## Problem

p0-audit confirmed: Debate intermediate state is stored purely in memory variable `partials`, with no checkpoint resume. When Service Worker
is terminated by Chrome mid-debate, the LLM call results from all three Agents (Bull/Bear/Predictor)
are lost entirely; user retry requires rerunning all from scratch, wasting tokens (single Agent approx 0.01-0.05 USD).

## Solution

Add "persistence + resume" around `runDebate`, without modifying any Agent prompt / LLM provider /
score-fusion / structured output parser.

### Checkpoint Data Flow

```
runDebate(ctx, opts)
  |
  +- 1. loadCheckpoint(debate-wip:...)  <- chrome.storage.local
  |     +- fp match -> reuse completed partial
  |     +- fp mismatch -> discard, rerun all
  |
  +- 2. Bull/Bear/Predictor concurrent (serial write chain avoids races)
  |     +- Already reused: Promise.resolve(chk.partials[role])
  |     +- New call: agent.run() -> fulfilled -> mergeChain persist
  |
  +- 3. Judge synthesis (successCount >= 2)
  |
  +- 4. await mergeChain (ensure persistence complete)
       -> return { partials, errors, judge, ... }
       -> background.js writes official cache -> clearDebateCheckpoint()
```

### Checkpoint Key

```
debate-wip:<market>.<code>:<period>:<bucket>:<style>:<decision>
```

Shares the same identity identifier as final cache key `analysis:...`, only changes namespace.

### Input Fingerprint

```
djb2(code|period|barCount|firstDate|lastDate|lastClose)  <- only closed bars
```

**Fingerprint fix (p0a-fix-fingerprint)**: Initial fingerprint included current-period unclosed K-line close;
intraday ticks during trading hours changed fingerprint on every tick, making checkpoint hits
impossible during session. Fixed by adding `isBarClosed()` filtering:

- monthly: current month does not participate in fingerprint
- weekly: current ISO week does not participate in fingerprint
- daily: today does not participate in fingerprint

Current bar close fluctuations no longer affect fingerprint. If closed bar close changes (adjustment factor base
switch, data source correction), fingerprint auto-invalidates and discards for rerun.

### Timeout Cleanup

Checkpoint automatically discarded after exceeding 1 hour (`CHECKPOINT_STALE_MS = 60*60*1000`),
preventing storage accumulation. All catch blocks use `console.warn`, no more silent error swallowing.

### Serial Write Chain

Three Agents call LLM concurrently, but checkpoint writes use serial write chain `mergeChain`:

```js
let mergeChain = Promise.resolve();
// Agent A done -> mergeChain = mergeChain.then(() => save(A))
// Agent B done -> mergeChain = mergeChain.then(() => save(B))
// ...
// await mergeChain -> ensure all persisted
```

Prevents concurrent read-modify-write races (two Agents simultaneously reading empty checkpoint, later write overwriting earlier write).

## Changed Files

### `lib/agents/runner.js` (+90 lines)

| Addition | Description |
|------|------|
| `CHECKPOINT_VERSION = 1` | Schema version; increment when structure changes in future |
| `CHECKPOINT_STALE_MS` | 1-hour timeout; expired checkpoint auto-discarded |
| `getIsoWeek(dateStr)` | ISO week calculation, consistent with background.js |
| `isBarClosed(dateStr, period)` | Determine if bar belongs to current period (monthly/weekly/daily), for fingerprint filtering |
| `buildFingerprint(ctx)` | djb2 hash of `code|period|barCount|firstDate|lastDate|lastClose` (closed bars only) |
| `loadCheckpoint(key, fp)` | Read storage, verify fingerprint+timeout; delete if mismatch |
| `mergeCheckpointPartial(key, role, value, fp)` | Merge-write single Agent success result |
| `mergeCheckpointError(key, role, errMsg, fp)` | Merge-write single Agent failure info |
| `clearDebateCheckpoint(key)` | Delete checkpoint (exported function, called by background.js) |

`runDebate` modifications:
- Entry: load checkpoint, verify fingerprint
- Agent loop: if checkpoint exists -> `Promise.resolve` reuse; otherwise call `agent.run()`
- Each Agent completion appends to `mergeChain` (does not block sibling Agents)
- After Judge: `await mergeChain` ensures persistence complete

**Untouched**: `sumCost`, Judge logic, return structure — field-by-field consistent with pre-change version.

### `background.js` (+4 lines)

```
:14   import { runDebate, clearDebateCheckpoint }  // added clearDebateCheckpoint
:274  const checkpointKey = `debate-wip:...`       // construct key
:280  checkpointKey,                                 // pass into opts
:507  await clearDebateCheckpoint(...)               // cleanup after official cache write
```

### `test/agents/runner.test.js` (+170 lines)

Added mock `chrome.storage.local` (Map implementation) + 6 checkpoint-specific tests:

| Test | Scenario | Assertion |
|------|------|------|
| No checkpoint -> all three called | storage empty | fetch 4 times (Bull+Bear+Pred+Judge), checkpoint persisted |
| bull+bear cached -> only predictor called | preset checkpoint with bull+bear | fetch 2 times (Pred+Judge), bull.text='bull checkpoint cached' |
| Fingerprint mismatch -> discard and rerun | preset checkpoint with wrong fp | fetch 4 times, old value not in results |
| >=2 success rule unbroken | checkpoint only bull, rest 500 | successCount=1, Judge skipped |
| Current-period bar close change -> fingerprint unchanged | closed bar unchanged, add current-period bar simulating intraday | fetch 2 times, bull+bear reuse checkpoint |
| Closed bar close change -> discard | modify closed bar close simulating adjustment factor switch | fetch 4 times, full rerun |

### `PROGRESS.md` (+20 lines)

Added "Debate Checkpoint Resume" section under "Mandatory Rules."

## Untouched (Hard Constraint Verification)

| Module | Verification Method |
|------|---------|
| Agent prompt (bull/bear/predictor/judge) | `git diff master -- lib/agents/{bull,bear,predictor,judge}.js` -> no changes |
| LLM provider (anthropic/deepseek) | `git diff master -- lib/llm/` -> no changes |
| score-fusion | `git diff master -- lib/score-fusion.js` -> no changes |
| Structured output parsing | `git diff master -- lib/parse-structured-output.js` -> no changes |
| runDebate return structure | source diff shows `return { partials, errors, judge, totalCost, totalDurationMs }` unchanged |

## P0 Boundary

- **Done**: User retry automatically resumes from checkpoint, reuses completed Agents, no duplicate spend
- **Not done**: alarm watchdog auto-recovery (P1), Judge result caching, persistent keepalive timer
- **Silent degradation**: storage unavailable -> catch swallows error, debate proceeds normally (falls back to no-checkpoint behavior)

## Tests

```
251 tests | 0 fail | 0 skip
```

Added 6 checkpoint tests (including 2 fingerprint-specific), existing tests zero regression.

## Manual Review Checklist

- [ ] `git diff master...p0a-fix-fingerprint` confirm no touch to prompt/LLM/scoring/parsing
- [ ] Compare `runDebate` return structure field-by-field with pre-change version
- [ ] Run `node --test test/agents/runner.test.js` -> 10/10 pass
- [ ] Real environment: Stop SW during trading hours -> retry analysis -> console confirm checkpoint hit, Agent reused
- [ ] Fingerprint specific: intraday current-period bar close fluctuation does not affect fingerprint -> checkpoint can hit
- [ ] Fingerprint specific: modify closed bar close -> checkpoint discarded and rerun
