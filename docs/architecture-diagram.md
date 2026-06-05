# Eastmoney Monthly AI Analysis Assistant — Architecture & Function Diagram

> Generated: 2026-05-29
> Last updated: 2026-05-29 (p0 audit correction: added Native Messaging bridge, re-labeled research layer connectivity, annotated score-fusion weight source)

---

## 1. System Architecture Diagram

```
+---------------------------------------------------------------------+
|                         UI Layer                                     |
+---------------+------------------+----------------------------------+
|  popup.html   |   content.js     |        viewer.html               |
|  popup.js     |  (Shadow DOM)    |        viewer.js                 |
|  Settings/    |  FAB button +    |       Analysis Viewer            |
|  Cost/        |  side panel      |       (URL parameter load)       |
|  History/Debug|  "Event" scraping|                                  |
+-------+-------+--------+---------+-------------+--------------------+
        | chrome.runtime | chrome.runtime        | URL parameter
        | .sendMessage() | .sendMessage()        |
        v                v                        v
+---------------------------------------------------------------------+
|                    Service Worker Orchestration (background.js)      |
|                                                                     |
|  +-------------------------------------------------------------+   |
|  |                    Three Analysis Modes                       |   |
|  |  +----------+  +------------------+  +------------------+   |   |
|  |  | Single   |  | Multi-Agent      |  | Multi-Period     |   |   |
|  |  | Analysis |  | Debate Mode      |  | Resonance Mode   |   |   |
|  |  |          |  | Bull+Bear+       |  | Monthly+Weekly+  |   |   |
|  |  |          |  | Predictor->Judge |  | Daily resonance  |   |   |
|  |  +----------+  +------------------+  +------------------+   |   |
|  +-------------------------------------------------------------+   |
|                                                                     |
|  Native Messaging Bridge (chrome.runtime.sendNativeMessage):         |
|    +- query_sector_alpha(code,period,lookback) -> sector alpha      |
|    +- read(mc_dropout/{code}) -> LSTM MC Dropout pre-computed JSON  |
+--------------+------------------------------------------------------+
               | chrome.runtime.sendNativeMessage("com.eastmoney_ai.sync", ...)
               v
+----------------------------------------------------------------------+
|            Native Messaging Host (native-host/server.js)              |
|            Independent Node process, Chrome launches on demand,      |
|            stdio protocol                                             |
|                                                                      |
|  +--------------------------------------------------------------+   |
|  | Message routing (handleMessage):                               |   |
|  |  query_sector_alpha -> import lib/sector/alpha.js -> calcSectorAlpha(db) |
|  |  read(mc_dropout/{code}) -> fs.readFileSync(JSON) -> return LSTM signal |
|  |  sync/sync_batch -> fs.writeFileSync -> .eastmoney-ai/storage/   |
|  +--------------------------------------------------------------+   |
|                                                   |                  |
|                         dynamic import('../lib/db/connection.js')    |
|                                                   v                  |
|                    +------------------------------+                  |
|                    |   lib/db/connection.js        |                  |
|                    |   better-sqlite3 (Node native)|                  |
|                    +--------------+---------------+                  |
|                                   |                                  |
+-----------------------------------+----------------------------------+
                                    v
                    +------------------------------+
                    |      SQLite Database         |
                    |  .eastmoney-ai/db/            |
                    |  klines-v2.sqlite             |
                    |  Monthly/Weekly/Daily/60min   |
                    |  Sector mapping / Adjust. factors|
                    +------------------------------+
        |                              |
        v                              v
+----------------------+   +------------------------------+
|   lib/ Core Engine   |   |      Cache & Persistence     |
|  (static import      |   |                              |
|   closure, no Node   |   | chrome.storage.local         |
|   native modules)    |   |  (analysis cache/history/    |
|                      |   |   settings)                  |
| +------------------+ |   |                              |
| | build-prompt.js  | |   +------------------------------+
| | (Prompt assembly)| |
| +--------+---------+ |
|          |           |
| +--------+---------+ |
| | lib/agents/      | |
| | (Multi-Agent)    | |
| +--------+---------+ |
|          |           |
| +--------+---------+ |
| | lib/indicators/  | |   +------------------------------+
| | (Technical lib)  | |   |   lib/data-sources/           |
| | MA/MACD/RSI/     | |   |   Multi-source K-line fetch   |
| | KDJ/BOLL/OBV/ATR | |   |   + degradation              |
| +--------+---------+ |   |                              |
|          |           |   | Eastmoney->Baidu->Sina->Tencent|
| +--------+---------+ |   | (timeout+rate-limit degrade)  |
| | lib/signals/     | |   +--------------+---------------+
| | (Signal factory) | |                  |
| | 15 long/short    | |   +--------------+---------------+
| +--------+---------+ |   | lib/data-validation/         |
|          |           |   | K-line quality check          |
| +--------+---------+ |   | (field/sequence/consistency)  |
| | lib/quant-factors| |   +------------------------------+
| | 5-dim quant      | |
| | factors (pure    | |   +------------------------------+
| | compute, no ext) | |   |  lib/eval/ + lib/evaluation/  |
| +------------------+ |   |  Eval framework + nightly     |
|                      |   |  pipeline                     |
| +------------------+ |   |  (CLI only, not in runtime    |
| | lib/self-backtest| |   |   closure)                    |
| | (Self-backtest   | |   +------------------------------+
| |  calibration)    | |
| +------------------+ |
|                      |
| +------------------+ |
| | lib/score-fusion | |  <- regime weights hardcoded (score-fusion.js:29-34)
| | (LLM+Quant fuse) | |     reads no config files or research outputs
| +------------------+ |
+----------------------+
        |
        v
+---------------------------------------------------------------------+
|                      LLM Provider Abstraction (lib/llm/)             |
|                                                                     |
|  +-----------------+    +-----------------+                         |
|  | Anthropic       |    | DeepSeek        |                         |
|  | claude-sonnet-4 |    | deepseek-chat   |                         |
|  | (SSE stream+    |    | (OpenAI compat, |                         |
|  |  Tool Use+      |    |  no Tool Use)   |                         |
|  |  extended think)|    |                 |                         |
|  +-----------------+    +-----------------+                         |
+---------------------------------------------------------------------+
        |
        v
+---------------------------------------------------------------------+
|                     External Data Sources & APIs                     |
|                                                                     |
|  Eastmoney API       Baidu Stock       Sina Finance     Tencent Fin. |
|  push2his.           finance.pae.     hq.sinajs.cn     web.ifzq.    |
|  eastmoney.com       baidu.com                          gtimg.cn    |
+---------------------------------------------------------------------+


+---------------------------------------------------------------------+
|                   Offline Research Layer (Python / Node CLI)         |
|                                                                     |
|  +===============================================================+  |
|  |  Connected to runtime (via Native Messaging, reads pre-computed) |
|  +===============================================================+  |
|  |                                                               |  |
|  |  lib/lstm/  Deep Learning                                     |  |
|  |  +---------------------------------------------------------+  |  |
|  |  | mc_dropout.py (MC Dropout uncertainty quantification)    |  |  |
|  |  |   v 50 forward passes -> y3_mean/y3_std/overall_confidence|  |  |
|  |  | cli/mc_export_json.py                                    |  |  |
|  |  |   v export JSON -> .eastmoney-ai/storage/mc_dropout/{code}.json |
|  |  | native-host/server.js (read handler)                     |  |  |
|  |  |   v sendNativeMessage('read', 'mc_dropout/{code}')       |  |  |
|  |  | background.js:325-358                                    |  |  |
|  |  |   v uncertainty gating: high=skip, low+medium=inject     |  |  |
|  |  | buildLstmSignalBlock() -> prompt                         |  |  |
|  |  +---------------------------------------------------------+  |  |
|  |  Native host unavailable or data not generated: silent degrade, |
|  |  does not affect normal analysis                                |
|  +===============================================================+  |
|                                                                     |
|  +===============================================================+  |
|  |  Independent R&D (0 runtime references, no runtime-consumed    |
|  |   config/outputs produced)                                     |
|  +===============================================================+  |
|  |                                                               |  |
|  |  kronos/  Transformer Prediction (BSQ+autoregressive)          |  |
|  |  lib/backtest/  Backtest Engine (Walk-Forward/sector neutral)  |  |
|  |  lib/portfolio/  Portfolio Optimization (Black-Litterman/      |
|  |                  Risk Parity)                                  |  |
|  |  lib/scanner/  Batch Scanning (HS300+watchlist, CLI only)     |  |
|  |  lib/uncertainty/  Uncertainty Quantification (CLI eval only) |  |
|  |                                                               |  |
|  |  These modules' research results are recorded in docs/*.md     |  |
|  |  and PROGRESS.md; runtime makes no reads.                     |  |
|  +===============================================================+  |
|                                                                     |
|  +----------------------------------------------------------+      |
|  | cli/ CLI Toolchain (Node)                                 |      |
|  | DB init | Batch eval | Nightly batch | Eval scripts       |      |
|  +----------------------------------------------------------+      |
|                                                                     |
|  +----------------------------------------------------------+      |
|  | scripts/ Python Research Experiments (~100+)              |      |
|  | Feature engineering | LSTM experiments | XGBoost ensemble |      |
|  | IC analysis | Factor research                            |      |
|  +----------------------------------------------------------+      |
+---------------------------------------------------------------------+
```

---

## 2. Function Module Diagram (Data Flow)

```
                        User Clicks Analyze
                             |
                             v
              +--------------------------+
              |     Parse URL and Code    |
              |    parse-url.js           |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |     Fetch K-line Data     |
              |  lib/data-sources/        |
              |  Multi-source degrade     |
              |  +-Eastmoney (primary)    |
              |  +-Baidu (backup 1)       |
              |  +-Sina (backup 2)        |
              |  +-Tencent (backup 3)     |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |     Data Quality Check    |
              |  lib/data-validation/      |
              |  Field/sequence/consistency|
              +------------+-------------+
                           |
       +-------------------+-------------------+
       v                   v                   v
+------------+    +---------------+    +---------------+
| Sector alpha|    | LSTM signal   |    |  K-line data  |
| (async)    |    | (async, opt)  |    |  (main path)  |
|            |    |               |    |               |
| sendNative |    | sendNative    |    |               |
| Message(   |    | Message(      |    |               |
|  query_    |    |  read,        |    |               |
|  sector_   |    |  mc_dropout/  |    |               |
|  alpha)    |    |  {code})      |    |               |
|    v       |    |    v          |    |               |
| native-    |    | MC Dropout    |    |               |
| host ->    |    | JSON -> uncert|    |               |
| lib/sector |    | gating        |    |               |
| /alpha.js  |    | (high=skip)   |    |               |
|    v       |    |    v          |    |               |
| sectorAlpha|    | lstmSignal    |    |               |
| Data       |    | Data          |    |               |
+----+-------+    +-------+-------+    +-------+-------+
     |                    |                    |
     +--------------------+--------------------+
                          |
           +--------------+--------------+
           v              v              v
    +------------+  +------------+  +------------+
    | Monthly    |  | Weekly     |  | Daily      |
    | direction  |  | direction  |  | direction  |
    | MA60 slope |  | MA20 slope |  | MA20 slope |
    | +MACD      |  | +MACD      |  | +MACD      |
    +-----+------+  +-----+------+  +-----+------+
          |               |               |
          +---------------+---------------+ 
                          |
                          v
              +--------------------------+
              |   Multi-Period Resonance  |
              |   lib/multi-period/       |
              |   strong/partial/divergent|
              +------------+-------------+
                           |
           +---------------+---------------+
           v               v               v
    +------------+  +------------+  +------------+
    | Technical  |  | Signal     |  | Quant      |
    | Indicators |  | Detection  |  | Factors    |
    | MA/MACD/   |  | 15 signals |  | 5-dim      |
    | RSI/KDJ/   |  | long+short |  | trend/pos/ |
    | BOLL/OBV   |  | factory    |  | vol/vol/   |
    |            |  |            |  | consistency|
    +-----+------+  +-----+------+  +-----+------+
          |               |               |
          +---------------+---------------+ 
                          |
                          v
              +--------------------------------------+
              |         Prompt Assembly               |
              |       lib/build-prompt.js             |
              |                                       |
              |  Input: K-line table + indicators     |
              |       + signals                       |
              |       + sectorAlphaData (optional)    |
              |       + lstmSignalData   (optional)   |
              |       + resonance (pure fn format)    |
              |                                       |
              |  4 styles: tech/chanlun/value/comp    |
              |  3 modes: single/debate/resonance     |
              +------------+-------------------------+
                           |
            +--------------+--------------+
            v                             v
    +--------------+             +--------------+
    |  Single      |             |  Debate Mode |
    |  Direct LLM  |             |  3 concurrent|
    |              |             |  Agents      |
    |              |             |  v           |
    |              |             |  Judge synth |
    +------+-------+             +------+-------+
           |                            |
           +------------+---------------+
                        |
                        v
              +--------------------------+
              |    LLM Provider Call      |
              |    lib/llm/              |
              |    Anthropic | DeepSeek  |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |    Structured Output     |
              |    Parsing               |
              |   parse-structured-output|
              |   Extract JSON, validate |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |    Score Fusion (opt)    |
              |   lib/score-fusion.js    |
              |   LLM score x Quant score|
              |                          |
              |   Adaptive Regime weights|
              |   (hardcoded constants   |
              |   score-fusion.js:29-34) |
              |   - strong_trend: L30%+Q70%|
              |   - sideways:     L92%+Q8%|
              |   - high_vol:     L85%+Q15%|
              |   - mixed:        L50%+Q50%|
              |   No config/research read |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |    Cache Write + Return   |
              |   chrome.storage.local   |
              |   analysis:market.code:  |
              |   period:bucket:style:   |
              |   mode:decision           |
              +--------------------------+

=====================================================

              Offline Eval & Research Pipeline

    +-----------------+     +-----------------+
    | Nightly Eval    |     | Research        |
    | Pipeline        |     | Experiments     |
    | lib/evaluation/ |     | scripts/         |
    |                 |     |                  |
    | nightly.js      |     | Feature eng.     |
    |  -> collector   |     | LSTM training    |
    |  -> cost-guard  |     | XGBoost ensemble |
    |  -> draft-review|     | IC analysis      |
    |  -> refine      |     | Factor research  |
    +--------+--------+     +--------+---------+
             |                       |
             v                       v
    +-----------------------------------------+
    |              SQLite Database             |
    |  Monthly/Weekly/Daily | Sector map | Adj.|
    +------------------+----------------------+
                       |
         +-------------+-------------+
         v             v             v
    +-----------+ +----------+ +-----------+
    |  LSTM     | | Backtest | | Portfolio |
    |  Training | | Engine   | | Optimizer |
    |  lib/lstm/| | backtest | | portfolio |
    |  [Connected]| [Indep. | | [Indep.  |
    |  via Native |  R&D]    | |  R&D]    |
    |  Messaging  | 0 runtime| | 0 runtime |
    |  inject     | refs     | | refs      |
    |  to prompt  |          | |           |
    +-----------+ +----------+ +-----------+
         |
         v
    +-----------+
    | mc_export |
    | _json.py  |
    | -> JSON   |
    | -> native-|
    |   host    |
    +-----------+
```

---

## 3. Module Relationship Summary Table

| Layer | Module | Responsibility | Runtime Status |
|------|------|------|-----------|
| **Interaction** | `content.js` / `popup.js` / `viewer.js` | User UI, settings management, analysis display | Runtime entry |
| **Orchestration** | `background.js` | Message routing, mode selection, cache management, Native Messaging bridge | Runtime hub |
| **Analysis Engine** | `lib/build-prompt.js` | Prompt assembly (4 styles x 3 modes) | Runtime |
| | `lib/agents/` | Multi-Agent debate (Bull/Bear/Predictor/Judge) | Runtime |
| | `lib/indicators/` | Technical indicator computation (MA/MACD/RSI/KDJ/BOLL/OBV) | Runtime |
| | `lib/signals/` | 15 trading signal detection | Runtime |
| | `lib/multi-period/` | Multi-period resonance analysis | Runtime |
| | `lib/quant-factors.js` | 5-dim quant factors (pure compute) | CLI only |
| | `lib/score-fusion.js` | LLM+Quant hybrid scoring, regime weights hardcoded (`:29-34`) | CLI only |
| | `lib/self-backtest.js` | Self-backtest calibration | Runtime |
| **LLM** | `lib/llm/` | Provider abstraction (Anthropic/DeepSeek) + pricing | Runtime |
| **Data** | `lib/data-sources/` | 4-source K-line fetch + degradation | Runtime |
| | `lib/data-validation/` | K-line quality validation | Runtime |
| | `lib/db/` | SQLite persistence -> **accessed via Native Messaging proxy** | CLI + native-host only |
| **Bridge** | `native-host/server.js` | stdio protocol, routes query_sector_alpha / read(mc_dropout) / sync | Independent Node process |
| | `lib/sector/alpha.js` | Sector alpha computation -> dynamically imported by native-host | native-host only |
| **Evaluation** | `lib/eval/` + `lib/evaluation/` | Eval framework + nightly pipeline | CLI only |
| **Research (Connected)** | `lib/lstm/` -> JSON -> native-host -> background.js | MC Dropout uncertainty quantification, uncertainty gating filter | Offline training, runtime reads outputs |
| **Research (Independent R&D)** | `kronos/` | Transformer price prediction, has own CLI | 0 runtime refs |
| | `lib/backtest/` | Walk-Forward backtest | 0 runtime refs |
| | `lib/portfolio/` | Black-Litterman / Risk Parity | 0 runtime refs |
| | `lib/scanner/` | Batch stock scanning | CLI only |
| | `lib/uncertainty/` | Uncertainty quantification | CLI only |
| **CLI** | `cli/` | Command-line tools (DB/batch/batch processing) | CLI only |
| **Scripts** | `scripts/` (~100+) | Python research experiments | Offline |

---

## 4. Key Architecture Decisions (p0 audit confirmed)

### 4.1 Native Messaging Bridge

```
background.js --sendNativeMessage--> native-host/server.js --dynamic import--> lib/db/connection.js -> SQLite
                                     ^
                                     Chrome launches on demand, stdio length-prefix protocol
                                     Message types: query_sector_alpha / read / sync / sync_batch / remove
```

- **Carried content**:
  - `query_sector_alpha(code, period, lookback)` -> `lib/sector/alpha.js` -> `calcSectorAlpha(db)` -> sector excess return
  - `read('mc_dropout/{code}')` -> read `.eastmoney-ai/storage/mc_dropout/{code}.json` -> LSTM MC Dropout signal
  - `sync/sync_batch` -> write `chrome.storage.local` data to disk files
- `background.js` **never directly imports `lib/db/`**, does not introduce `better-sqlite3` into Service Worker

### 4.2 Research Layer Connectivity

| Module | Runtime Reference? | Data Channel |
|------|------------|---------|
| `lib/lstm/` | **Yes** | `mc_dropout.py` -> `mc_export_json.py` -> `.eastmoney-ai/storage/mc_dropout/{code}.json` -> native-host `read` handler -> `background.js:325-358` -> uncertainty gating(high=skip) -> `buildLstmSignalBlock()` -> prompt |
| `kronos/` | **No** | Only `cli_predict.py` manual invocation |
| `lib/backtest/` | **No** | Independent Python script |
| `lib/portfolio/` | **No** | Independent Python script |

### 4.3 Score-fusion Weight Source

`lib/score-fusion.js:29-34` — **Hardcoded constants**, reads no config files or research outputs:

```js
const ADAPTIVE_WEIGHTS = {
  strong_trend: { llm: 0.30, quant: 0.70 },
  sideways:     { llm: 0.92, quant: 0.08 },
  high_vol:     { llm: 0.85, quant: 0.15 },
  mixed:        { llm: 0.50, quant: 0.50 },
};
```

Regime detection (`detectStockRegime`) based on runtime-computed `quantResult.factors.f2` (price position) and `f3` (volatility percentile), pure function.

### 4.4 Static Import Boundary

Service Worker static import closure = `background.js` -> transitive closure of 7 subdirectories under lib/.
`lib/db/`, `lib/lstm/`, `lib/backtest/`, `lib/portfolio/`, `lib/sector/` are all NOT in the closure.
`lib/db/` is only accessed by native-host and CLI scripts via **dynamic import()**.

Guard test: `test/import-guard.test.js`.

---

## 5. Directory Structure

```
eastmoney-monthly-ai/
+-- manifest.json              # Chrome MV3 extension manifest
+-- background.js              # Service Worker orchestration + Native Messaging consumer
+-- content.js / content.css   # Shadow DOM injection
+-- popup.html / popup.js      # Settings panel
+-- viewer.html / viewer.js    # Independent analysis viewer
+-- native-host/               # Native Messaging Host (Node process)
|   +-- server.js              #   stdio protocol, dynamic import lib/db
|   +-- install.js / uninstall.js
|   +-- launcher.bat
|   +-- manifest/
+-- lib/                       # Core library
|   +-- llm/                   #   Provider abstraction
|   +-- agents/                #   Multi-Agent system
|   +-- indicators/            #   Technical indicators
|   +-- signals/               #   Signal factory
|   +-- multi-period/          #   Multi-period resonance
|   +-- data-sources/          #   Multi-source K-line
|   +-- data-validation/       #   Data validation
|   +-- tools/                 #   LLM tools
|   +-- db/                    #   SQLite (CLI + native-host reachable only)
|   +-- sector/                #   Sector alpha (native-host reachable only)
|   +-- eval/                  #   Eval framework (CLI only)
|   +-- evaluation/            #   Nightly pipeline (CLI only)
|   +-- lstm/                  #   Deep learning (offline training, outputs via native-host inject into runtime)
|   +-- backtest/              #   Backtest engine (independent R&D)
|   +-- portfolio/             #   Portfolio optimization (independent R&D)
|   +-- scanner/               #   Batch scanning (CLI only)
|   +-- dashboard/             #   Scoring dashboard (viewer)
|   +-- uncertainty/           #   Uncertainty quantification (CLI only)
+-- cli/                       # CLI tools
+-- scripts/                   # Python research scripts
+-- kronos/                    # Transformer prediction (independent R&D)
+-- data/                      # Static data
+-- test/                      # Tests (includes import-guard.test.js)
+-- docs/                      # Documentation (includes p0-audit.md)
```
