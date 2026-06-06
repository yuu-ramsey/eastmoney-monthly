# Eastmoney Monthly Chart AI Analysis Assistant

> **Project Charter**: [.claude-charter.md](.claude-charter.md) (must read before every task)

Chrome MV3 extension, injected into Eastmoney stock pages (quote.eastmoney.com),
analyzing stock trends via LLM API. Local personal use only, not published to Chrome Web Store.

## Regime-Adaptive MoE Architecture (designed 2026-06-05)

### Design Principles

- **No single point of failure**: Any component failure does not affect the whole
- **LLM isolation**: LLM is outside the signal chain, only does interpretation
- **Mechanical lookup table**: regime→weights are hardcoded, not passing through any model
- **3D expert grid**: Expert = regime × scale × perspective intersection, not model names

### Expert Registry (3D key)

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

### Gating Lookup Table (mechanical, zero LLM)

```python
def get_weights(regime, scale):
    """Only activate experts with status=='verified' under this (regime, scale)"""
    active = {k: v for k, v in expert_registry.items()
              if k[1] in (regime, 'all') and k[2] in (scale, 'multi')
              and v['status'] == 'verified'}
    # Assign weights by spread magnitude, each ≤40%, normalize
    return normalize_by_spread(active)
```

### LLM Role in Signal Chain

| Layer | Has LLM? | Role |
|----|---------|------|
| Regime Detection | ❌ | Three-detector statistical vote |
| Gating | ❌ | Hardcoded lookup table |
| Signal Aggregation | ❌ | Weighted average |
| Interpretation Output | ✅ | Translate to plain language |

### Pending Validation Slots (per-regime split tasks)

| Signal | Current | Target |
|------|------|------|
| S8 Reversal | 24tp overall −19.2% | per-regime breakdown, may be effective under sideways |
| S6 LGB | bull market −42.2% / bear market +20.6% | split into (lgb, bear, daily) and (lgb, bull, daily) |
| S7 Momentum | already has per-regime | already split→verified |
| S5 GRU | Level 2 boundary | split into (gru_wf, bear, daily) re-verify |
| S1 LLM | 70% neutral | extract (llm_strong, all, monthly) strong labels only |

## Core Architecture

- **content.js**: Injects Shadow DOM-isolated FAB + side panel, scrapes "Major Events"
  from page DOM
- **background.js** (service worker): Fetches Eastmoney monthly/weekly/daily charts + capital flow + calls
  LLM API + cache
- **lib/llm/**: Provider abstraction layer, supports Anthropic and DeepSeek
- **lib/agents/**: Multi-agent architecture Bull/Bear/Predictor/Judge
- **lib/build-prompt.js**: 4 styles (technical/chanlun/value/comprehensive)
  + multi-period resonance prompt assembly

## Key Design Decisions (do not overturn lightly)

### Path Y: Decision aid, not investment advice

The prompt does not directly say "recommend buy/sell"; conclusions use conditional expressions like "if X then verify Y".
Only when `decisionMode = true` does it append a "Personal Decision Perspective" paragraph with explicit suggestions (local
personal use only, to avoid compliance risk).

### Cache Key Structure

`analysis:<market>.<code>:<period>:<bucket>:<style>:<mode>:<decision>`

- bucket uses different granularity by period (monthly=YYYY-MM / weekly=YYYY-WW /
  daily=YYYY-MM-DD)
- No provider dimension——avoids massive duplicate token consumption when user switches providers
- Switching style / period / decisionMode all trigger new cache

### Multi-Provider Isolation

- API keys stored separately: apiKey:anthropic / apiKey:deepseek
- Model fields stored separately: model:anthropic / model:deepseek
- Anthropic defaults to claude-sonnet-4-6, DeepSeek defaults to deepseek-chat
- When switching providers in popup, onProviderChange reads storage directly, **does NOT call
  loadSettings** (avoids the bug where select.value gets overwritten by old value)

### Chanlun style uses FULL strictness constraints (not differentiated by provider)

Previously tried using LITE relaxed constraints for DeepSeek, but output quality was worse (produced
ridiculous judgments like ZG=58/ZD=16 treating the entire segment as a pivot). Rolled back to FULL version for all
providers, but popup added warning "chanlun recommends Anthropic Claude".

### DeepSeek model field mandatory validation

DeepSeek API receiving an invalid model name (e.g., the once-mistyped deepseek-v4-pro) will **silently
fall back** to an inferior model, causing price hallucinations in output (92.21 written as 79.95). popup.js's
validateModel() validates against the KNOWN_MODELS list on blur.

## Core Differences Between the 4 Styles

- **technical**: Moving averages + MACD + volume-price + trap signal detection
- **chanlun**: Pivot identification + stroke/segment + three types of buy/sell points + divergence judgment
- **value**: Historical percentile + long-term trend phasing + valuation stage positioning
- **comprehensive**: Technical + value resonance or divergence

Each style's "analysis task" paragraph must be written independently; do not let multiple styles share
the same prompt paragraph to save tokens.

## Multi-Agent Debate Mode

- Bull / Bear / Predictor three Agents called **concurrently** (Promise.allSettled)
- Only call Judge for synthesis when successCount >= 2
- Judge input = partials from three Agents + base facts
- Judge does NOT redo technical analysis, only compares + evaluates solidity + synthesizes
- Judge must explicitly pick one from [bullish/bearish/neutral/signal inconsistent]
- Debate mode **NOT enabled** under multi-period resonance mode (token cost explosion)

## Prompt Design Principles

1. Conclusion first: Each section opens with a one-sentence conclusion then expands on evidence
2. Synthesis conclusion takes 20-30%: placed at the end, is the user's highest-priority reading section
3. Historical review ≤ 2 sentences: Every historical reference must serve the current judgment, not stated in isolation
4. Under decisionMode, price levels precise to 2 decimal places + label data sources
5. **Strictly prohibit** CAPM / DDM / Kelly Criterion / Beta / risk premium / dividend discount and other
   academic financial models——A-share individual stock level produces false precision

## Test File Locations

- ``test/*.test.js`` — Pure function tests
- ``test/agents/*.test.js`` — Agent layer tests

Run tests: ``node --test test/*.test.js test/agents/*.test.js``

## Working Environment

- Windows system, PowerShell environment
- Node.js v24.15.0 installed at ``node`` (from PATH)
- Project root: ``.``

### Scraping Tool

- **Scrapling v0.4.8**: Adaptive web scraping framework, bypasses Cloudflare
  - Python path: ``.venv\Scripts\python.exe``
  - Invocation: Isolated within project venv, called via absolute path
  - Features: CSS/XPath selectors, adaptive parsing, anti-crawler bypass, concurrent scraping
  - GitHub: https://github.com/D4Vinci/Scrapling

## Current Phase

### LSTM Monthly Prediction Optimization (completed 2026-05-23)

**Final approach**: Monthly 61d features + LightGBM/XGBoost/MLP/Ridge Ensemble
- Weighted Ensemble: **CS_IC=+0.177, Hit=59.4%, Top-5% LS=+0.181**
- 46 test months (2024+), 2247 A-shares, 51K samples
- Annualized performance: 2024 IC=+0.195, 2025 IC=+0.186
- 45/46 months IC positive (97.8%)

**All path verification conclusion**: 6 improvement paths all completed, none surpassed baseline. Monthly 61d + tree model ensemble is the current ceiling.

**Python environment**: `.venv` (Python 3.13.11, torch 2.12+cu128, RTX 5070)
**Data**: 2247 A-shares, 217K sequences, 56 flat features + 47 seq features

### LSTM Academic Research Compilation

#### Core Papers (read and verified)

1. **xLSTM (2405.04517)**: Hochreiter team, exponential gating + matrix memory, GitHub: NX-AI/xlstm
2. **xLSTM-TS (2408.12408)**: 72.8% directional accuracy (EWZ daily), wavelet denoising + xLSTM, GitHub: gonzalopezgil/xlstm-ts
3. **SGP-LSTM (Nature 2024)**: A-share 4500 cross-section, Rank IC +1128%, SGP automatic feature construction is the largest increment
4. **GBDT+LSTM (2505.23084)**: Ensemble boost 10-15%, validates "LSTM ≠ tree model replacement, but complementary feature extractor"
5. **LSTM+GCN (PeerJ 2024)**: LSTM hidden state → downstream model feature columns, A50 test
6. **WD-LSTM (Comput.Econ 2025)**: Wavelet denoising + dual-layer feature selection, causal denoising (only uses historical data)
7. **TFT Confidence Intervals (IIETA 2025)**: q10/q50/q90 quantiles, only trade when 80% confidence interval is same direction

#### Key Conclusions (from papers)

- **Feature engineering > model architecture**: SGP-LSTM's 1128% IC improvement comes from feature construction, not model improvement
- **LSTM positioning should switch**: From "direct predictor" → "tree model's feature extractor" (LSTM+GCN approach)
- **Full-sample 60% is already academic top-tier**: Paper Chu(2025) validates that out-of-sample often collapses to 50%
- **>70% requires confidence filtering**: TFT strategy——only trade when quantiles are same direction, coverage suffers but accuracy jumps

### Execution Paths (by priority)

#### Already executed (do not repeat)
- [x] Phase A: Classification framework → negative returns, inferior to regression
- [x] Phase B+C: Rank Loss + Wavelet + EMA + Walk-Forward → IC improved but Hit still low
- [x] LSTM GPU sweep (7 configs) → BiLSTM h256-l2 optimal, CS_IC +0.126
- [x] v4 Roadmap (3-class + Attention + MC Dropout) → Hit 51.7%, CS_IC +0.114, three-class ineffective

#### Next steps (ordered, verify with papers before running)

1. **Path B: LSTM hidden state → LightGBM feature** (supported by papers #5 #4)
   - ~~Don't retrain LSTM, just forward pass extract hidden state (512d)~~ **Executed, failed. LSTM hidden noise drowned tree model, CS_IC dropped 12%. Closed.**

2. **Path A: Expanded feature engineering** (supported by paper #3)
   - ~~55d→80d: ADX/CCI/OBV, candlestick patterns, consecutive up/down etc.~~ **Executed, adding 11 dimensions actually reduced IC 17%. FFT is the core contributor. Closed.**

3. **Path E: Daily microstructure LSTM** (intra-month volume-price patterns)
   - ~~12 months×22 days→240 timesteps→two-layer LSTM~~ **Executed, Val→Test gap=0.175 severe overfitting, 300 stocks insufficient. Closed.**

4. **Path C: Causal wavelet denoising** (supported by papers #2 #6)
   - Current wdenoise uses global sequence (has future information leakage risk)
   - Switch to sliding window causal denoising
   - Expectation: Fixing leakage may lower IC, but more honest

5. **Path F: Confidence filtering** (supported by paper #7)
   - Based on existing LGB+XGB ensemble (CS_IC=+0.178)
   - Use MC Dropout or 5-seed voting
   - Don't pursue 70% on full sample, only achieve it on high-confidence subset

### Working Conventions (newly added)

- **Read papers to verify, then write plan, finally run code**——no skipping steps
- After each path completes, must produce comparison table (vs previous step baseline)
- Features/labels strictly prohibit future information leakage, must use strict time splits
- Rank correlation between tree models and LSTM is the key metric for ensemble benefit (< 0.6 = high benefit)
- Full-sample Hit 60% is a reasonable ceiling, don't pursue 70% (unless adding confidence filtering)

### Data Expansion Plan (2026-05-24)

**Goal**: Daily coverage from 300 stocks → all 3265 A-shares, daily+monthly integration coverage 11% → ~100%

**Steps**:
1. [ ] `download_daily_v2.py` — Scrapling downloads ~3000 missing stock daily data (in progress)
2. [ ] Retrain Daily LSTM (10-seed, full 3000+ stocks)
3. [ ] Daily→Monthly aggregation (coverage from 11% → ~100%)
4. [ ] Daily+Monthly re-integration, target IC +0.200

### Daily LSTM Optimization (launched 2026-05-23)

**Current baseline**: LSTM-7 (7 residual layers, hidden=128, lookback=252 days)
- IC range: +0.014 ~ +0.062 (5 seeds), extremely unstable
- Parameter count: 938K
- Val/test gap significant

**Core problem**: Overly deep architecture + overly long sequence → huge seed variance (4.4x), different seeds fall into different local optima

**Qlib benchmark** (A-share CSI300, 20-seed mean):
- LSTM (2 layers, h=64, Alpha360): IC=0.045±0.00 ← zero variance
- ALSTM (2 layers): IC=0.050±0.00
- HIST: Rank IC=0.067 ← strongest DL model
- LightGBM (Alpha158): IC=0.045

**Key insight**: All Qlib stable models are **2 layers + small hidden + short lookback**.

#### Daily Execution Paths

1. **Path 1: Architecture slimming** (highest priority)
   - 7 layers→2 layers, hidden 128→64, lookback 252→60 days
   - Parameter target <200K
   - 5 seeds run, target std<0.005

2. **Path 2: ALSTM attention** (supported by paper #2)
   - Input Attention + Temporal Attention
   - Qlib: ALSTM IC improvement 11-14% over LSTM

3. **Path 3: Input feature optimization**
   - Adopt Alpha360 style: 6 ratio features (not absolute values)
   - Principle: LSTM sees raw time series, tree model sees engineered indicators

4. **Path 4: Multi-seed ensemble**
   - 10-20 seeds take mean, directly eliminate variance

5. **Path 5: HIST concept graph** (requires industry data, advanced path)

**Expected target**: Stable IC 0.050-0.060, seed std<0.005

**Daily-Monthly integration**: Daily LSTM prediction → monthly aggregation → monthly LGB new feature

**References**:
- Qlib: github.com/microsoft/qlib, arXiv:2009.11189
- ALSTM: Qin et al. (2017) IJCAI, arXiv:1704.02971
- HIST: Xu et al. (2021) arXiv:2110.13716
- TRA: Lin et al. (2021) KDD
- SFM: Zhang et al. (2017) KDD

## Working Conventions for Claude Code

### Process Discipline (established 2026-06-05)

1. **Read literature/docs first, then act**: Any changes involving model architecture, feature engineering, training strategy,
   must first consult existing paper notes (`docs/p3-lstm-literature.md` etc.) and project documentation
   (`CLAUDE.md`, `PROGRESS.md`), confirm not already negated by existing conclusions. Do not rely on memory.

2. **Write tests before code**: Python scripts involving GPU training must first pass small-scale smoke
   test (20 stocks, 5 epochs, small hidden) verifying all modules work, no syntax errors, no
   numpy compatibility issues, JSON serialization normal, before launching full overnight run.

3. **Overfitting prevention is mandatory check**: All training scripts must include:
   - Strict time split (train/val/test by date, not random)
   - val-test gap monitoring (>0.05 → mark overfit)
   - Dropout + weight decay (at least one regularization method each)
   - Normalization parameters computed only from pre-train data (prevent leakage)

4. **Data leakage prevention is mandatory check**: Before commit, verify item by item:
   - Whether normalization mean/std only used data before training period
   - Whether rolling indicators (MA/MACD/RSI) only used data before cutoff
   - Whether label (forward return) calculation strictly only used data after cutoff

### Operating Conventions

- When changing prompt text, only modify constants in lib/build-prompt.js or lib/agents/*.js,
  do not change function signatures and code logic
- Changing cache key structure invalidates all old cache, do so carefully
- Do not reverse-modify src to make tests pass (test failure means src has a problem, pause and wait for user decision)
- When running JS tests, use ``node --test test/*.test.js test/agents/*.test.js``
- After completing a task, must tell user: changed file list, test pass status, key design points
- User prefers concise responses, no need for excessive explanation

### Overnight Experiment Pipeline

- **Core library**: `lib/overnight_core.py` (importable, does not trigger execution)
- **Smoke test**: `scripts/test_overnight.py` (20 stocks, run this first before full run)
- **Full runner**: `scripts/run_overnight.py` (runnable nightly, incremental continuation)
- **Logs**: `.eastmoney-ai/overnight_v2/overnight_YYYY-MM-DD.log`
- **Results**: `.eastmoney-ai/overnight_v2/results.jsonl` + `leaderboard-*.md`
