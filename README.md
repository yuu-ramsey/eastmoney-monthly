# eastmoney-monthly-ai

**A regime-adaptive analysis system for China A-share equities — designed to help you avoid losses, not chase profits.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Node.js 18+](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)

> ⚠️ **This is a personal research project, not a commercial product.**
> Updates and bug fixes happen on my own schedule — there are no guarantees of timely patches, backward compatibility, or continued development.
> **Use entirely at your own risk.**

---

## 📌 News

- 🚩 **[2026.06]** Regime-Adaptive MoE architecture designed (v2). Three-detector voting for regime detection, no single point of failure.
- 🚩 **[2026.05]** Survivorship bias quantified at **8.4 percentage points** — rebuilt evaluation pool with Baostock (includes delisted stocks). All prior signal conclusions invalidated and re-evaluated.
- 🚩 **[2026.05]** P0–P3 research cycle completed: 6 signal sources tested on 24-timepoint unbiased pool. Kronos is the only ML model that passed hold-out validation.
- 🚩 **[2026.05]** Multi-agent debate system (Bull/Bear/Predictor/Judge) with checkpoint/resume for Chrome MV3 service worker lifecycle.
- 🚩 **[2026.04]** Initial Chrome Extension (MV3) + Node.js CLI dual-entry system.

---

## What This Project Is

A multi-signal analysis system for monthly-frequency A-share technical analysis. It combines statistical factors, machine learning models, and LLM-based interpretation under a regime-adaptive framework.

**Core philosophy:** No single model works in all market conditions. The system detects the current market regime and activates only the signals that have been validated for that specific environment.

### What It Does

- 📉 **Flags risk** — Tells you when signals conflict, when confidence is low, and when **not** to act. Knowing when to stay out is more valuable than knowing when to get in.
- 📊 **Multi-signal analysis** — Combines independently validated signals (statistical factors, transformer predictions, LLM interpretation) rather than relying on any single model.
- 🔄 **Regime awareness** — Different market conditions (trending, volatile, sideways) activate different signal combinations.
- 📝 **Plain-language explanation** — LLM translates quantitative signals into readable technical analysis. The LLM explains; it does not decide.

### What It Does NOT Do

- ❌ Guarantee profits or positive returns
- ❌ Execute trades or connect to any brokerage
- ❌ Provide real-time signals (monthly frequency only)
- ❌ Replace professional financial advice

---

## Architecture

```
Layer 1: Perception          Multi-scale features (monthly/weekly/daily) + data quality gate
            ↓
Layer 2: Regime Detection    Three independent detectors (HMM / volatility / trend) → majority vote
            ↓                No LLM in the loop — pure statistics and math
Layer 3: Expert Pool         Multiple signals, each validated per-regime
            ↓                Mechanical weight lookup — no model decides the weights
Layer 4: Output              Weighted signal (math) + LLM interpretation (text, isolated from signal chain)
```

**Design principles:**
- **No single point of failure** — Any component can fail without breaking the system
- **LLM isolation** — LLM generates explanations only; it never touches signal routing or weighting
- **Mechanical gating** — Regime → weight mapping is a hardcoded lookup table, not a model prediction

---

## Signal Sources

Validated on a 24-timepoint unbiased evaluation pool (3,100+ stock-timepoint pairs, includes delisted stocks via Baostock):

| Signal | Type | Hold-out Test CI | Status |
|--------|------|-----------------|--------|
| Kronos | External Transformer | [+5.1, +15.3] | ✅ Validated |
| LLM Analysis | LLM (Anthropic/DeepSeek) | Under re-evaluation | 🔄 Active (interpreter role) |
| Reversal Factor | Statistical Factor | Per-regime validation pending | 🔄 Experimental |
| Momentum Factor | Statistical Factor | Per-regime validation pending | 🔄 Experimental |
| GRU (Triple-Barrier) | Own ML Model | Borderline | 🧪 Research |
| LightGBM | Own ML Model | Regime-dependent | 🧪 Research |
| LSTM (Daily) | Own ML Model | Training failed | ⏸️ Suspended |

---

## Running Modes

The analysis engine is **platform-independent**. The Chrome Extension is one interface, not the only one.

| Interface | Description | Use Case |
|-----------|-------------|----------|
| **Chrome Extension (MV3)** | Analyze stocks directly on Eastmoney pages | Real-time single-stock analysis |
| **Node.js CLI** | `cli/` commands for batch analysis and evaluation | Batch processing, evaluation, data pipeline |
| **Python Scripts** | `scripts/` and `lib/` for ML training and research | Model training, backtesting, data construction |
| **Native Host** | Bridge between extension and local compute | SQLite access, ML model inference |
| **Future** | Standalone desktop UI, API service | Not yet implemented |

---

## Research Findings (P0–P3)

Key methodological discoveries from the validation cycle:

- **Survivorship bias = 8.4pp.** Excluding delisted stocks inflated low-position stock returns by 8.4 percentage points and flipped the reversal factor's sign from +6.6% to −19.2%. All conclusions from survivor-only pools were artifacts.
- **Evaluation metric was pathological.** The original scoring matrix rewarded "neutral" predictions with a 0.3 floor, making "always predict neutral" the optimal strategy (score 0.401 vs. best model 0.197).
- **Complex models underperformed simple ones on monthly data.** LSTM, GRU, and LightGBM all failed hold-out validation. The pattern is consistent: monthly-frequency data cannot feed deep learning models enough signal.
- **LLM is too conservative.** 70.9% of LLM predictions are "neutral" — but when it does commit (strong_bull, n=30), alpha is +26%. The capability exists; the prompt suppresses it.

Full research documentation in `docs/`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Runtime | Chrome Extension (MV3) + Node.js native host |
| Research | Python 3.10+ (PyTorch, scikit-learn, LightGBM, hmmlearn) |
| Data | [Baostock](http://baostock.com) (free, no registration, includes delisted stocks) |
| LLM | Anthropic Claude / DeepSeek (configurable) |
| Database | SQLite (better-sqlite3) |
| Technical Indicators | Self-implemented (15 indicators, zero npm dependencies) |

---

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- Chrome browser (for extension mode)

### Installation

```bash
# Clone
git clone https://github.com/<your-username>/eastmoney-monthly-ai.git
cd eastmoney-monthly-ai

# Node.js dependencies
npm install

# Python dependencies (research/ML)
pip install -r requirements.txt

# Download Kronos pretrained weights (MIT License)
python kronos/download_weights.py
```

### Chrome Extension
1. Open `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" → select the project root
4. Navigate to any stock page on Eastmoney

### CLI Analysis
```bash
node cli/index.js analyze 600519    # Analyze a single stock
node cli/index.js batch             # Batch analysis
```

---

## Acknowledgments & References

### Code-level References

- **[Kronos](https://github.com/shiyu-coder/Kronos)** — Financial K-line foundation model (AAAI 2026, [arXiv:2508.02739](https://arxiv.org/abs/2508.02739)). Core prediction modules reproduced from this project. Pretrained weights from [HuggingFace](https://huggingface.co/NeoQuasar/Kronos-base). MIT License. See `kronos/LICENSE`.
- **[Microsoft Qlib](https://github.com/microsoft/qlib)** — AI-oriented quantitative investment platform. Used as architecture reference and A-share daily performance benchmark for LSTM/ALSTM models. MIT License.
- **[Baostock](http://baostock.com)** — Free, open-source A-share market data provider including delisted stock history. Primary data source for unbiased evaluation.
- **[AkShare](https://akshare.akfamily.xyz)** — Free financial data interface library. Used for industry classification data. MIT License.

### Methodology References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. — Triple-Barrier labeling method.
- Qin, Y. et al. (2017). A Dual-Stage Attention-Based RNN for Time Series Prediction. IJCAI 2017. [arXiv:1704.02971](https://arxiv.org/abs/1704.02971) — ALSTM architecture reference.
- Xu, W. et al. (2021). HIST: A Graph-based Framework for Stock Trend Forecasting. [arXiv:2110.13716](https://arxiv.org/abs/2110.13716) — Concept-oriented architecture reference.
- TradingAgents (2024). [arXiv:2412.20138](https://arxiv.org/abs/2412.20138) — Multi-agent debate architecture inspiration.

### Architecture Design References

See `docs/regime-adaptive-moe-architecture.md` for the complete architecture document with full literature references, including:
Gupta et al. 2025 (HMM ensemble voting), Vallarino 2025 (MoE framework, [arXiv:2508.02686](https://arxiv.org/abs/2508.02686)), LLMoE 2025 ([arXiv:2501.09636](https://arxiv.org/abs/2501.09636)), Meta-LMPS 2025, AEDL 2025, PortRSMs 2025, and others.

---

## ⚠️ Disclaimer

This software is provided for **educational and research purposes only**. It is not financial advice, and no part of this system constitutes a recommendation to buy, sell, or hold any security.

**This tool is designed to help identify risk, not to generate trading signals.** Telling you when NOT to act is its primary function. If you use any output from this system as the sole basis for trading decisions, you do so entirely at your own risk.

The author makes no warranties regarding accuracy, completeness, or reliability of any analysis produced by this system. Past statistical validation does not guarantee future performance. Markets are inherently unpredictable.

---

## License

This project's own code is released under the [MIT License](LICENSE).

Third-party components retain their original licenses:
- `kronos/` — MIT License (Copyright © 2025 ShiYu). See [`kronos/LICENSE`](kronos/LICENSE).
- Pretrained weights ([NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)) — MIT License.
