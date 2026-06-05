// service worker: receives content script messages, fetches Eastmoney K-lines + calls LLM API
// with result cache: re-analysis of the same stock/period/timeframe reads from chrome.storage.local directly, no API call
// provider abstraction layer supports multiple LLM providers
//
// cache key format: analysis:<market>.<code>:<period>:<bucket>:<style>:<mode>

import { parseStockUrl } from './lib/parse-url.js';
import { parseKlines } from './lib/parse-klines.js';
import { computeMA } from './lib/compute-ma.js';
import { computeMACD } from './lib/compute-macd.js';
import { buildPrompt, buildPromptByTemplate, buildMultiPeriodPrompt } from './lib/build-prompt.js';
import { getProvider } from './lib/llm/index.js';
import { estimateCost } from './lib/llm/pricing.js';
import { runDebate, clearDebateCheckpoint } from './lib/agents/runner.js';
import { extractStructuredOutput } from './lib/parse-structured-output.js';
import { checkCrossLevelConsistency } from './lib/cross-level-check.js';
import { HISTORY_KEY, MAX_HISTORY_ITEMS, MAX_HISTORY_BYTES, generateHistoryId, trimHistory, historyToMarkdown, formatHistoryDate } from './lib/history.js';
import { getFinancialsTool } from './lib/tools/get-financials.js';
import { getMoneyFlowTool } from './lib/tools/get-money-flow.js';
import { runHistoricalAnalysis, calculateActualReturn, buildSelfCalibrationBlock, getBacktestCache, setBacktestCache, backtestCacheKey } from './lib/self-backtest.js';

const EASTMONEY_KLINE_ENDPOINT = 'https://push2his.eastmoney.com/api/qt/stock/kline/get';

const DEFAULT_KLINE_LIMIT = 60;
const DEFAULT_MAX_TOKENS = 4000;

const PERIOD_KLTS = { monthly: '103', weekly: '102', daily: '101' };

const ERR = {
  NO_KEY: 'Please fill in your API key in the extension popup first',
  PARSE_URL: 'Unable to parse stock code from current page URL',
  EASTMONEY: 'Failed to fetch Eastmoney K-line data',
};

// ---- migrate legacy fields (v3a) ----

async function migrateSettings() {
  const items = await chrome.storage.local.get(['apiKey', 'model', 'provider', 'migrated:v3a']);
  if (items['migrated:v3a']) return;

  const updates = {};
  // old apiKey → apiKey:anthropic
  if (typeof items.apiKey === 'string' && items.apiKey.length > 0) {
    updates['apiKey:anthropic'] = items.apiKey;
  }
  // old model → model:anthropic
  if (typeof items.model === 'string' && items.model.length > 0) {
    updates['model:anthropic'] = items.model;
  }
  // default provider to anthropic
  if (!items.provider) {
    updates.provider = 'anthropic';
  }
  updates['migrated:v3a'] = true;

  if (Object.keys(updates).length > 1 || !items['migrated:v3a']) {
    await chrome.storage.local.set(updates);
  }
  // delete old fields
  if (items.apiKey !== undefined || items.model !== undefined) {
    await chrome.storage.local.remove(['apiKey', 'model']);
  }
}

// run once on service worker startup
migrateSettings().catch(() => {});

// ---- message routing ----

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === 'ANALYZE') {
    handleAnalyze(msg.url, { force: !!msg.force, pageEvents: msg.pageEvents, tabId: sender?.tab?.id })
      .then((result) => sendResponse({ ok: true, ...result }))
      .catch((err) => sendResponse({ ok: false, error: err.message || String(err) }));
    return true;
  }
  if (msg?.type === 'FOLLOW_UP') {
    handleFollowUp(msg)
      .then((result) => sendResponse({ ok: true, ...result }))
      .catch((err) => sendResponse({ ok: false, error: err.message || String(err) }));
    return true;
  }
  if (msg?.type === 'SAVE_CONVERSATION') {
    handleSaveConversation(msg.id, msg.conversationHistory)
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message || String(err) }));
    return true;
  }
  if (msg?.type === 'GET_HISTORY') {
    handleGetHistory()
      .then((result) => sendResponse({ ok: true, ...result }))
      .catch((err) => sendResponse({ ok: false, error: err.message || String(err) }));
    return true;
  }
  if (msg?.type === 'DELETE_HISTORY') {
    handleDeleteHistory(msg.id)
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message || String(err) }));
    return true;
  }
  if (msg?.type === 'CLEAR_HISTORY') {
    handleClearHistory()
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message || String(err) }));
    return true;
  }
  if (msg?.type === 'EXPORT_HISTORY') {
    handleExportHistory(msg.id)
      .then((result) => sendResponse({ ok: true, ...result }))
      .catch((err) => sendResponse({ ok: false, error: err.message || String(err) }));
    return true;
  }
  return false;
});

// ---- main analysis flow ----

async function handleAnalyze(pageUrl, opts = {}) {
  const force = !!opts.force;
  const tabId = opts.tabId || null;
  const startTime = Date.now();

  const parsed = parseStockUrl(pageUrl);
  if (!parsed) throw new Error(ERR.PARSE_URL);
  const { market, code } = parsed;

  const pageEvents = Array.isArray(opts.pageEvents) ? opts.pageEvents : [];

  const settings = await loadSettings();
  if (!settings.apiKey) throw new Error(ERR.NO_KEY);

  const style = settings.analysisStyle || 'technical';
  const period = settings.period || 'monthly';
  const isMulti = period === 'multi';
  const debateMode = isMulti ? false : !!settings.debateMode; // debate auto-disabled for multi-period
  const decisionMode = !!settings.decisionMode;
  const mode = debateMode ? 'debate' : 'single';
  const decision = decisionMode ? 'on' : 'off';

  const PERIOD_LABELS = { monthly: 'Monthly', weekly: 'Weekly', daily: 'Daily', multi: 'Multi-Period' };
  const periodLabel = PERIOD_LABELS[period] || 'Monthly';

  // ---- multi-period resonance branch ----
  if (isMulti) {
    return handleMultiPeriod({ market, code, settings, style, decisionMode, decision, extraContext: { events: pageEvents } });
  }

  // fetch data (stock + HS300 in parallel)
  const [eastmoney, hs300Data] = await Promise.all([
    fetchEastmoneyKlines(market, code, settings.klineLimit, period),
    fetchHS300Klines(settings.klineLimit, period),
  ]);
  const klines = parseKlines(eastmoney.klines);
  if (klines.length === 0) throw new Error(ERR.EASTMONEY);

  // HS300 indicator calculation
  let hs300IndexData = null;
  const hs300Klines = parseKlines(hs300Data ? hs300Data.klines : []);
  if (hs300Klines.length > 0) {
    const hs300Closes = hs300Klines.map((k) => k.close);
    const hs300MA5 = computeMA(hs300Closes, 5);
    const hs300MA20 = computeMA(hs300Closes, 20);
    const hs300MA60 = computeMA(hs300Closes, 60);
    const hs300MACD = computeMACD(hs300Closes);
    const hs300WithMA = hs300Klines.map((k, i) => ({
      ...k, ma5: hs300MA5[i], ma20: hs300MA20[i], ma60: hs300MA60[i], dif: hs300MACD.dif[i], dea: hs300MACD.dea[i], hist: hs300MACD.hist[i],
    }));
    hs300IndexData = { name: hs300Data.name || 'HS300', klines: hs300WithMA };
  }

  const latestDate = klines[klines.length - 1].date;
  const bucket = timeBucket(latestDate, period);
  const key = cacheKey(market, code, period, bucket, style, mode, decision);

  // cache hit
  if (!force) {
    const stored = (await chrome.storage.local.get([key]))[key];
    if (stored?.analysis) {
      return { ...stored, cached: true };
    }
  }

  // compute indicators
  const closes = klines.map((k) => k.close);
  const ma5 = computeMA(closes, 5);
  const ma20 = computeMA(closes, 20);
  const ma60 = computeMA(closes, 60);
  const { dif, dea, hist } = computeMACD(closes);
  const klinesWithMA = klines.map((k, i) => ({
    ...k, ma5: ma5[i], ma20: ma20[i], ma60: ma60[i], dif: dif[i], dea: dea[i], hist: hist[i],
  }));

  const provider = getProvider(settings.provider);
  const model = settings.model || provider.defaultModel;
  const extraContext = { events: pageEvents };

  // ---- self-backtest (single mode only, >= 36 bars) ----
  if (settings.enableSelfBacktest && mode === 'single' && klines.length >= 36) {
    try {
      const cutoffIndexes = klines.length >= 48
        ? [klines.length - 24, klines.length - 12]
        : [klines.length - 12];

      console.log(`[backtest] Starting self-backtest, cutoff=${cutoffIndexes.join(',')}, klines=${klines.length}`);

      const backtestResults = [];
      for (const cutoff of cutoffIndexes) {
        try {
          // check cache first (only cache LLM judgment, actual return computed live)
          const cacheKey = backtestCacheKey(code, settings.template, cutoff, settings.provider);
          const cached = await getBacktestCache(chrome.storage.local, cacheKey);

          let judgmentResult;
          if (cached) {
            console.log(`[backtest] cutoff=${cutoff} cache hit`);
            judgmentResult = cached;
          } else {
            console.log(`[backtest] cutoff=${cutoff} calling LLM...`);
            judgmentResult = await runHistoricalAnalysis(
              klinesWithMA, cutoff, settings.template,
              settings.provider,
              { ...settings, name: eastmoney.name, code, apiKey: settings.apiKey },
            );
            await setBacktestCache(chrome.storage.local, cacheKey, { date: judgmentResult.date, judgment: judgmentResult.judgment, keyLevels: judgmentResult.keyLevels });
          }

          // compute actual return (not cached, changes with live price)
          const actualReturn = calculateActualReturn(
            klinesWithMA, cutoff, klinesWithMA.length - 1,
            hs300IndexData ? hs300IndexData.klines : null,
          );

          backtestResults.push({
            date: judgmentResult.date,
            judgment: judgmentResult.judgment,
            keyLevels: judgmentResult.keyLevels,
            actualReturn,
          });
        } catch (err) {
          console.warn(`[backtest] cutoff=${cutoff} failed, skipped:`, err.message);
        }
      }

      if (backtestResults.length > 0) {
        extraContext.backtestBlock = buildSelfCalibrationBlock(backtestResults);
        console.log(`[backtest] Generated calibration block, ${backtestResults.length} time points`);
      }
    } catch (err) {
      console.warn('[backtest] Backtest flow failed, skipped:', err.message);
    }
  }

  console.log(`[analyze] ${settings.provider}/${model} K-line tail 3:`,
    klinesWithMA.slice(-3).map((k) => `${k.date} O=${k.open} H=${k.high} L=${k.low} C=${k.close}`));
  console.log(`[analyze] ${settings.provider}/${model} mode=${mode} klines:${klinesWithMA.length}`);

  let analysis;
  let cost = 0;
  let usage = null;
  let debateResult = null;
  let prompt = null;

  if (debateMode) {
    // multi-agent debate path
    const debateCtx = {
      name: eastmoney.name,
      code,
      period,
      periodLabel,
      klines: klinesWithMA,
      extraContext,
      decisionMode,
    };
    const checkpointKey = `debate-wip:${market}.${code}:${period}:${bucket}:${style}:${decision}`;
    debateResult = await runDebate(debateCtx, {
      provider: settings.provider,
      apiKey: settings.apiKey,
      model,
      maxTokens: DEFAULT_MAX_TOKENS,
      checkpointKey,
    });
    // render: Judge main output > three-agent fallback
    if (debateResult.judge) {
      analysis = debateResult.judge.text;
    } else {
      const sections = [];
      if (debateResult.partials.bull) sections.push('# Bullish Perspective\n\n' + debateResult.partials.bull.text);
      if (debateResult.partials.bear) sections.push('# Bearish Perspective\n\n' + debateResult.partials.bear.text);
      if (debateResult.partials.predictor) sections.push('# Key Price Level Identification\n\n' + debateResult.partials.predictor.text);
      for (const [role, err] of Object.entries(debateResult.errors || {})) {
        if (err) sections.push(`# ${role} call failed\n\n> ${err}`);
      }
      analysis = sections.join('\n\n---\n\n') || 'Debate mode generation failed';
    }
    cost = debateResult.totalCost;

    // monthly usage stats: accumulate all successful agents (incl. Judge)
    for (const agentResult of Object.values(debateResult.partials)) {
      if (agentResult?.usage) {
        await accumulateUsage(settings.provider, model, agentResult.usage, agentResult.cost);
      }
    }
    if (debateResult.judge?.usage) {
      await accumulateUsage(settings.provider, model, debateResult.judge.usage, debateResult.judge.cost);
    }
  } else {
    // single analysis path
    // attempt to fetch sector alpha (via Native Messaging querying SQLite)
    let sectorAlphaData = null;
    try {
      const alphaResp = await chrome.runtime.sendNativeMessage(NATIVE_HOST, {
        type: 'query_sector_alpha',
        code,
        period,
        lookback: 12,
      });
      if (alphaResp && alphaResp.type === 'sector_alpha' && alphaResp.data) {
        sectorAlphaData = alphaResp.data;
      }
    } catch (_) {
      // silent degrade when Native host unavailable
    }

    // Kronos signal injection (default ON — hold-out validation passed, see docs/p3-kronos-confirm.md)
    const ENABLE_KRONOS_SIGNAL = true;
    let kronosSignalData = null;
    if (ENABLE_KRONOS_SIGNAL) {
      try {
        const krResp = await chrome.runtime.sendNativeMessage(NATIVE_HOST, {
          type: 'read', key: `kronos/${code}`,
        });
        if (krResp && krResp.type === 'read_result' && krResp.data) {
          kronosSignalData = { prediction_6m_pct: krResp.data.prediction_6m_pct, direction: krResp.data.direction };
        }
      } catch (_) { /* degrade */ }
    }

    // Reversal factor withdrawn (24tp pool turned negative -19.2%, see docs/p3-signal-gating.md)

    // LSTM signal injection switch (default OFF — training data has time-window leakage, see docs/p1-lstm-leak-check.md)
    const ENABLE_LSTM_SIGNAL = false;

    let lstmSignalData = null;
    if (ENABLE_LSTM_SIGNAL) {
      try {
        const mcResp = await chrome.runtime.sendNativeMessage(NATIVE_HOST, {
          type: 'read',
          key: `mc_dropout/${code}`,
        });
        if (mcResp && mcResp.type === 'read_result' && mcResp.data) {
          const d = mcResp.data;
          const ulevel = d.uncertainty_level || 'medium';
          if (ulevel === 'high') {
            console.log(`[analyze] MC Dropout high uncertainty for ${code}, skipping LSTM signal`);
            lstmSignalData = null;
          } else {
            lstmSignalData = {
              lstm_signal: d.signal,
              lstm_signal_raw: d.signal_raw,
              y3_mean: d.y3_mean,
              y3_std: d.y3_std,
              y6_mean: d.y6_mean,
              y6_std: d.y6_std,
              overall_confidence: d.overall_confidence,
              uncertainty_level: ulevel,
              uncertainty_emoji: { low: 'LOW', medium: 'MEDIUM', high: 'HIGH' }[ulevel] || 'MEDIUM',
              uncertainty_desc: {
                low: 'Model predictions show strong consensus, signal reliability is high.',
                medium: 'Model predictions show divergence, signal requires technical verification.',
                high: 'Model predictions show significant divergence, signal is unreliable — rely on technical analysis.',
              }[ulevel] || '',
              mc_samples: 50,
            };
          }
        }
      } catch (_) {
        // silent degrade when Native host unavailable or data not pre-computed
      }
    }

    prompt = await buildPromptByTemplate({ templateKey: settings.template, name: eastmoney.name, code, market, klines: klinesWithMA, period, provider: settings.provider, extraContext, decisionMode, indexData: hs300IndexData, sectorAlphaData, lstmSignalData, kronosSignalData });
    console.log(`[analyze] ${settings.provider}/${model} prompt length:${prompt.length}`);

    // enable tool_use only for Anthropic provider
    const tools = settings.provider === 'anthropic'
      ? [getFinancialsTool, getMoneyFlowTool]
      : undefined;

    // streaming progress + tool call tracking
    const toolCalls = [];
    let toolCallSeq = 0;
    const onProgress = tabId ? (event) => {
      if (event.type === 'tool_start') {
        const seq = ++toolCallSeq;
        toolCalls.push({ seq, name: event.name, input: event.input, result: null, startMs: Date.now() });
      } else if (event.type === 'tool_result') {
        // search from back for same name and unfilled, to avoid same-name confusion
        for (let i = toolCalls.length - 1; i >= 0; i--) {
          if (toolCalls[i].name === event.name && toolCalls[i].result === null) {
            toolCalls[i].result = event.result;
            toolCalls[i].durationMs = Date.now() - toolCalls[i].startMs;
            break;
          }
        }
      }
      try {
        chrome.tabs.sendMessage(tabId, { type: 'STREAM_PROGRESS', event });
      } catch (_) { /* tab may have been closed */ }
    } : undefined;

    // keep SW awake with alarm during streaming
    let alarmName = null;
    if (tabId) {
      alarmName = `stream_${Date.now()}`;
      chrome.alarms.create(alarmName, { periodInMinutes: 0.5 });
    }

    let result;
    try {
      result = await provider.call(prompt, {
        model,
        apiKey: settings.apiKey,
        maxTokens: DEFAULT_MAX_TOKENS,
        tools,
        onProgress,
        enableThinking: settings.enableThinking,
      });
    } finally {
      // whether success or failure, clean up alarm + notify content
      if (alarmName) {
        try { chrome.alarms.clear(alarmName); } catch (_) { /* ignore */ }
        try { chrome.tabs.sendMessage(tabId, { type: 'STREAM_PROGRESS', event: { type: 'done' } }); } catch (_) { /* ignore */ }
      }
    }

    analysis = result.text;
    usage = result.usage || null;
    cost = result.usage ? estimateCost(settings.provider, model, result.usage) : 0;

    // debug log (only when switch enabled)
    if (settings.enableDebugLog) {
      try {
        const debugRecord = {
          timestamp: Date.now(),
          code, name: eastmoney.name,
          template: settings.template, provider: settings.provider, model,
          settings: {
            enableSelfBacktest: settings.enableSelfBacktest,
            enableThinking: settings.enableThinking,
            period: settings.period,
            decisionMode: settings.decisionMode,
          },
          fullPrompt: prompt,
          toolCalls,
          rawResponse: analysis,
          usage: result.usage || { inputTokens: 0, outputTokens: 0 },
          cost: { cny: cost },
          durationMs: Date.now() - startTime,
        };
        await chrome.storage.local.set({ 'debug:lastAnalysis': debugRecord });
      } catch (_) { /* ignore */ }
    }

    if (result.usage) {
      await accumulateUsage(settings.provider, model, result.usage, cost);
    }
  }

  // structured output extraction + cross-level consistency check
  const structured = extractStructuredOutput(analysis);
  const crossLevelWarnings = [];
  if (structured.data) {
    // TODO: structured key dimensions (only market/code/period) are fewer than analysis key (7 dimensions),
    // switching style causes new analysis to overwrite old structured data.
    // If structured output quality varies significantly across styles, add style to the key.
    const structuredKey = `structured:${market}.${code}:${period}`;
    await chrome.storage.local.set({
      [structuredKey]: { data: structured.data, timestamp: Date.now() },
    });

    if (period === 'daily' || period === 'weekly') {
      const parentPeriod = period === 'daily' ? 'weekly' : 'monthly';
      const parentKey = `structured:${market}.${code}:${parentPeriod}`;
      const parentItems = await chrome.storage.local.get([parentKey]);
      const parentRecord = parentItems[parentKey];
      if (parentRecord?.data) {
        const checkResult = checkCrossLevelConsistency(structured.data, parentRecord.data);
        crossLevelWarnings.push(...checkResult.warnings);
      }
    }
  }

  // history record ID
  const historyId = generateHistoryId();

  // write cache
  const record = {
    name: eastmoney.name,
    code,
    market,
    mode,
    decision: decision,
    period,
    bucket,
    monthKey: bucket,
    latestDate,
    klineCount: klines.length,
    provider: settings.provider,
    model,
    style,
    template: settings.template,
    prompt,
    extraContext,
    usage,
    cost,
    debate: debateResult,
    analyzedAt: Date.now(),
    analysis,
    ...(crossLevelWarnings.length > 0 && { crossLevelWarnings }),
  };
  await chrome.storage.local.set({ [key]: record });

  // debate checkpoint cleanup: formal cache written, no longer need resume data
  if (debateMode) {
    try { await clearDebateCheckpoint(`debate-wip:${market}.${code}:${period}:${bucket}:${style}:${decision}`); } catch (_) { /* ignore */ }
  }

  // write to analysis history (without conversationHistory, to be supplemented by content.js later)
  await saveToHistory({
    id: historyId,
    code,
    name: eastmoney.name,
    template: settings.template,
    provider: settings.provider,
    model,
    timestamp: Date.now(),
    analysis,
    conversationHistory: null,
    prompt,
  });

  return { ...record, cached: false, historyId };
}

// ---- multi-turn conversation ----

async function handleFollowUp(msg) {
  const settings = await loadSettings();
  if (!settings.apiKey) throw new Error(ERR.NO_KEY);

  const provider = getProvider(settings.provider);
  const model = settings.model || provider.defaultModel;

  const messages = Array.isArray(msg.history) ? [...msg.history] : [];
  messages.push({ role: 'user', content: msg.question });

  console.log(`[followup] ${settings.provider}/${model} history:${msg.history?.length || 0} messages`);

  const result = await provider.call(messages, {
    model,
    apiKey: settings.apiKey,
    maxTokens: DEFAULT_MAX_TOKENS,
  });

  const cost = result.usage ? estimateCost(settings.provider, model, result.usage) : 0;
  if (result.usage) {
    await accumulateUsage(settings.provider, model, result.usage, cost);
  }

  return { text: result.text, usage: result.usage || null, cost };
}

// ---- cache key ----

function cacheKey(market, code, period, bucket, style, mode = 'single', decision = 'off') {
  return `analysis:${market}.${code}:${period}:${bucket}:${style}:${mode}:${decision}`;
}

// ---- time bucket ----

function timeBucket(dateStr, period) {
  switch (period) {
    case 'daily':
      return String(dateStr).slice(0, 10);
    case 'weekly':
      return isoWeekFromDate(dateStr);
    case 'monthly':
    default:
      return String(dateStr).slice(0, 7);
  }
}

function isoWeekFromDate(dateStr) {
  const d = new Date(dateStr.replace(/\s.*$/, '') + 'T00:00:00');
  const dayOfWeek = d.getDay() || 7;
  d.setDate(d.getDate() + 4 - dayOfWeek);
  const year = d.getFullYear();
  const yearStart = new Date(year, 0, 1);
  const weekNum = Math.ceil(((d - yearStart) / 86400000 + 1) / 7);
  return `${year}-W${String(weekNum).padStart(2, '0')}`;
}

// ---- Eastmoney K-lines ----

async function fetchEastmoneyKlines(market, code, limit, period) {
  const klt = PERIOD_KLTS[period] || '103';
  const params = new URLSearchParams({
    secid: `${market}.${code}`,
    klt,
    fqt: '1',
    fields1: 'f1,f2,f3,f4,f5,f6',
    fields2: 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
    beg: '0',
    end: '20500101',
    lmt: String(limit),
  });

  let resp;
  try {
    resp = await fetch(`${EASTMONEY_KLINE_ENDPOINT}?${params.toString()}`);
  } catch (_) {
    throw new Error('Network error, please check your connection');
  }
  if (!resp.ok) throw new Error(ERR.EASTMONEY);

  const json = await resp.json();
  const data = json?.data;
  if (!data || !Array.isArray(data.klines)) throw new Error(ERR.EASTMONEY);
  return { name: data.name || code, klines: data.klines };
}

// ---- HS300 K-lines ----

async function fetchHS300Klines(limit, period) {
  const klt = PERIOD_KLTS[period] || '103';
  const params = new URLSearchParams({
    secid: '1.000300',
    klt,
    fqt: '1',
    fields1: 'f1,f2,f3,f4,f5,f6',
    fields2: 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
    beg: '0',
    end: '20500101',
    lmt: String(limit),
  });

  try {
    const resp = await fetch(`${EASTMONEY_KLINE_ENDPOINT}?${params.toString()}`);
    if (!resp.ok) return null;
    const json = await resp.json();
    const data = json?.data;
    if (!data || !Array.isArray(data.klines)) return null;
    return { name: data.name || 'HS300', klines: data.klines };
  } catch (_) {
    // index data fetch failure does not block main flow
    return null;
  }
}

// ---- monthly usage statistics ----

async function accumulateUsage(providerId, model, usage, cost) {
  const monthKey = new Date().toISOString().slice(0, 7);
  const storageKey = `usage:${providerId}:${monthKey}`;

  const items = await chrome.storage.local.get([storageKey]);
  let record = items[storageKey];
  if (!record) {
    record = {
      provider: providerId,
      month: monthKey,
      callCount: 0,
      inputTokens: 0,
      outputTokens: 0,
      totalCost: 0,
      byModel: {},
    };
  }

  record.callCount += 1;
  record.inputTokens += usage.inputTokens;
  record.outputTokens += usage.outputTokens;
  record.totalCost += cost;

  if (!record.byModel[model]) {
    record.byModel[model] = { callCount: 0, inputTokens: 0, outputTokens: 0, totalCost: 0 };
  }
  record.byModel[model].callCount += 1;
  record.byModel[model].inputTokens += usage.inputTokens;
  record.byModel[model].outputTokens += usage.outputTokens;
  record.byModel[model].totalCost += cost;

  await chrome.storage.local.set({ [storageKey]: record });
}

export async function getMonthlyUsage(providerId, monthKey) {
  const items = await chrome.storage.local.get([`usage:${providerId}:${monthKey}`]);
  return items[`usage:${providerId}:${monthKey}`] || null;
}

// ---- multi-period resonance ----

async function handleMultiPeriod({ market, code, settings, style, decisionMode, decision, extraContext }) {
  const limit = settings.klineLimit;

  // three-period concurrent fetch + HS300 monthly
  const [monthlyData, weeklyData, dailyData, hs300Data] = await Promise.all([
    fetchEastmoneyKlines(market, code, limit, 'monthly'),
    fetchEastmoneyKlines(market, code, limit, 'weekly'),
    fetchEastmoneyKlines(market, code, limit, 'daily'),
    fetchHS300Klines(limit, 'monthly'),
  ]);

  const monthlyKlines = parseKlines(monthlyData.klines);
  const weeklyKlines = parseKlines(weeklyData.klines);
  const dailyKlines = parseKlines(dailyData.klines);

  if (monthlyKlines.length === 0 && weeklyKlines.length === 0 && dailyKlines.length === 0) {
    throw new Error(ERR.EASTMONEY);
  }

  // compute MA + MACD for each
  function attachIndicators(klines) {
    const closes = klines.map((k) => k.close);
    const ma5 = computeMA(closes, 5);
    const ma20 = computeMA(closes, 20);
    const ma60 = computeMA(closes, 60);
    const { dif, dea, hist } = computeMACD(closes);
    return klines.map((k, i) => ({
      ...k, ma5: ma5[i], ma20: ma20[i], ma60: ma60[i], dif: dif[i], dea: dea[i], hist: hist[i],
    }));
  }

  const monthlyWithMA = attachIndicators(monthlyKlines);
  const weeklyWithMA = attachIndicators(weeklyKlines);
  const dailyWithMA = attachIndicators(dailyKlines);

  // HS300 monthly data
  let hs300IndexData = null;
  const hs300Klines = hs300Data ? parseKlines(hs300Data.klines) : [];
  if (hs300Klines.length > 0) {
    const hs300WithMA = attachIndicators(hs300Klines);
    hs300IndexData = { name: hs300Data.name || 'HS300', klines: hs300WithMA };
  }

  // bucket by daily latest date (finest granularity)
  const latestDate = dailyKlines.length > 0 ? dailyKlines[dailyKlines.length - 1].date
    : (weeklyKlines.length > 0 ? weeklyKlines[weeklyKlines.length - 1].date
      : monthlyKlines[monthlyKlines.length - 1].date);
  const bucket = timeBucket(latestDate, 'daily');
  const key = cacheKey(market, code, 'multi', bucket, style, 'single', decision);

  // multi mode not cached for now (prompt is large, low cache value)

  if (!settings.apiKey) throw new Error(ERR.NO_KEY);

  const prompt = buildMultiPeriodPrompt({
    name: monthlyData.name || code,
    code,
    monthlyKlines: monthlyWithMA,
    weeklyKlines: weeklyWithMA,
    dailyKlines: dailyWithMA,
    style,
    provider: settings.provider,
    extraContext,
    decisionMode,
    indexData: hs300IndexData,
  });

  const provider = getProvider(settings.provider);
  const model = settings.model || provider.defaultModel;

  console.log(`[analyze] multi/${settings.provider}/${model} monthly:${monthlyKlines.length} weekly:${weeklyKlines.length} daily:${dailyKlines.length} prompt:${prompt.length}`);

  const result = await provider.call(prompt, {
    model,
    apiKey: settings.apiKey,
    maxTokens: DEFAULT_MAX_TOKENS * 2, // multi-period output needs more tokens
  });

  const analysis = result.text;
  const usage = result.usage || null;
  const cost = result.usage ? estimateCost(settings.provider, model, result.usage) : 0;

  if (result.usage) {
    await accumulateUsage(settings.provider, model, result.usage, cost);
  }

  // history record ID
  const historyId = generateHistoryId();

  const record = {
    name: monthlyData.name || code,
    code,
    market,
    mode: 'single',
    decision,
    period: 'multi',
    bucket,
    monthKey: bucket,
    latestDate,
    klineCount: monthlyKlines.length + weeklyKlines.length + dailyKlines.length,
    provider: settings.provider,
    model,
    style,
    template: settings.template,
    prompt,
    extraContext,
    usage,
    cost,
    debate: null,
    analyzedAt: Date.now(),
    analysis,
  };

  await chrome.storage.local.set({ [key]: record });

  // write to analysis history
  await saveToHistory({
    id: historyId,
    code,
    name: monthlyData.name || code,
    template: settings.template,
    provider: settings.provider,
    model,
    timestamp: Date.now(),
    analysis,
    conversationHistory: null,
    prompt,
  });

  return { ...record, cached: false, historyId };
}

// ---- settings ----

async function loadSettings() {
  // run migration first (idempotent)
  await migrateSettings();

  const allItems = await chrome.storage.local.get([
    'provider', 'klineLimit', 'analysisStyle', 'period', 'debateMode', 'decisionMode',
    'analysisDepth', 'template',
    'apiKey:anthropic', 'apiKey:deepseek',
    'model:anthropic', 'model:deepseek',
    'enableSelfBacktest', 'enableThinking', 'enableDebugLog',
  ]);

  const provider = allItems.provider || 'anthropic';
  const analysisDepth = allItems.analysisDepth || 'standard';

  // model selection priority:
  // 1. user manually entered model:${provider} (non-empty)
  // 2. analysisDepth mapping (Anthropic only): standard → claude-sonnet-4-6, deep → claude-opus-4-7
  // 3. provider default model
  let model = allItems[`model:${provider}`] || '';
  if (!model && provider === 'anthropic') {
    model = analysisDepth === 'deep' ? 'claude-opus-4-7' : 'claude-sonnet-4-6';
  }

  return {
    provider,
    apiKey: allItems[`apiKey:${provider}`] || '',
    model,
    klineLimit: Number(allItems.klineLimit) || DEFAULT_KLINE_LIMIT,
    template: allItems.template || 'technical',
    analysisStyle: allItems.analysisStyle || 'technical',
    period: allItems.period || 'monthly',
    debateMode: allItems.debateMode || false,
    decisionMode: allItems.decisionMode || false,
    enableSelfBacktest: allItems.enableSelfBacktest !== undefined ? !!allItems.enableSelfBacktest : true,
    enableThinking: !!allItems.enableThinking,
    enableDebugLog: !!allItems.enableDebugLog,
  };
}

// ---- analysis history ----

async function saveToHistory(entry) {
  const items = await chrome.storage.local.get([HISTORY_KEY]);
  let list = Array.isArray(items[HISTORY_KEY]) ? items[HISTORY_KEY] : [];

  // dedup: same code + template + timestamp (second-level) treated as duplicate
  const dupIdx = list.findIndex((e) => e.code === entry.code && e.template === entry.template && Math.abs((e.timestamp || 0) - (entry.timestamp || 0)) < 2000);
  if (dupIdx >= 0) {
    list[dupIdx] = entry;
    await persistHistory(list);
    return;
  }

  list.push(entry);

  // capacity management
  list = trimHistory(list);
  await persistHistory(list);
}

async function persistHistory(list) {
  // trimHistory already handles count limit, here only check size
  let json = JSON.stringify(list);
  while (json.length > MAX_HISTORY_BYTES && list.length > 1) {
    list.shift();
    json = JSON.stringify(list);
  }

  await chrome.storage.local.set({ [HISTORY_KEY]: list });
}

async function handleSaveConversation(id, conversationHistory) {
  const items = await chrome.storage.local.get([HISTORY_KEY]);
  const list = Array.isArray(items[HISTORY_KEY]) ? items[HISTORY_KEY] : [];
  const idx = list.findIndex((e) => e.id === id);
  if (idx < 0) throw new Error('History record does not exist');
  list[idx].conversationHistory = conversationHistory;
  // update timestamp
  list[idx].timestamp = Date.now();
  await persistHistory(list);
}

async function handleGetHistory() {
  const items = await chrome.storage.local.get([HISTORY_KEY]);
  const list = Array.isArray(items[HISTORY_KEY]) ? items[HISTORY_KEY] : [];
  // estimate usage
  let bytes = 0;
  try {
    bytes = JSON.stringify(list).length;
  } catch (_) { /* ignore */ }
  return {
    list: list.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0)),
    count: list.length,
    bytes,
    maxBytes: MAX_HISTORY_BYTES,
    maxItems: MAX_HISTORY_ITEMS,
  };
}

async function handleDeleteHistory(id) {
  const items = await chrome.storage.local.get([HISTORY_KEY]);
  let list = Array.isArray(items[HISTORY_KEY]) ? items[HISTORY_KEY] : [];
  list = list.filter((e) => e.id !== id);
  await chrome.storage.local.set({ [HISTORY_KEY]: list });
}

async function handleClearHistory() {
  await chrome.storage.local.remove([HISTORY_KEY]);
}

async function handleExportHistory(id) {
  const items = await chrome.storage.local.get([HISTORY_KEY]);
  const list = Array.isArray(items[HISTORY_KEY]) ? items[HISTORY_KEY] : [];
  if (id) {
    // export single
    const entry = list.find((e) => e.id === id);
    if (!entry) throw new Error('Record does not exist');
    const md = historyToMarkdown(entry);
    return { markdown: md, filename: `${entry.name || entry.code}_${formatHistoryDate(entry.timestamp)}.md` };
  }
  // export all
  const md = list.map(historyToMarkdown).join('\n\n---\n\n');
  return { markdown: md, filename: `analysis_history_${formatHistoryDate(Date.now())}.md`, count: list.length };
}

// ---- Native Messaging auto-sync to Node side ----

const NATIVE_HOST = 'com.eastmoney_ai.sync';

// sensitive key prefixes not to sync
const SYNC_SKIP_PREFIXES = [
  'apiKey:',
  'model:',
  'migrated:',
];

function shouldSyncKey(key) {
  return !SYNC_SKIP_PREFIXES.some((prefix) => key.startsWith(prefix));
}

// debounce: accumulate changes for same key, batch-send after 500ms
const syncQueue = new Map();
let syncTimer = null;

function enqueueSync(key, newValue) {
  if (!shouldSyncKey(key)) return;
  syncQueue.set(key, newValue);
  if (syncTimer) clearTimeout(syncTimer);
  syncTimer = setTimeout(flushSync, 500);
}

async function flushSync() {
  const items = {};
  for (const [k, v] of syncQueue) {
    items[k] = v;
  }
  syncQueue.clear();
  syncTimer = null;

  try {
    await chrome.runtime.sendNativeMessage(NATIVE_HOST, {
      type: 'sync_batch',
      items,
    });
  } catch (_) {
    // Native host not installed or startup failed, silently ignore
    // data remains in chrome.storage.local, will sync on next successful attempt
  }
}

// listen for storage changes (trigger auto-sync when content.js / popup.js writes)
chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== 'local') return;
  for (const [key, change] of Object.entries(changes)) {
    if (change.newValue !== undefined && shouldSyncKey(key)) {
      enqueueSync(key, change.newValue);
    }
  }
});
