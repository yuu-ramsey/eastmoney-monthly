// service worker: 接收 content script 消息,fetch 东财 K 线 + 调 LLM API
// 带结果缓存:同股同周期同时段二次分析直接读 chrome.storage.local,不再调 API
// provider 抽象层支持多 LLM 提供商
//
// 缓存 key 格式:analysis:<market>.<code>:<period>:<bucket>:<style>:<mode>

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
  NO_KEY: '请先在扩展弹窗里填入 API key',
  PARSE_URL: '无法从当前页 URL 解析股票代码',
  EASTMONEY: '获取东财 K 线数据失败',
};

// ---- 旧字段迁移（v3a）----

async function migrateSettings() {
  const items = await chrome.storage.local.get(['apiKey', 'model', 'provider', 'migrated:v3a']);
  if (items['migrated:v3a']) return;

  const updates = {};
  // 旧 apiKey → apiKey:anthropic
  if (typeof items.apiKey === 'string' && items.apiKey.length > 0) {
    updates['apiKey:anthropic'] = items.apiKey;
  }
  // 旧 model → model:anthropic
  if (typeof items.model === 'string' && items.model.length > 0) {
    updates['model:anthropic'] = items.model;
  }
  // provider 默认 anthropic
  if (!items.provider) {
    updates.provider = 'anthropic';
  }
  updates['migrated:v3a'] = true;

  if (Object.keys(updates).length > 1 || !items['migrated:v3a']) {
    await chrome.storage.local.set(updates);
  }
  // 删除旧字段
  if (items.apiKey !== undefined || items.model !== undefined) {
    await chrome.storage.local.remove(['apiKey', 'model']);
  }
}

// service worker 启动时执行一次
migrateSettings().catch(() => {});

// ---- 消息路由 ----

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

// ---- 分析主流程 ----

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
  const debateMode = isMulti ? false : !!settings.debateMode; // 多周期下辩论自动禁用
  const decisionMode = !!settings.decisionMode;
  const mode = debateMode ? 'debate' : 'single';
  const decision = decisionMode ? 'on' : 'off';

  const PERIOD_LABELS = { monthly: '月线', weekly: '周线', daily: '日线', multi: '多周期' };
  const periodLabel = PERIOD_LABELS[period] || '月线';

  // ---- 多周期共振分支 ----
  if (isMulti) {
    return handleMultiPeriod({ market, code, settings, style, decisionMode, decision, extraContext: { events: pageEvents } });
  }

  // 取数据（个股 + 沪深300 并行）
  const [eastmoney, hs300Data] = await Promise.all([
    fetchEastmoneyKlines(market, code, settings.klineLimit, period),
    fetchHS300Klines(settings.klineLimit, period),
  ]);
  const klines = parseKlines(eastmoney.klines);
  if (klines.length === 0) throw new Error(ERR.EASTMONEY);

  // 沪深300 指标计算
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
    hs300IndexData = { name: hs300Data.name || '沪深300', klines: hs300WithMA };
  }

  const latestDate = klines[klines.length - 1].date;
  const bucket = timeBucket(latestDate, period);
  const key = cacheKey(market, code, period, bucket, style, mode, decision);

  // 命中缓存
  if (!force) {
    const stored = (await chrome.storage.local.get([key]))[key];
    if (stored?.analysis) {
      return { ...stored, cached: true };
    }
  }

  // 计算指标
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

  // ---- 自我回测（仅 single 模式，数据 ≥36 根） ----
  if (settings.enableSelfBacktest && mode === 'single' && klines.length >= 36) {
    try {
      const cutoffIndexes = klines.length >= 48
        ? [klines.length - 24, klines.length - 12]
        : [klines.length - 12];

      console.log(`[backtest] 开始自我回测，cutoff=${cutoffIndexes.join(',')}，K线=${klines.length}根`);

      const backtestResults = [];
      for (const cutoff of cutoffIndexes) {
        try {
          // 先查缓存（仅缓存 LLM 判断，actual return 实时计算）
          const cacheKey = backtestCacheKey(code, settings.template, cutoff, settings.provider);
          const cached = await getBacktestCache(chrome.storage.local, cacheKey);

          let judgmentResult;
          if (cached) {
            console.log(`[backtest] cutoff=${cutoff} 命中缓存`);
            judgmentResult = cached;
          } else {
            console.log(`[backtest] cutoff=${cutoff} 调 LLM...`);
            judgmentResult = await runHistoricalAnalysis(
              klinesWithMA, cutoff, settings.template,
              settings.provider,
              { ...settings, name: eastmoney.name, code, apiKey: settings.apiKey },
            );
            await setBacktestCache(chrome.storage.local, cacheKey, { date: judgmentResult.date, judgment: judgmentResult.judgment, keyLevels: judgmentResult.keyLevels });
          }

          // 计算实际涨跌（不缓存，随价格实时变化）
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
          console.warn(`[backtest] cutoff=${cutoff} 失败，跳过：`, err.message);
        }
      }

      if (backtestResults.length > 0) {
        extraContext.backtestBlock = buildSelfCalibrationBlock(backtestResults);
        console.log(`[backtest] 生成校准段，${backtestResults.length} 个时点`);
      }
    } catch (err) {
      console.warn('[backtest] 回测流程失败，跳过：', err.message);
    }
  }

  console.log(`[analyze] ${settings.provider}/${model} K线尾3行:`,
    klinesWithMA.slice(-3).map((k) => `${k.date} O=${k.open} H=${k.high} L=${k.low} C=${k.close}`));
  console.log(`[analyze] ${settings.provider}/${model} mode=${mode} K线:${klinesWithMA.length}根`);

  let analysis;
  let cost = 0;
  let usage = null;
  let debateResult = null;
  let prompt = null;

  if (debateMode) {
    // 多 Agent 辩论路径
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
    // 渲染：Judge 主输出 > 三段拼接退路
    if (debateResult.judge) {
      analysis = debateResult.judge.text;
    } else {
      const sections = [];
      if (debateResult.partials.bull) sections.push('# 看多视角\n\n' + debateResult.partials.bull.text);
      if (debateResult.partials.bear) sections.push('# 看空视角\n\n' + debateResult.partials.bear.text);
      if (debateResult.partials.predictor) sections.push('# 关键价位识别\n\n' + debateResult.partials.predictor.text);
      for (const [role, err] of Object.entries(debateResult.errors || {})) {
        if (err) sections.push(`# ${role} 调用失败\n\n> ${err}`);
      }
      analysis = sections.join('\n\n---\n\n') || '辩论模式生成失败';
    }
    cost = debateResult.totalCost;

    // 月度用量统计:累加所有成功 Agent（含 Judge）
    for (const agentResult of Object.values(debateResult.partials)) {
      if (agentResult?.usage) {
        await accumulateUsage(settings.provider, model, agentResult.usage, agentResult.cost);
      }
    }
    if (debateResult.judge?.usage) {
      await accumulateUsage(settings.provider, model, debateResult.judge.usage, debateResult.judge.cost);
    }
  } else {
    // 单次分析路径
    // 尝试获取行业 alpha（通过 Native Messaging 查询 SQLite）
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
      // Native host 不可用时静默降级
    }

    // Kronos 信号注入（默认 ON — hold-out 验证通过，参见 docs/p3-kronos-confirm.md）
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
      } catch (_) { /* 降级 */ }
    }

    // 反转因子信号（默认 ON — hold-out 验证通过，参见 docs/p3-signal-gating.md）
    const ENABLE_REVERSAL_SIGNAL = true;
    let reversalSignalData = null;
    if (ENABLE_REVERSAL_SIGNAL && klinesWithMA && klinesWithMA.length >= 60) {
      const closes = klinesWithMA.map(k => k.close);
      const n = klinesWithMA.length;
      const ci = n - 1; // cutoff bar is the last bar
      const rev1m = ci >= 1 ? (closes[ci] - closes[ci - 1]) / closes[ci - 1] : 0;
      const rev3m = ci >= 3 ? (closes[ci] - closes[ci - 3]) / closes[ci - 3] : 0;
      const ma60 = closes.slice(Math.max(0, ci - 60), ci + 1).reduce((a, b) => a + b, 0) / Math.min(60, ci + 1);
      const ma60disc = (closes[ci] - ma60) / ma60;
      reversalSignalData = { rev1m, rev3m, ma60disc, composite: rev1m + rev3m + ma60disc };
    }

    // LSTM 信号注入开关（默认 OFF — 训练数据存在时间窗泄漏，参见 docs/p1-lstm-leak-check.md）
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
            console.log(`[analyze] MC Dropout high uncertainty for ${code}, 跳过 LSTM 信号`);
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
              uncertainty_emoji: { low: '🟢', medium: '🟡', high: '🔴' }[ulevel] || '🟡',
              uncertainty_desc: {
                low: '模型预测一致性强，信号可信度较高',
                medium: '模型预测存在分歧，信号需结合技术面验证',
                high: '模型预测分歧大，信号不可靠，以技术分析为主',
              }[ulevel] || '',
              mc_samples: 50,
            };
          }
        }
      } catch (_) {
        // Native host 不可用或数据未预计算时静默降级
      }
    }

    prompt = await buildPromptByTemplate({ templateKey: settings.template, name: eastmoney.name, code, market, klines: klinesWithMA, period, provider: settings.provider, extraContext, decisionMode, indexData: hs300IndexData, sectorAlphaData, lstmSignalData, kronosSignalData, reversalSignalData });
    console.log(`[analyze] ${settings.provider}/${model} prompt长度:${prompt.length}`);

    // 仅 Anthropic provider 启用 tool_use
    const tools = settings.provider === 'anthropic'
      ? [getFinancialsTool, getMoneyFlowTool]
      : undefined;

    // 流式进度 + 工具调用跟踪
    const toolCalls = [];
    let toolCallSeq = 0;
    const onProgress = tabId ? (event) => {
      if (event.type === 'tool_start') {
        const seq = ++toolCallSeq;
        toolCalls.push({ seq, name: event.name, input: event.input, result: null, startMs: Date.now() });
      } else if (event.type === 'tool_result') {
        // 从后往前找同名且未填充的，避免同名混淆
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
      } catch (_) { /* tab 可能已关闭 */ }
    } : undefined;

    // 流式期间用 alarm 保持 SW 唤醒
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
      // 无论成功/失败，清理 alarm + 通知 content
      if (alarmName) {
        try { chrome.alarms.clear(alarmName); } catch (_) { /* ignore */ }
        try { chrome.tabs.sendMessage(tabId, { type: 'STREAM_PROGRESS', event: { type: 'done' } }); } catch (_) { /* ignore */ }
      }
    }

    analysis = result.text;
    usage = result.usage || null;
    cost = result.usage ? estimateCost(settings.provider, model, result.usage) : 0;

    // 调试日志（仅当开关开启）
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

  // 结构化输出提取 + 跨级别一致性校验
  const structured = extractStructuredOutput(analysis);
  const crossLevelWarnings = [];
  if (structured.data) {
    // TODO: structured key 维度（仅 market/code/period）少于 analysis key（7 维度），
    // 切换 style 时新分析会覆盖旧 structured 数据。
    // 若不同风格输出的 structured 质量差异大，需将 style 加入 key。
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

  // 历史记录 ID
  const historyId = generateHistoryId();

  // 写缓存
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

  // 辩论 checkpoint 清理：正式缓存已写入，不再需要续跑数据
  if (debateMode) {
    try { await clearDebateCheckpoint(`debate-wip:${market}.${code}:${period}:${bucket}:${style}:${decision}`); } catch (_) { /* ignore */ }
  }

  // 写入分析历史（不含 conversationHistory，后续由 content.js 补充）
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

// ---- 多轮对话 ----

async function handleFollowUp(msg) {
  const settings = await loadSettings();
  if (!settings.apiKey) throw new Error(ERR.NO_KEY);

  const provider = getProvider(settings.provider);
  const model = settings.model || provider.defaultModel;

  const messages = Array.isArray(msg.history) ? [...msg.history] : [];
  messages.push({ role: 'user', content: msg.question });

  console.log(`[followup] ${settings.provider}/${model} history:${msg.history?.length || 0}条`);

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

// ---- 缓存 key ----

function cacheKey(market, code, period, bucket, style, mode = 'single', decision = 'off') {
  return `analysis:${market}.${code}:${period}:${bucket}:${style}:${mode}:${decision}`;
}

// ---- 时间 bucket ----

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

// ---- 东财 K 线 ----

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
    throw new Error('网络错误，请检查连接');
  }
  if (!resp.ok) throw new Error(ERR.EASTMONEY);

  const json = await resp.json();
  const data = json?.data;
  if (!data || !Array.isArray(data.klines)) throw new Error(ERR.EASTMONEY);
  return { name: data.name || code, klines: data.klines };
}

// ---- 沪深300 K 线 ----

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
    return { name: data.name || '沪深300', klines: data.klines };
  } catch (_) {
    // 大盘数据获取失败不阻塞主流程
    return null;
  }
}

// ---- 月度用量统计 ----

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

// ---- 多周期共振 ----

async function handleMultiPeriod({ market, code, settings, style, decisionMode, decision, extraContext }) {
  const limit = settings.klineLimit;

  // 三周期并发抓取 + 沪深300 月线
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

  // 各自计算 MA + MACD
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

  // 沪深300 月线数据
  let hs300IndexData = null;
  const hs300Klines = hs300Data ? parseKlines(hs300Data.klines) : [];
  if (hs300Klines.length > 0) {
    const hs300WithMA = attachIndicators(hs300Klines);
    hs300IndexData = { name: hs300Data.name || '沪深300', klines: hs300WithMA };
  }

  // 桶取日线最新日期（最细精度）
  const latestDate = dailyKlines.length > 0 ? dailyKlines[dailyKlines.length - 1].date
    : (weeklyKlines.length > 0 ? weeklyKlines[weeklyKlines.length - 1].date
      : monthlyKlines[monthlyKlines.length - 1].date);
  const bucket = timeBucket(latestDate, 'daily');
  const key = cacheKey(market, code, 'multi', bucket, style, 'single', decision);

  // multi 模式暂不缓存（prompt 体积大，缓存价值低）

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

  console.log(`[analyze] multi/${settings.provider}/${model} 月:${monthlyKlines.length} 周:${weeklyKlines.length} 日:${dailyKlines.length} prompt:${prompt.length}`);

  const result = await provider.call(prompt, {
    model,
    apiKey: settings.apiKey,
    maxTokens: DEFAULT_MAX_TOKENS * 2, // 多周期输出需要更多 token
  });

  const analysis = result.text;
  const usage = result.usage || null;
  const cost = result.usage ? estimateCost(settings.provider, model, result.usage) : 0;

  if (result.usage) {
    await accumulateUsage(settings.provider, model, result.usage, cost);
  }

  // 历史记录 ID
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

  // 写入分析历史
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

// ---- 设置 ----

async function loadSettings() {
  // 先跑迁移（幂等）
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

  // 模型选择优先级：
  // 1. 用户手动填写的 model:${provider}（非空）
  // 2. analysisDepth 映射（仅 Anthropic）：standard → claude-sonnet-4-6, deep → claude-opus-4-7
  // 3. provider 默认模型
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

// ---- 分析历史 ----

async function saveToHistory(entry) {
  const items = await chrome.storage.local.get([HISTORY_KEY]);
  let list = Array.isArray(items[HISTORY_KEY]) ? items[HISTORY_KEY] : [];

  // 去重：同一 code + template + timestamp（秒级）视为重复
  const dupIdx = list.findIndex((e) => e.code === entry.code && e.template === entry.template && Math.abs((e.timestamp || 0) - (entry.timestamp || 0)) < 2000);
  if (dupIdx >= 0) {
    list[dupIdx] = entry;
    await persistHistory(list);
    return;
  }

  list.push(entry);

  // 容量管理
  list = trimHistory(list);
  await persistHistory(list);
}

async function persistHistory(list) {
  // trimHistory 已处理条数上限，这里只做体积检查
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
  if (idx < 0) throw new Error('历史记录不存在');
  list[idx].conversationHistory = conversationHistory;
  // 更新时间戳
  list[idx].timestamp = Date.now();
  await persistHistory(list);
}

async function handleGetHistory() {
  const items = await chrome.storage.local.get([HISTORY_KEY]);
  const list = Array.isArray(items[HISTORY_KEY]) ? items[HISTORY_KEY] : [];
  // 估算占用
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
    // 导出单条
    const entry = list.find((e) => e.id === id);
    if (!entry) throw new Error('记录不存在');
    const md = historyToMarkdown(entry);
    return { markdown: md, filename: `${entry.name || entry.code}_${formatHistoryDate(entry.timestamp)}.md` };
  }
  // 导出全部
  const md = list.map(historyToMarkdown).join('\n\n---\n\n');
  return { markdown: md, filename: `分析历史_${formatHistoryDate(Date.now())}.md`, count: list.length };
}

// ---- Native Messaging 自动同步到 Node 端 ----

const NATIVE_HOST = 'com.eastmoney_ai.sync';

// 不同步的敏感 key 前缀
const SYNC_SKIP_PREFIXES = [
  'apiKey:',
  'model:',
  'migrated:',
];

function shouldSyncKey(key) {
  return !SYNC_SKIP_PREFIXES.some((prefix) => key.startsWith(prefix));
}

// 防抖：同一 key 的变更累积，500ms 后批量发送
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
    // Native host 未安装或启动失败，静默忽略
    // 数据仍在 chrome.storage.local 中，下次成功同步时会补上
  }
}

// 监听 storage 变化（content.js / popup.js 写入时触发自动同步）
chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== 'local') return;
  for (const [key, change] of Object.entries(changes)) {
    if (change.newValue !== undefined && shouldSyncKey(key)) {
      enqueueSync(key, change.newValue);
    }
  }
});

