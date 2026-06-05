// Multi-agent scheduler (with checkpoint resume)
// Each agent result is immediately persisted to chrome.storage.local; if service worker
// is terminated mid-run, completed agent results are auto-reused on retry, saving tokens.

import { bullAgent } from './bull.js';
import { bearAgent } from './bear.js';
import { predictorAgent } from './predictor.js';
import { judgeAgent } from './judge.js';

const CHECKPOINT_VERSION = 1;
const CHECKPOINT_STALE_MS = 60 * 60 * 1000; // 1-hour timeout; discard after expiration to avoid accumulation

function sumCost(partials, judge) {
  let total = 0;
  if (partials.bull) total += partials.bull.cost;
  if (partials.bear) total += partials.bear.cost;
  if (partials.predictor) total += partials.predictor.cost;
  if (judge) total += judge.cost;
  return total;
}

// ---- checkpoint ----

function getIsoWeek(dateStr) {
  // Consistent with background.js:isoWeekFromDate
  const d = new Date(dateStr.replace(/\s.*$/, '') + 'T00:00:00');
  const dayOfWeek = d.getDay() || 7;
  d.setDate(d.getDate() + 4 - dayOfWeek);
  const yearStart = new Date(d.getFullYear(), 0, 1);
  const weekNum = Math.ceil(((d - yearStart) / 86400000 + 1) / 7);
  return `${d.getFullYear()}-W${String(weekNum).padStart(2, '0')}`;
}

function isBarClosed(dateStr, period) {
  // Current bar not closed, intraday close fluctuation invalidates fingerprint -> exclude
  const now = new Date();
  const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  switch (period) {
    case 'monthly':
      return dateStr.slice(0, 7) !== today.slice(0, 7);
    case 'weekly':
      return getIsoWeek(dateStr) !== getIsoWeek(today);
    case 'daily':
    default:
      return dateStr.slice(0, 10) !== today;
  }
}

function buildFingerprint(ctx) {
  // 只用已收盘 bar 做指纹 — 排除当期未收盘 bar
  // 盘中 close 跳动不会使指纹失效；数据源/复权基准变化时 closed bar 的
  // close 会变，指纹能正确丢弃过期 checkpoint
  const closedBars = ctx.klines.filter((k) => isBarClosed(k.date, ctx.period));
  const barCount = closedBars.length;
  if (barCount === 0) return '0';
  const first = closedBars[0];
  const last = closedBars[barCount - 1];
  const input = `${ctx.code}|${ctx.period}|${barCount}|${first.date}|${last.date}|${last.close}`;
  let hash = 5381;
  for (let i = 0; i < input.length; i++) {
    hash = ((hash << 5) + hash + input.charCodeAt(i)) | 0;
  }
  return String(hash >>> 0);
}

async function loadCheckpoint(key, fingerprint) {
  try {
    const items = await chrome.storage.local.get([key]);
    const raw = items[key];
    if (!raw || raw.v !== CHECKPOINT_VERSION) return null;
    if (raw.fp !== fingerprint || (Date.now() - raw.ts > CHECKPOINT_STALE_MS)) {
      // 指纹不匹配 / 超过 1 小时 → 丢弃
      if (raw.fp !== fingerprint) {
        console.warn('[debate-wip] 指纹不匹配，丢弃旧 checkpoint');
      } else {
        console.warn('[debate-wip] checkpoint 超时 (>1h)，丢弃');
      }
      await chrome.storage.local.remove([key]);
      return null;
    }
    return raw;
  } catch (e) { console.warn('[debate-wip] loadCheckpoint 失败:', e.message); return null; }
}

async function mergeCheckpointPartial(key, role, value, fingerprint) {
  try {
    const items = await chrome.storage.local.get([key]);
    const prev = items[key] || { partials: {}, errors: {} };
    await chrome.storage.local.set({
      [key]: {
        v: CHECKPOINT_VERSION,
        ts: Date.now(),
        fp: fingerprint,
        partials: { ...prev.partials, [role]: value },
        errors: { ...prev.errors },
      },
    });
  } catch (e) { console.warn('[debate-wip] mergeCheckpointPartial 失败:', e.message); }
}

async function mergeCheckpointError(key, role, errorMsg, fingerprint) {
  try {
    const items = await chrome.storage.local.get([key]);
    const prev = items[key] || { partials: {}, errors: {} };
    await chrome.storage.local.set({
      [key]: {
        v: CHECKPOINT_VERSION,
        ts: Date.now(),
        fp: fingerprint,
        partials: { ...prev.partials },
        errors: { ...prev.errors, [role]: errorMsg },
      },
    });
  } catch (e) { console.warn('[debate-wip] mergeCheckpointError 失败:', e.message); }
}

export async function clearDebateCheckpoint(key) {
  if (!key) return;
  try { await chrome.storage.local.remove([key]); } catch (e) { console.warn('[debate-wip] clear 失败:', e.message); }
}

/**
 * 运行辩论流程：Bull/Bear/Predictor 并发 → Judge 综合
 * 若 opts.checkpointKey 存在：读 checkpoint 复用已完成的 Agent，每完成一个立即落盘
 */
export async function runDebate(ctx, opts) {
  const startTime = Date.now();
  const checkpointKey = opts.checkpointKey || null;

  // 加载 checkpoint
  let checkpoint = null;
  let fingerprint = null;
  if (checkpointKey) {
    fingerprint = buildFingerprint(ctx);
    checkpoint = await loadCheckpoint(checkpointKey, fingerprint);
  }

  const hasPartial = (role) => checkpoint && checkpoint.partials && checkpoint.partials[role];

  // 第一阶段：三方并发（有 checkpoint 则复用已完成 Agent）
  // 每个 Agent 一旦 fulfilled 立即写入 checkpoint，用串行链避免读-改-写竞态
  const agentDefs = [
    { agent: bullAgent, role: 'bull' },
    { agent: bearAgent, role: 'bear' },
    { agent: predictorAgent, role: 'predictor' },
  ];

  let mergeChain = Promise.resolve();

  const promises = agentDefs.map(({ agent, role }) => {
    if (hasPartial(role)) {
      return Promise.resolve(checkpoint.partials[role]);
    }
    return agent.run(ctx, opts).then(
      (value) => {
        if (checkpointKey) {
          mergeChain = mergeChain.then(() => mergeCheckpointPartial(checkpointKey, role, value, fingerprint));
        }
        return value;
      },
      (err) => {
        if (checkpointKey) {
          mergeChain = mergeChain.then(() => mergeCheckpointError(checkpointKey, role, String(err), fingerprint));
        }
        throw err;
      },
    );
  });

  const [bull, bear, predictor] = await Promise.allSettled(promises);

  const partials = {
    bull:      bull.status === 'fulfilled' ? bull.value : null,
    bear:      bear.status === 'fulfilled' ? bear.value : null,
    predictor: predictor.status === 'fulfilled' ? predictor.value : null,
  };

  const errors = {
    bull:      bull.status === 'rejected' ? String(bull.reason) : null,
    bear:      bear.status === 'rejected' ? String(bear.reason) : null,
    predictor: predictor.status === 'rejected' ? String(predictor.reason) : null,
  };

  // 第二阶段：Judge 综合（至少 2 个 Agent 成功才调）
  let judge = null;
  let judgeError = null;
  const successCount = Object.values(partials).filter((p) => p !== null).length;
  if (successCount >= 2) {
    try {
      judge = await judgeAgent.run({ ...ctx, partials }, opts);
    } catch (err) {
      judgeError = String(err);
    }
  } else {
    judgeError = '可用 Agent 不足 2 个，跳过综合裁判';
  }

  // 等待所有 checkpoint 写入完成
  if (checkpointKey) {
    try { await mergeChain; } catch (e) { console.warn('[debate-wip] mergeChain 未预期错误:', e.message); }
  }

  return {
    partials,
    errors: { ...errors, judge: judgeError },
    judge,
    totalCost: sumCost(partials, judge),
    totalDurationMs: Date.now() - startTime,
  };
}
