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
  // Only use closed bars for fingerprint — exclude current period's open bar
  // Intraday close fluctuation won't invalidate fingerprint; when data source /
  // adjustment basis changes, closed bar close will change, fingerprint correctly discards stale checkpoint
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
      // Fingerprint mismatch / >1 hour stale → discard
      if (raw.fp !== fingerprint) {
        console.warn('[debate-wip] Fingerprint mismatch, discarding old checkpoint');
      } else {
        console.warn('[debate-wip] Checkpoint expired (>1h), discarding');
      }
      await chrome.storage.local.remove([key]);
      return null;
    }
    return raw;
  } catch (e) { console.warn('[debate-wip] loadCheckpoint failed:', e.message); return null; }
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
  } catch (e) { console.warn('[debate-wip] mergeCheckpointPartial failed:', e.message); }
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
  } catch (e) { console.warn('[debate-wip] mergeCheckpointError failed:', e.message); }
}

export async function clearDebateCheckpoint(key) {
  if (!key) return;
  try { await chrome.storage.local.remove([key]); } catch (e) { console.warn('[debate-wip] clear failed:', e.message); }
}

/**
 * Run debate workflow: Bull/Bear/Predictor concurrent → Judge synthesis
 * If opts.checkpointKey exists: read checkpoint to reuse completed Agents, persist each immediately on completion
 */
export async function runDebate(ctx, opts) {
  const startTime = Date.now();
  const checkpointKey = opts.checkpointKey || null;

  // Load checkpoint
  let checkpoint = null;
  let fingerprint = null;
  if (checkpointKey) {
    fingerprint = buildFingerprint(ctx);
    checkpoint = await loadCheckpoint(checkpointKey, fingerprint);
  }

  const hasPartial = (role) => checkpoint && checkpoint.partials && checkpoint.partials[role];

  // Phase 1: Three-way concurrent (reuse completed Agents if checkpoint exists)
  // Each Agent writes to checkpoint immediately on fulfillment, using serial chain to avoid read-modify-write races
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

  // Phase 2: Judge synthesis (call only if at least 2 Agents succeeded)
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

  // Wait for all checkpoint writes to complete
  if (checkpointKey) {
    try { await mergeChain; } catch (e) { console.warn('[debate-wip] mergeChain unexpected error:', e.message); }
  }

  return {
    partials,
    errors: { ...errors, judge: judgeError },
    judge,
    totalCost: sumCost(partials, judge),
    totalDurationMs: Date.now() - startTime,
  };
}
