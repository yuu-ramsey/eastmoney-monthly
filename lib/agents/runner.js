// 多 Agent 调度器（带 checkpoint 续跑）
// 每个 Agent 跑完立即落盘到 chrome.storage.local，service worker 中途
// 被终止后，用户重试时自动复用已完成的 Agent 结果，不重复消耗 token。

import { bullAgent } from './bull.js';
import { bearAgent } from './bear.js';
import { predictorAgent } from './predictor.js';
import { judgeAgent } from './judge.js';

const CHECKPOINT_VERSION = 1;

function sumCost(partials, judge) {
  let total = 0;
  if (partials.bull) total += partials.bull.cost;
  if (partials.bear) total += partials.bear.cost;
  if (partials.predictor) total += partials.predictor.cost;
  if (judge) total += judge.cost;
  return total;
}

// ---- checkpoint ----

function buildFingerprint(ctx) {
  // 用 code + period + klines 尾部数据做输入指纹
  // 若 K 线数据或 cutoff 变化，指纹不匹配 → 丢弃整个 checkpoint 重跑
  const tail = ctx.klines.slice(-5).map((k) => `${k.date}:${k.close}`).join(',');
  const input = `${ctx.code}|${ctx.period}|${tail}`;
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
    if (raw.fp !== fingerprint) {
      // 指纹不匹配 → 输入数据已变，丢弃旧 checkpoint
      await chrome.storage.local.remove([key]);
      return null;
    }
    return raw;
  } catch (_) { return null; }
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
  } catch (_) { /* storage 不可用时静默降级，不影响分析 */ }
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
  } catch (_) { /* 静默降级 */ }
}

export async function clearDebateCheckpoint(key) {
  if (!key) return;
  try { await chrome.storage.local.remove([key]); } catch (_) { /* ignore */ }
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
    try { await mergeChain; } catch (_) { /* merge chain 内部已吞错误 */ }
  }

  return {
    partials,
    errors: { ...errors, judge: judgeError },
    judge,
    totalCost: sumCost(partials, judge),
    totalDurationMs: Date.now() - startTime,
  };
}
