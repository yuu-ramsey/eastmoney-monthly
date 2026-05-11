// 多 Agent 调度器

import { bullAgent } from './bull.js';
import { bearAgent } from './bear.js';
import { predictorAgent } from './predictor.js';
import { judgeAgent } from './judge.js';

function sumCost(partials, judge) {
  let total = 0;
  if (partials.bull) total += partials.bull.cost;
  if (partials.bear) total += partials.bear.cost;
  if (partials.predictor) total += partials.predictor.cost;
  if (judge) total += judge.cost;
  return total;
}

/**
 * 运行辩论流程：Bull/Bear/Predictor 并发 → Judge 综合
 */
export async function runDebate(ctx, opts) {
  const startTime = Date.now();

  // 第一阶段：三方并发
  const [bull, bear, predictor] = await Promise.allSettled([
    bullAgent.run(ctx, opts),
    bearAgent.run(ctx, opts),
    predictorAgent.run(ctx, opts),
  ]);

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

  return {
    partials,
    errors: { ...errors, judge: judgeError },
    judge,
    totalCost: sumCost(partials, judge),
    totalDurationMs: Date.now() - startTime,
  };
}
