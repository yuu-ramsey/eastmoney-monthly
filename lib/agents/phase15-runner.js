// Phase 15 Multi-Agent 辩论调度
// Bull+Bear parallel → Technical+Sector parallel → Judge
import { bullResearcher } from './bull-researcher.js';
import { bearResearcher } from './bear-researcher.js';
import { technicalAgent } from './technical-agent.js';
import { sectorAgent } from './sector-agent.js';
import { judgeAgent } from './judge-agent.js';

export async function runMultiAgentDebate(ctx, opts) {
  const startTime = Date.now();
  const partials = {};
  const errors = {};

  // Phase 1: Bull + Bear 并行
  const [bullR, bearR] = await Promise.allSettled([
    bullResearcher.run(ctx, opts),
    bearResearcher.run(ctx, opts),
  ]);
  partials.bull_researcher = bullR.status === 'fulfilled' ? bullR.value : null;
  partials.bear_researcher = bearR.status === 'fulfilled' ? bearR.value : null;
  if (bullR.status === 'rejected') errors.bull_researcher = String(bullR.reason);
  if (bearR.status === 'rejected') errors.bear_researcher = String(bearR.reason);

  // Phase 2: Technical + Sector 并行
  const [techR, secR] = await Promise.allSettled([
    technicalAgent.run(ctx, opts),
    sectorAgent.run(ctx, opts),
  ]);
  partials.technical_agent = techR.status === 'fulfilled' ? techR.value : null;
  partials.sector_agent = secR.status === 'fulfilled' ? secR.value : null;
  if (techR.status === 'rejected') errors.technical_agent = String(techR.reason);
  if (secR.status === 'rejected') errors.sector_agent = String(secR.reason);

  // Phase 3: Judge — 至少 3/4 成功
  const successCount = Object.values(partials).filter(p => p !== null).length;
  let judge = null;
  let judgeError = null;
  if (successCount >= 3) {
    try {
      judge = await judgeAgent.run({ ...ctx, partials }, opts);
    } catch (err) {
      judgeError = String(err);
    }
  } else {
    judgeError = `不足 3 个 Agent 成功 (${successCount}/4)，跳过 Judge`;
  }

  // 总成本
  let totalCost = 0;
  for (const p of Object.values(partials)) { if (p) totalCost += p.cost || 0; }
  if (judge) totalCost += judge.cost || 0;

  return {
    partials,
    errors,
    judge,
    judgeError,
    totalCost,
    successCount,
    durationMs: Date.now() - startTime,
  };
}

/** 从 Judge 输出解析 signal */
export function parseJudgeSignal(judgeText) {
  if (!judgeText) return null;
  try {
    const m = judgeText.match(/```json\s*([\s\S]*?)```/);
    if (m) {
      const data = JSON.parse(m[1].trim());
      return {
        signal: data.signal || 'neutral',
        confidence: data.confidence || 'medium',
        summary: data.one_line_summary || '',
      };
    }
  } catch (_) {}
  return null;
}
