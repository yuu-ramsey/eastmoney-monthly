// Multi-period resonance analysis engine
import { calculateAll } from '../indicators/calculate.js';
import { calcMonthlyDirection, calcWeeklyDirection, calcDailyDirection } from './direction.js';

/**
 * @param {string} code — 股票代码
 * @param {string} cutoffDate — eval 场景截止日期
 * @returns {{ monthly:'bull'|'bear'|'neutral', weekly, daily, resonanceLevel:'strong'|'partial'|'divergent', resonanceCount:number, dominant:'bull'|'bear'|'neutral', allBull:boolean, allBear:boolean, description:string }}
 */
export async function getResonanceAsOf(code, cutoffDate) {
  // Dynamic import to avoid service worker loading Node native modules (better-sqlite3)
  const { getKlines } = await import('../db/klines-repo.js');

  const mkt = code.startsWith('6') ? '1' : '0';

  const [mR, wR, dR] = await Promise.all([
    getKlines({ code, market: mkt, period: 'monthly', cutoffDate, limit: 200 }),
    getKlines({ code, market: mkt, period: 'weekly', cutoffDate, limit: 100 }),
    getKlines({ code, market: mkt, period: 'daily', cutoffDate, limit: 60 }),
  ]);

  const mKlines = mR.klines, wKlines = wR.klines, dKlines = dR.klines;
  if (mKlines.length < 12 || wKlines.length < 12 || dKlines.length < 12) {
    return { monthly: 'neutral', weekly: 'neutral', daily: 'neutral', resonanceLevel: 'divergent', resonanceCount: 0, dominant: 'neutral', allBull: false, allBear: false, description: '数据不足' };
  }

  const mInd = calculateAll(mKlines);
  const wInd = calculateAll(wKlines);
  const dInd = calculateAll(dKlines);

  const monthly = calcMonthlyDirection(mKlines, mInd);
  const weekly = calcWeeklyDirection(wKlines, wInd);
  const daily = calcDailyDirection(dKlines, dInd);

  return calcResonanceFromDirections(monthly, weekly, daily);
}

/** 直接从已知方向计算共振（用于批量/缓存场景） */
export function calcResonanceFromDirections(monthly, weekly, daily) {
  const dirs = [monthly, weekly, daily];
  const counts = { bull: 0, bear: 0, neutral: 0 };
  dirs.forEach(d => { counts[d]++; });

  let dominant = 'neutral';
  if (counts.bull > counts.bear && counts.bull > counts.neutral) dominant = 'bull';
  else if (counts.bear > counts.bull && counts.bear > counts.neutral) dominant = 'bear';

  let resonanceLevel = 'divergent';
  if (counts[dominant] === 3) resonanceLevel = 'strong';
  else if (counts[dominant] === 2) resonanceLevel = 'partial';

  const resonanceCount = counts[dominant];
  const allBull = monthly === 'bull' && weekly === 'bull' && daily === 'bull';
  const allBear = monthly === 'bear' && weekly === 'bear' && daily === 'bear';

  const periodLabels = { monthly: '月线', weekly: '周线', daily: '日线' };
  const dirLabels = { bull: '偏多', bear: '偏空', neutral: '中性' };
  const desc = `月线:${dirLabels[monthly]} 周线:${dirLabels[weekly]} 日线:${dirLabels[daily]} | 共振:${resonanceLevel}(${resonanceCount}/3 ${dirLabels[dominant]})`;

  return { monthly, weekly, daily, resonanceLevel, resonanceCount, dominant, allBull, allBear, description: desc };
}

/** 生成 prompt 注入用文本 */
export function formatResonanceSummary(resonance) {
  if (!resonance) return '';
  const dirLabels = { bull: '偏多 ✅', bear: '偏空 ❌', neutral: '中性 ⚠' };
  const lines = ['\n## 多周期共振分析（程序完成）'];
  lines.push('');
  lines.push(`| 周期 | 方向 |`);
  lines.push(`|------|------|`);
  lines.push(`| 月线 | ${dirLabels[resonance.monthly]} |`);
  lines.push(`| 周线 | ${dirLabels[resonance.weekly]} |`);
  lines.push(`| 日线 | ${dirLabels[resonance.daily]} |`);
  lines.push('');
  const resonanceLabels = { strong: '🟢 强共振', partial: '🟡 中等共振', divergent: '🔴 无共振' };
  lines.push(`**共振等级**: ${resonanceLabels[resonance.resonanceLevel]} (${resonance.resonanceCount}/3 同向 ${resonance.dominant === 'bull' ? '偏多' : resonance.dominant === 'bear' ? '偏空' : '中性'})`);
  return lines.join('\n') + '\n';
}

/** 生成 HARD_CONSTRAINTS #11 指引 — 反向版本 */
export function buildResonanceConstraint(resonance, signalSummary) {
  if (!resonance) return '';

  if (resonance.resonanceLevel === 'divergent') {
    return '\n⚠ 共振：divergent 无共振（月/周/日方向不一致），按技术面自由判断。';
  }

  if (resonance.allBull) {
    return `\n⚠ 共振反向约束：**strong_bull 共振**（三周期全偏多）= 趋势末端警告。禁止判 strong_bull，最多判 bull；倾向判 neutral。历史数据：全偏多后 6 月 alpha 仅 4.31%，跑输全偏空后 7.67%。`;
  }
  if (resonance.allBear) {
    return `\n⚠ 共振反向约束：**strong_bear 共振**（三周期全偏空）= 均值回归机会。可判 bull 或 strong_bull（若技术面支持）；不要判 bear/strong_bear。历史数据：全偏空后 6 月 alpha 7.67%，跑赢全偏多后 4.31%。`;
  }
  if (resonance.dominant === 'bull') {
    return '\n⚠ 共振：partial 共振偏多，方向参考但禁止判 strong_bull。';
  }
  if (resonance.dominant === 'bear') {
    return '\n⚠ 共振反向约束：partial 共振偏空 = 弱均值回归信号，可倾向 bull，不要判 bear。';
  }
  return '\n⚠ 共振：方向不明确，按技术面自由判断。';
}
