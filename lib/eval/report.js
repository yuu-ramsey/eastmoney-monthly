// Evaluation report generation
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const EVAL_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval');

/**
 * @param {string} runId - e.g. "current-2026-05-15-03-00-00"
 * @returns {string} Report path
 */
export function generateEvalReport(runId) {
  const runPath = path.join(EVAL_DIR, 'runs', `${runId}.jsonl`);
  if (!fs.existsSync(runPath)) throw new Error(`Run 不存在: ${runId}`);

  const lines = fs.readFileSync(runPath, 'utf-8').trim().split('\n').filter(Boolean);
  const results = lines.map((l) => JSON.parse(l));
  const successful = results.filter((r) => !r.error);
  const failed = results.filter((r) => r.error);

  const totalCost = successful.reduce((s, r) => s + (r.cost || 0), 0);
  const promptVersion = successful[0]?.promptVersion || '?';
  const provider = successful[0]?.model || '?';

  // 整体统计
  const overall = computeStats(successful);

  // 分模板
  const byTemplate = groupBy(successful, 'template');
  const templateStats = {};
  for (const [tpl, items] of Object.entries(byTemplate)) {
    templateStats[tpl] = computeStats(items);
  }

  // 分类别
  const byCategory = groupBy(successful, 'category');
  const categoryStats = {};
  for (const [cat, items] of Object.entries(byCategory)) {
    categoryStats[cat] = computeStats(items);
  }

  // 失误 top 10
  const worst = successful.filter((r) => r.score != null && r.score < 0).sort((a, b) => (a.score || 0) - (b.score || 0)).slice(0, 10);

  // 生成 markdown
  let md = `# Eval 报告 ${runId}\n\n`;
  md += `## 基本信息\n\n`;
  md += `- promptVersion: ${promptVersion}\n`;
  md += `- provider/model: ${provider}\n`;
  md += `- 总样本: ${successful.length} (成功) / ${failed.length} (失败)\n`;
  md += `- 总成本: ¥${totalCost.toFixed(4)}\n\n`;

  md += `## 总体准确率\n\n`;
  md += formatOverall(overall);

  md += `\n## 分模板表现\n\n`;
  md += `| 模板 | 完全正确 | 方向正确 | 方向错误 | 加权得分 | 样本数 |\n`;
  md += `|------|---------|---------|---------|---------|--------|\n`;
  for (const [tpl, s] of Object.entries(templateStats)) {
    md += `| ${tpl} | ${pct(s.perfect, s.total)} | ${pct(s.directionCorrect, s.total)} | ${pct(s.directionWrong, s.total)} | ${fmt(s.weightedScore)} | ${s.total} |\n`;
  }

  // 预测分布
  md += `\n## Prediction Distribution（预测分布）\n\n`;
  md += formatDistribution(successful);

  md += `\n## 分股票类别表现\n\n`;
  md += `| 类别 | 完全正确 | 加权得分 | 样本数 |\n`;
  md += `|------|---------|---------|--------|\n`;
  for (const [cat, s] of Object.entries(categoryStats)) {
    md += `| ${cat} | ${pct(s.perfect, s.total)} | ${fmt(s.weightedScore)} | ${s.total} |\n`;
  }

  if (worst.length > 0) {
    md += `\n## 失误案例 top ${worst.length}\n\n`;
    md += `| stockCode | cutoffDate | template | predicted | actual | alpha | score |\n`;
    md += `|-----------|------------|----------|-----------|--------|-------|-------|\n`;
    for (const r of worst) {
      md += `| ${r.stockCode} | ${r.cutoffDate} | ${r.template} | ${r.predictedSignal} | ${r.groundTruth} | ${fmt(r.alpha)}% | ${fmt(r.score)} |\n`;
    }
  }

  // 写文件
  const reportDir = path.join(EVAL_DIR, 'reports');
  if (!fs.existsSync(reportDir)) fs.mkdirSync(reportDir, { recursive: true });
  const reportPath = path.join(reportDir, `${runId}.md`);
  fs.writeFileSync(reportPath, md, 'utf-8');
  return reportPath;
}

function computeStats(items) {
  const total = items.length;
  if (total === 0) return { total: 0, perfect: 0, directionCorrect: 0, directionWrong: 0, weightedScore: 0 };
  let perfect = 0, dirCorrect = 0, dirWrong = 0, scoreSum = 0;
  for (const r of items) {
    if (r.score === 1.0) perfect++;
    if (r.score != null && r.score >= 0.5) dirCorrect++;
    if (r.score != null && r.score < 0) dirWrong++;
    if (r.score != null) scoreSum += r.score;
  }
  return { total, perfect, directionCorrect: dirCorrect, directionWrong: dirWrong, weightedScore: +(scoreSum / total).toFixed(3) };
}

function formatOverall(s) {
  return `- 完全正确: ${pct(s.perfect, s.total)}\n- 方向正确: ${pct(s.directionCorrect, s.total)}\n- 方向错误: ${pct(s.directionWrong, s.total)}\n- 加权得分: ${fmt(s.weightedScore)}\n`;
}

function groupBy(arr, key) {
  const map = {};
  for (const item of arr) {
    const k = item[key] || 'unknown';
    if (!map[k]) map[k] = [];
    map[k].push(item);
  }
  return map;
}

function pct(n, total) { return total > 0 ? ((n / total) * 100).toFixed(0) + '%' : '0%'; }
function fmt(v) { return v != null ? (typeof v === 'number' ? v.toFixed(2) : String(v)) : '-'; }

/**
 * 对比两个 run
 */
export function compareRuns(runIdA, runIdB) {
  const pathA = path.join(EVAL_DIR, 'runs', `${runIdA}.jsonl`);
  const pathB = path.join(EVAL_DIR, 'runs', `${runIdB}.jsonl`);
  if (!fs.existsSync(pathA)) throw new Error(`Run 不存在: ${runIdA}`);
  if (!fs.existsSync(pathB)) throw new Error(`Run 不存在: ${runIdB}`);

  const readResults = (p) => fs.readFileSync(p, 'utf-8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l)).filter((r) => !r.error);
  const resultsA = readResults(pathA);
  const resultsB = readResults(pathB);

  const statsA = computeStats(resultsA);
  const statsB = computeStats(resultsB);

  // 分模板对比
  const byTemplateA = groupBy(resultsA, 'template');
  const byTemplateB = groupBy(resultsB, 'template');

  let md = `# ${runIdA} vs ${runIdB} 对比\n\n`;
  md += `## 整体\n`;
  md += `- 完全正确率: ${pct(statsA.perfect, statsA.total)} → ${pct(statsB.perfect, statsB.total)} (${delta(statsB.perfect - statsA.perfect)})\n`;
  md += `- 加权得分: ${fmt(statsA.weightedScore)} → ${fmt(statsB.weightedScore)} (${delta(statsB.weightedScore - statsA.weightedScore)})\n\n`;

  md += `## 各模板\n`;
  md += `| 模板 | ${runIdA} | ${runIdB} | Δ |\n`;
  md += `|------|-----|-----|---|\n`;
  for (const tpl of Object.keys({ ...byTemplateA, ...byTemplateB })) {
    const sa = computeStats(byTemplateA[tpl] || []);
    const sb = computeStats(byTemplateB[tpl] || []);
    md += `| ${tpl} | ${fmt(sa.weightedScore)} | ${fmt(sb.weightedScore)} | ${delta(sb.weightedScore - sa.weightedScore)} |\n`;
  }

  // 修复和新增的失误
  const keyFn = (r) => `${r.testPointId}_${r.template}`;
  const mapA = new Map(resultsA.map((r) => [keyFn(r), r]));
  const mapB = new Map(resultsB.map((r) => [keyFn(r), r]));

  const fixed = []; // A错B对
  const broken = []; // A对B错
  for (const [key, ra] of mapA) {
    const rb = mapB.get(key);
    if (rb && ra.score != null && rb.score != null) {
      if (ra.score < 0 && rb.score > 0.5) fixed.push({ ...ra, scoreB: rb.score });
      if (ra.score > 0.5 && rb.score < 0) broken.push({ ...rb, scoreA: ra.score });
    }
  }

  if (fixed.length > 0) {
    md += `\n## 修复的失误 (${runIdA}错→${runIdB}对)\n`;
    md += fixed.slice(0, 10).map((r) => `- ${r.stockCode} ${r.cutoffDate} ${r.template}: ${fmt(r.score)}→${fmt(r.scoreB)}`).join('\n') + '\n';
  }
  if (broken.length > 0) {
    md += `\n## 新增的失误 (${runIdA}对→${runIdB}错) ⚠️\n`;
    md += broken.slice(0, 10).map((r) => `- ${r.stockCode} ${r.cutoffDate} ${r.template}: ${fmt(r.scoreA)}→${fmt(r.score)}`).join('\n') + '\n';
  }

  const reportDir = path.join(EVAL_DIR, 'reports');
  if (!fs.existsSync(reportDir)) fs.mkdirSync(reportDir, { recursive: true });
  const reportPath = path.join(reportDir, `compare-${runIdA}-vs-${runIdB}.md`);
  fs.writeFileSync(reportPath, md, 'utf-8');
  return reportPath;
}

function formatDistribution(items) {
  const signals = ['strong_bull', 'bull', 'neutral', 'bear', 'strong_bear'];
  const predCount = {};
  const gtCount = {};
  const total = items.length;
  for (const s of signals) { predCount[s] = 0; gtCount[s] = 0; }
  for (const r of items) {
    if (r.predictedSignal) predCount[r.predictedSignal] = (predCount[r.predictedSignal] || 0) + 1;
    if (r.groundTruth) gtCount[r.groundTruth] = (gtCount[r.groundTruth] || 0) + 1;
  }

  let md = `| signal | 预测占比 | groundTruth占比 | 差异 |\n`;
  md += `|--------|----------|-----------------|------|\n`;
  let biasWarning = false;
  let bearBias = 0;
  for (const s of signals) {
    const pp = total > 0 ? (predCount[s] / total * 100).toFixed(1) : '0';
    const gp = total > 0 ? (gtCount[s] / total * 100).toFixed(1) : '0';
    const diff = +pp - +gp;
    const flag = Math.abs(diff) > 10 ? ' ⚠️' : '';
    if (Math.abs(diff) > 10) biasWarning = true;
    if ((s === 'bear' || s === 'strong_bear') && diff > 0) bearBias += diff;
    md += `| ${s} | ${pp}% | ${gp}% | ${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%${flag} |\n`;
  }

  if (biasWarning) {
    md += `\n⚠️ 预测分布与groundTruth偏差>10%，模型可能存在方向偏差。\n`;
    if (bearBias > 10) md += `⚠️ 模型偏向看空(bear bias +${bearBias.toFixed(1)}%)，建议关注。\n`;
  }
  return md;
}

function delta(v) { return v >= 0 ? `+${fmt(v)} ↑` : `${fmt(v)} ↓`; }
