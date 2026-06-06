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
  if (!fs.existsSync(runPath)) throw new Error(`Run not found: ${runId}`);

  const lines = fs.readFileSync(runPath, 'utf-8').trim().split('\n').filter(Boolean);
  const results = lines.map((l) => JSON.parse(l));
  const successful = results.filter((r) => !r.error);
  const failed = results.filter((r) => r.error);

  const totalCost = successful.reduce((s, r) => s + (r.cost || 0), 0);
  const promptVersion = successful[0]?.promptVersion || '?';
  const provider = successful[0]?.model || '?';

  // Overall stats
  const overall = computeStats(successful);

  // By template
  const byTemplate = groupBy(successful, 'template');
  const templateStats = {};
  for (const [tpl, items] of Object.entries(byTemplate)) {
    templateStats[tpl] = computeStats(items);
  }

  // By category
  const byCategory = groupBy(successful, 'category');
  const categoryStats = {};
  for (const [cat, items] of Object.entries(byCategory)) {
    categoryStats[cat] = computeStats(items);
  }

  // Top 10 mistakes
  const worst = successful.filter((r) => r.score != null && r.score < 0).sort((a, b) => (a.score || 0) - (b.score || 0)).slice(0, 10);

  // Generate markdown
  let md = `# Eval Report ${runId}\n\n`;
  md += `## Basic Info\n\n`;
  md += `- promptVersion: ${promptVersion}\n`;
  md += `- provider/model: ${provider}\n`;
  md += `- Total samples: ${successful.length} (success) / ${failed.length} (failed)\n`;
  md += `- Total cost: ¥${totalCost.toFixed(4)}\n\n`;

  md += `## Overall Accuracy\n\n`;
  md += formatOverall(overall);

  md += `\n## Performance by Template\n\n`;
  md += `| Template | Perfect | Direction Correct | Direction Wrong | Weighted Score | Samples |\n`;
  md += `|------|---------|---------|---------|---------|--------|\n`;
  for (const [tpl, s] of Object.entries(templateStats)) {
    md += `| ${tpl} | ${pct(s.perfect, s.total)} | ${pct(s.directionCorrect, s.total)} | ${pct(s.directionWrong, s.total)} | ${fmt(s.weightedScore)} | ${s.total} |\n`;
  }

  // Prediction distribution
  md += `\n## Prediction Distribution\n\n`;
  md += formatDistribution(successful);

  md += `\n## Performance by Stock Category\n\n`;
  md += `| Category | Perfect | Weighted Score | Samples |\n`;
  md += `|------|---------|---------|--------|\n`;
  for (const [cat, s] of Object.entries(categoryStats)) {
    md += `| ${cat} | ${pct(s.perfect, s.total)} | ${fmt(s.weightedScore)} | ${s.total} |\n`;
  }

  if (worst.length > 0) {
    md += `\n## Top ${worst.length} Mistakes\n\n`;
    md += `| stockCode | cutoffDate | template | predicted | actual | alpha | score |\n`;
    md += `|-----------|------------|----------|-----------|--------|-------|-------|\n`;
    for (const r of worst) {
      md += `| ${r.stockCode} | ${r.cutoffDate} | ${r.template} | ${r.predictedSignal} | ${r.groundTruth} | ${fmt(r.alpha)}% | ${fmt(r.score)} |\n`;
    }
  }

  // Write file
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
  return `- Perfect: ${pct(s.perfect, s.total)}\n- Direction Correct: ${pct(s.directionCorrect, s.total)}\n- Direction Wrong: ${pct(s.directionWrong, s.total)}\n- Weighted Score: ${fmt(s.weightedScore)}\n`;
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
 * Compare two runs
 */
export function compareRuns(runIdA, runIdB) {
  const pathA = path.join(EVAL_DIR, 'runs', `${runIdA}.jsonl`);
  const pathB = path.join(EVAL_DIR, 'runs', `${runIdB}.jsonl`);
  if (!fs.existsSync(pathA)) throw new Error(`Run not found: ${runIdA}`);
  if (!fs.existsSync(pathB)) throw new Error(`Run not found: ${runIdB}`);

  const readResults = (p) => fs.readFileSync(p, 'utf-8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l)).filter((r) => !r.error);
  const resultsA = readResults(pathA);
  const resultsB = readResults(pathB);

  const statsA = computeStats(resultsA);
  const statsB = computeStats(resultsB);

  // Compare by template
  const byTemplateA = groupBy(resultsA, 'template');
  const byTemplateB = groupBy(resultsB, 'template');

  let md = `# ${runIdA} vs ${runIdB} Comparison\n\n`;
  md += `## Overall\n`;
  md += `- Perfect rate: ${pct(statsA.perfect, statsA.total)} → ${pct(statsB.perfect, statsB.total)} (${delta(statsB.perfect - statsA.perfect)})\n`;
  md += `- Weighted score: ${fmt(statsA.weightedScore)} → ${fmt(statsB.weightedScore)} (${delta(statsB.weightedScore - statsA.weightedScore)})\n\n`;

  md += `## By Template\n`;
  md += `| Template | ${runIdA} | ${runIdB} | Δ |\n`;
  md += `|------|-----|-----|---|\n`;
  for (const tpl of Object.keys({ ...byTemplateA, ...byTemplateB })) {
    const sa = computeStats(byTemplateA[tpl] || []);
    const sb = computeStats(byTemplateB[tpl] || []);
    md += `| ${tpl} | ${fmt(sa.weightedScore)} | ${fmt(sb.weightedScore)} | ${delta(sb.weightedScore - sa.weightedScore)} |\n`;
  }

  // Fixed and new mistakes
  const keyFn = (r) => `${r.testPointId}_${r.template}`;
  const mapA = new Map(resultsA.map((r) => [keyFn(r), r]));
  const mapB = new Map(resultsB.map((r) => [keyFn(r), r]));

  const fixed = []; // A wrong, B correct
  const broken = []; // A correct, B wrong
  for (const [key, ra] of mapA) {
    const rb = mapB.get(key);
    if (rb && ra.score != null && rb.score != null) {
      if (ra.score < 0 && rb.score > 0.5) fixed.push({ ...ra, scoreB: rb.score });
      if (ra.score > 0.5 && rb.score < 0) broken.push({ ...rb, scoreA: ra.score });
    }
  }

  if (fixed.length > 0) {
    md += `\n## Fixed Mistakes (${runIdA} wrong → ${runIdB} correct)\n`;
    md += fixed.slice(0, 10).map((r) => `- ${r.stockCode} ${r.cutoffDate} ${r.template}: ${fmt(r.score)}→${fmt(r.scoreB)}`).join('\n') + '\n';
  }
  if (broken.length > 0) {
    md += `\n## New Mistakes (${runIdA} correct → ${runIdB} wrong) ⚠️\n`;
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

  let md = `| Signal | Predicted % | groundTruth % | Diff |\n`;
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
    md += `\n⚠️ Prediction distribution deviates >10% from groundTruth. Model may have directional bias.\n`;
    if (bearBias > 10) md += `⚠️ Model leans bearish (bear bias +${bearBias.toFixed(1)}%). Monitor closely.\n`;
  }
  return md;
}

function delta(v) { return v >= 0 ? `+${fmt(v)} ↑` : `${fmt(v)} ↓`; }
