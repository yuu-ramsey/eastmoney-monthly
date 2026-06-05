// Output calibrator — based on Batch Calibration approach
// arxiv 2310.17256, 2402.10353
// Correct LLM output bias using real distribution from historical eval data
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const CALIB_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval', 'calibrators');

function ensureDir() { if (!fs.existsSync(CALIB_DIR)) fs.mkdirSync(CALIB_DIR, { recursive: true }); }

/**
 * Build calibration mapping table
 * @param {Array} evaluations - Results array from runner
 * @returns {object} 校准表
 */
export function buildCalibrator(evaluations) {
  const valid = evaluations.filter(r => !r.error && r.predictedSignal && r.groundTruth && r.score != null);
  if (valid.length === 0) return null;

  // 按 predictedSignal 分组统计实际 groundTruth 分布
  const groups = {};

  for (const r of valid) {
    const key = r.predictedSignal;
    if (!groups[key]) groups[key] = { total: 0, counts: {} };
    groups[key].total++;
    groups[key].counts[r.groundTruth] = (groups[key].counts[r.groundTruth] || 0) + 1;
  }

  // 计算每个分组的分布
  const table = {};
  for (const [key, g] of Object.entries(groups)) {
    const dist = {};
    for (const [gt, count] of Object.entries(g.counts)) {
      dist[gt] = +(count / g.total).toFixed(3);
    }
    table[key] = { total: g.total, distribution: dist };
  }

  return {
    version: '1.0',
    builtAt: new Date().toISOString(),
    totalSamples: valid.length,
    table,
  };
}

/**
 * 校准单个LLM输出
 * @param {object} rawOutput — { signal, score, ... }
 * @param {object} calibrator — buildCalibrator返回
 * @returns {object} — { signal: 校准后signal, rawSignal: 原始, confidence: 'calibrated' }
 */
export function calibrateOutput(rawOutput, calibrator) {
  if (!calibrator || !rawOutput) return { ...rawOutput, calibrated: false };

  const key = rawOutput.signal;
  const entry = calibrator.table[key];

  if (!entry || entry.total < 5) {
    return { ...rawOutput, calibrated: false };
  }

  const dist = entry.distribution;
  // 取分布最大值作为校准后信号
  let bestGT = rawOutput.signal;
  let bestProb = 0;
  for (const [gt, prob] of Object.entries(dist)) {
    if (prob > bestProb) { bestProb = prob; bestGT = gt; }
  }

  // 如果最可能的 groundTruth 占比不足 30%，不翻转（不确定）
  if (bestProb < 0.30) {
    return { ...rawOutput, calibrated: false };
  }

  return {
    ...rawOutput,
    signal: bestGT,
    rawSignal: rawOutput.signal,
    calibrationConfidence: bestProb,
    calibrated: true,
  };
}

/**
 * 保存校准器
 */
export function saveCalibrator(calibrator, label = 'active') {
  ensureDir();
  const p = path.join(CALIB_DIR, `${label}.json`);
  fs.writeFileSync(p, JSON.stringify(calibrator, null, 2), 'utf-8');
  return p;
}

/**
 * 加载校准器
 */
export function loadCalibrator(label = 'active') {
  const p = path.join(CALIB_DIR, `${label}.json`);
  try { return JSON.parse(fs.readFileSync(p, 'utf-8')); } catch (_) { return null; }
}

/**
 * 对eval结果批量校准，返回新的结果数组
 */
export function calibrateEvalResults(results, calibrator) {
  return results.map(r => {
    if (r.error) return r;
    const calibrated = calibrateOutput({ signal: r.predictedSignal, score: r.score }, calibrator);
    const newScore = calibrated.calibrated
      ? scorePredictionStatic(calibrated.signal, r.groundTruth)
      : r.score;
    return {
      ...r,
      originalPredictedSignal: r.predictedSignal,
      predictedSignal: calibrated.signal,
      calibrated: calibrated.calibrated,
      score: newScore,
    };
  });
}

// 避免循环依赖，内联scorePrediction
const SIGNAL_MAP = { strong_bull: 2, bull: 1, neutral: 0, bear: -1, strong_bear: -2 };
function scorePredictionStatic(predicted, groundTruth) {
  const p = SIGNAL_MAP[predicted] ?? null;
  const g = SIGNAL_MAP[groundTruth] ?? null;
  if (p === null || g === null) return 0;
  if (p === g) return 1.0;
  if (p * g < 0) return (Math.abs(p) === 2 && Math.abs(g) === 2) ? -1.0 : -0.5;
  if (p === 0 || g === 0) return 0.3;
  return 0.5;
}
