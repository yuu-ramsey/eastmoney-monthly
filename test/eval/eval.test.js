// eval 测试 — dataset builder / runner score / report / cache
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

// ---- KlinesCache ----
test('KlinesCache: 基本存取', async () => {
  const { KlinesCache } = await import('../../lib/eval/klines-cache.js');
  const c = new KlinesCache();
  assert.equal(c.get('600519', '1'), null);
  c.set('600519', '1', 'monthly', 200, [{ date: '2024-01', close: 10 }]);
  const v = c.get('600519', '1');
  assert.ok(v);
  assert.equal(v.length, 1);
  assert.equal(c.size(), 1);
});

test('KlinesCache: 命中率统计', async () => {
  const { KlinesCache } = await import('../../lib/eval/klines-cache.js');
  const c = new KlinesCache();
  c.get('a', '1'); // miss
  c.get('a', '1'); // miss (not set yet)
  c.set('a', '1', 'monthly', 200, []);
  c.get('a', '1'); // hit
  c.get('a', '1'); // hit
  assert.ok(c.hitRate().startsWith('50')); // 2 hits / 4 total = 50%
});

// ---- scorePrediction ----

test('scorePrediction: 完全匹配=1.0', async () => {
  const { scorePrediction } = await import('../../lib/eval/runner.js');
  assert.equal(scorePrediction('strong_bull', 'strong_bull'), 1.0);
  assert.equal(scorePrediction('bear', 'bear'), 1.0);
  assert.equal(scorePrediction('neutral', 'neutral'), 1.0);
});

test('scorePrediction: 方向对强度错=0.5', async () => {
  const { scorePrediction } = await import('../../lib/eval/runner.js');
  assert.equal(scorePrediction('strong_bull', 'bull'), 0.5);
  assert.equal(scorePrediction('bull', 'strong_bull'), 0.5);
  assert.equal(scorePrediction('bear', 'strong_bear'), 0.5);
});

test('scorePrediction: neutral任意=0.3', async () => {
  const { scorePrediction } = await import('../../lib/eval/runner.js');
  assert.equal(scorePrediction('neutral', 'strong_bull'), 0.3);
  assert.equal(scorePrediction('strong_bull', 'neutral'), 0.3);
});

test('scorePrediction: 方向反=-0.5', async () => {
  const { scorePrediction } = await import('../../lib/eval/runner.js');
  assert.equal(scorePrediction('bull', 'bear'), -0.5);
  assert.equal(scorePrediction('strong_bull', 'bear'), -0.5);
});

test('scorePrediction: 强方向反=-1.0', async () => {
  const { scorePrediction } = await import('../../lib/eval/runner.js');
  assert.equal(scorePrediction('strong_bull', 'strong_bear'), -1.0);
});

// ---- groundTruth判定 ----

test('groundTruth: alpha阈值', async () => {
  const { buildDataset } = await import('../../lib/eval/dataset-builder.js');
  // 间接测试：通过mock buildDataset
  const mockFetch = async () => {
    const klines = [];
    for (let i = 0; i < 100; i++) {
      const y = 2018 + Math.floor(i / 12);
      const m = (i % 12) + 1;
      klines.push({ date: `${y}-${String(m).padStart(2, '0')}-01`, close: 10 + i * 0.3 });
    }
    return { klines };
  };
  const stocks = [{ code: '600519', market: '1', name: '贵州茅台', category: 'trend_strong', industry: '白酒' }];

  const result = await buildDataset(stocks, {
    fetchKlines: mockFetch,
    fetchIndexKlines: async () => ({ klines: Array.from({ length: 100 }, (_, i) => { const y = 2018 + Math.floor(i / 12); const m = (i % 12) + 1; return { date: `${y}-${String(m).padStart(2, '0')}-01`, close: 3000 + i * 10 }; }) }),
    testPointsPerStock: 2,
    earliestMonthsAgo: 48,
    evaluationHorizonMonths: 6,
  });

  assert.ok(result.testPointCount > 0);
  const dataset = JSON.parse(fs.readFileSync(result.path, 'utf-8'));
  assert.ok(dataset.testPoints.length > 0);
  // 上涨行情中应该主要是bull/strong_bull
  const gtSet = new Set(dataset.testPoints.map((t) => t.groundTruth));
  assert.ok(gtSet.has('strong_bull') || gtSet.has('bull'));
});

// ---- report 生成 ----

test('generateEvalReport: 基本格式+预测分布', async () => {
  const { generateEvalReport } = await import('../../lib/eval/report.js');

  const runDir = path.join(os.tmpdir(), 'eastmoney-eval-test');
  fs.mkdirSync(runDir, { recursive: true });
  const runId = 'test-mock-run-v2';
  const lines = [];
  for (let i = 0; i < 40; i++) {
    lines.push(JSON.stringify({
      testPointId: `tp_${i}`, stockCode: '600519', cutoffDate: '2024-03-31',
      category: 'trend_strong', template: ['technical', 'trend', 'valuation', 'sentiment'][i % 4],
      model: 'deepseek-chat', promptVersion: 'current',
      predictedSignal: ['strong_bull', 'bull', 'neutral', 'bear'][i % 4],
      groundTruth: i % 3 === 0 ? 'bull' : 'strong_bull',
      score: [1.0, 0.5, 0.3, -0.5][i % 4], cost: 0.02, alpha: 8.5,
    }) + '\n');
  }
  fs.writeFileSync(`${runDir}/${runId}.jsonl`, lines.join(''));

  const reportPath = generateEvalReport(runId);
  const report = fs.readFileSync(reportPath, 'utf-8');
  assert.ok(report.includes('# Eval 报告'));
  assert.ok(report.includes('总体准确率'));
  assert.ok(report.includes('Prediction Distribution'));
  assert.ok(report.includes('预测占比'));
  assert.ok(report.includes('groundTruth占比'));
  assert.ok(report.includes('分模板表现'));
  assert.ok(report.includes('分股票类别表现'));
});

// ---- compareRuns ----

test('compareRuns: 对比两个版本', async () => {
  const { compareRuns } = await import('../../lib/eval/report.js');
  const runDir = path.join(os.tmpdir(), 'eastmoney-eval-compare');
  fs.mkdirSync(runDir, { recursive: true });

  // 写 v1
  const lines1 = [];
  for (let i = 0; i < 20; i++) {
    lines1.push(JSON.stringify({
      testPointId: `tp_${i}`, stockCode: '600519', cutoffDate: '2024-03-31',
      category: 'trend_strong', template: 'technical',
      model: 'deepseek-chat', promptVersion: 'v1',
      predictedSignal: 'neutral', groundTruth: 'bull', score: 0.3, cost: 0.02, alpha: 5,
    }) + '\n');
  }
  fs.writeFileSync(`${runDir}/v1.jsonl`, lines1.join(''));

  // 写 v2（改善）
  const lines2 = [];
  for (let i = 0; i < 20; i++) {
    lines2.push(JSON.stringify({
      testPointId: `tp_${i}`, stockCode: '600519', cutoffDate: '2024-03-31',
      category: 'trend_strong', template: 'technical',
      model: 'deepseek-chat', promptVersion: 'v2',
      predictedSignal: 'bull', groundTruth: 'bull', score: 0.5, cost: 0.02, alpha: 5,
    }) + '\n');
  }
  fs.writeFileSync(`${runDir}/v2.jsonl`, lines2.join(''));

  const reportPath = compareRuns('v1', 'v2');
  const report = fs.readFileSync(reportPath, 'utf-8');
  assert.ok(report.includes('v1 vs v2'));
  assert.ok(report.includes('↑')); // 应该有改善
});

// ---- seed-stocks ----

test('seed-stocks: 6类齐全≥40只', () => {
  const seedPath = path.join(__dirname, '..', '..', 'lib', 'eval', 'seed-stocks.json');
  const seeds = JSON.parse(fs.readFileSync(seedPath, 'utf-8'));
  const categories = Object.keys(seeds.stocks);
  assert.equal(categories.length, 6);
  let total = 0;
  for (const cat of categories) {
    assert.ok(seeds.stocks[cat].length >= 5, `${cat}: ${seeds.stocks[cat].length} < 5`);
    total += seeds.stocks[cat].length;
  }
  assert.ok(total >= 40, `总股票数 ${total} < 40`);
  // 检查无重复code
  const codes = new Set();
  for (const cat of categories) {
    for (const s of seeds.stocks[cat]) {
      assert.ok(!codes.has(s.code), `重复code: ${s.code}`);
      codes.add(s.code);
    }
  }
});
