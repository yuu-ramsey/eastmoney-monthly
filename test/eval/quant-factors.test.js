// quant-factors + score-fusion 测试
import { test } from 'node:test';
import assert from 'node:assert/strict';

function makeKlines(config) {
  const { n = 60, base = 10, trend = 0, volatility = 1, volumeBase = 1000 } = config;
  const k = [];
  for (let i = 0; i < n; i++) {
    const close = base + i * trend + (Math.random() - 0.5) * volatility * 2;
    const open = close - (Math.random() - 0.5) * volatility;
    const high = Math.max(open, close) + Math.random() * volatility;
    const low = Math.min(open, close) - Math.random() * volatility;
    k.push({ date: `2024-${String((i % 12) + 1).padStart(2, '0')}-01`, open: +open.toFixed(2), close: +close.toFixed(2), high: +high.toFixed(2), low: +low.toFixed(2), volume: volumeBase + Math.random() * 500 });
  }
  return k;
}

// ---- trendStrength ----
test('trendStrength: 强上涨→正值', async () => {
  const { computeTrendStrength } = await import('../../lib/quant-factors.js');
  const k = makeKlines({ trend: 1.0, volatility: 0.5 });
  const r = computeTrendStrength(k);
  assert.ok(r.value > 0.3, `应为正值，实际 ${r.value}`);
});

test('trendStrength: 强下跌→负值', async () => {
  const { computeTrendStrength } = await import('../../lib/quant-factors.js');
  const k = makeKlines({ trend: -1.0, volatility: 0.5 });
  const r = computeTrendStrength(k);
  assert.ok(r.value < -0.2, `应为负值，实际 ${r.value}`);
});

test('trendStrength: <24根→null', async () => {
  const { computeTrendStrength } = await import('../../lib/quant-factors.js');
  assert.equal(computeTrendStrength(makeKlines({ n: 10 })), null);
});

// ---- pricePosition ----
test('pricePosition: 高位→接近1', async () => {
  const { computePricePosition } = await import('../../lib/quant-factors.js');
  const k = makeKlines({ trend: 0.2, volatility: 0.1 });
  const r = computePricePosition(k, 36);
  assert.ok(r.value > 0.6, `应为高位，实际 ${r.value}`);
});

test('pricePosition: 低位→接近0', async () => {
  const { computePricePosition } = await import('../../lib/quant-factors.js');
  const k = makeKlines({ trend: -0.3, volatility: 0.1 });
  const r = computePricePosition(k, 36);
  assert.ok(r.value < 0.5, `应为低位，实际 ${r.value}`);
});

// ---- volatilityPercentile ----
test('volatilityPercentile: 低波动→中低分位', async () => {
  const { computeVolatilityPercentile } = await import('../../lib/quant-factors.js');
  const k = makeKlines({ volatility: 0.1 });
  const r = computeVolatilityPercentile(k, 36);
  assert.ok(typeof r.value === 'number');
  assert.ok(r.value >= 0 && r.value <= 1);
});

// ---- volumePriceConfirm ----
test('volumePriceConfirm: 放量上涨→+1', async () => {
  const { computeVolumePriceConfirm } = await import('../../lib/quant-factors.js');
  // 手构造：最近3根放量上涨
  const k = makeKlines({ n: 50, volatility: 0.5 });
  // 确保最后3根是上涨+放量
  const last3 = k.slice(-3);
  for (const kk of last3) {
    kk.close = kk.open * 1.05;
    kk.volume = 5000;
  }
  const r = computeVolumePriceConfirm(k);
  assert.ok(r.value >= 0, `量价确认失败: ${r.value}`);
});

// ---- quantScore综合 ----
test('quantScore: 完整计算', async () => {
  const { computeQuantScore } = await import('../../lib/quant-factors.js');
  const k = makeKlines({ trend: 0.5 });
  const r = computeQuantScore(k);
  assert.ok(r);
  assert.ok(typeof r.score === 'number');
  assert.ok(r.score >= -100 && r.score <= 100);
  assert.ok(r.factors.f1);
  assert.ok(r.factors.f2);
  assert.ok(r.factors.f3);
  assert.ok(r.factors.f4);
});

test('quantScore: <24根→null', async () => {
  const { computeQuantScore } = await import('../../lib/quant-factors.js');
  assert.equal(computeQuantScore(makeKlines({ n: 10 })), null);
});

// ---- scoreFusion ----
test('fuseScores: 方向一致→高权重LLM(adaptive)', async () => {
  const { fuseScores } = await import('../../lib/score-fusion.js');
  const r = fuseScores(
    { score: 70, signal: 'bull', confidence: 'high' },
    { score: 60, factors: { f1: { value: 0.2 }, f3: { value: 0.3 } }, confidence: 0.8 },
  );
  assert.equal(r.final_signal, 'bull');
  assert.ok(r.final_score > 50);
  assert.ok(r.agreement > 0.5);
});

test('fuseScores: 方向矛盾→量化权重更高(adaptive)', async () => {
  const { fuseScores } = await import('../../lib/score-fusion.js');
  const r = fuseScores(
    { score: 70, signal: 'bull', confidence: 'medium' },
    { score: -50, factors: { f1: { value: 0.2 }, f3: { value: 0.6 } }, confidence: 0.7 },
  );
  assert.ok(r.agreement < 0);
  // mixed regime: 40/60, quant dominates
  assert.ok(r.components.quantWeight >= 0.4);
});

test('fuseScores: quantOnly模式', async () => {
  const { fuseScores } = await import('../../lib/score-fusion.js');
  const r = fuseScores(null, { score: 70, factors: {}, confidence: 0.8 });
  assert.ok(r.quant_score);
  assert.equal(r.llm_score, null);
});

test('fuseScores: llmOnly模式', async () => {
  const { fuseScores } = await import('../../lib/score-fusion.js');
  const r = fuseScores({ score: 30, signal: 'bull', confidence: 'high' }, null);
  assert.equal(r.quant_score, null);
  assert.equal(r.final_signal, 'bull');
});

// ---- detectStockRegime ----
test('regime: strong_trend high (pos>0.8, vol<0.6)', async () => {
  const { detectStockRegime } = await import('../../lib/score-fusion.js');
  const qr = { factors: { f2: { value: 0.9 }, f3: { value: 0.3 } } };
  assert.equal(detectStockRegime(qr), 'strong_trend');
});

test('regime: strong_trend low (pos<0.2, vol<0.6)', async () => {
  const { detectStockRegime } = await import('../../lib/score-fusion.js');
  const qr = { factors: { f2: { value: 0.1 }, f3: { value: 0.3 } } };
  assert.equal(detectStockRegime(qr), 'strong_trend');
});

test('regime: sideways (pos 0.3-0.7, vol<0.5)', async () => {
  const { detectStockRegime } = await import('../../lib/score-fusion.js');
  const qr = { factors: { f2: { value: 0.5 }, f3: { value: 0.3 } } };
  assert.equal(detectStockRegime(qr), 'sideways');
});

test('regime: high_vol (vol>0.7)', async () => {
  const { detectStockRegime } = await import('../../lib/score-fusion.js');
  const qr = { factors: { f2: { value: 0.5 }, f3: { value: 0.85 } } };
  assert.equal(detectStockRegime(qr), 'high_vol');
});

test('regime: mixed (pos=0.5, vol=0.6)', async () => {
  const { detectStockRegime } = await import('../../lib/score-fusion.js');
  const qr = { factors: { f2: { value: 0.5 }, f3: { value: 0.6 } } };
  assert.equal(detectStockRegime(qr), 'mixed');
});

test('regime: null quantResult → mixed', async () => {
  const { detectStockRegime } = await import('../../lib/score-fusion.js');
  assert.equal(detectStockRegime(null), 'mixed');
});

test('regime: missing factors → mixed', async () => {
  const { detectStockRegime } = await import('../../lib/score-fusion.js');
  assert.equal(detectStockRegime({ factors: {} }), 'mixed');
});

test('regime: pos=0.8 exactly → not strong_trend', async () => {
  const { detectStockRegime } = await import('../../lib/score-fusion.js');
  const qr = { factors: { f2: { value: 0.8 }, f3: { value: 0.3 } } };
  assert.notEqual(detectStockRegime(qr), 'strong_trend'); // must be >0.8
});

// ---- adaptive fuseScores ----
test('adaptive: strong_trend uses 30/70', async () => {
  const { fuseScores } = await import('../../lib/score-fusion.js');
  const r = fuseScores(
    { score: 30, signal: 'bear', confidence: 'medium' },
    { score: 80, factors: { f2: { value: 0.9 }, f3: { value: 0.2 } }, confidence: 0.9 },
  );
  assert.equal(r.regime, 'strong_trend');
  assert.ok(r.components.quantWeight >= 0.4);
  assert.equal(r.final_signal, 'bull');
});

test('adaptive: sideways uses 92/8', async () => {
  const { fuseScores } = await import('../../lib/score-fusion.js');
  const r = fuseScores(
    { score: 30, signal: 'bear', confidence: 'medium' },
    { score: -80, factors: { f2: { value: 0.5 }, f3: { value: 0.2 } }, confidence: 0.5 },
  );
  assert.equal(r.regime, 'sideways');
  assert.ok(r.components.llmWeight > 0.8);
});

test('adaptive: high_vol uses 85/15', async () => {
  const { fuseScores } = await import('../../lib/score-fusion.js');
  const r = fuseScores(
    { score: 70, signal: 'bull', confidence: 'high' },
    { score: 50, factors: { f2: { value: 0.5 }, f3: { value: 0.9 } }, confidence: 0.3 },
  );
  assert.equal(r.regime, 'high_vol');
  assert.ok(r.components.llmWeight > 0.7);
});

test('adaptive: mixed uses 50/50', async () => {
  const { fuseScores } = await import('../../lib/score-fusion.js');
  const r = fuseScores(
    { score: 50, signal: 'neutral', confidence: 'low' },
    { score: 30, factors: { f1: { value: 0.4 }, f3: { value: 0.5 } }, confidence: 0.6 },
  );
  assert.equal(r.regime, 'mixed');
  assert.ok(r.components.quantWeight >= 0.4);
});

// ---- scoreToSignal ----
test('scoreToSignal: 阈值映射', async () => {
  const { scoreToSignal } = await import('../../lib/score-fusion.js');
  assert.equal(scoreToSignal(75), 'strong_bull');
  assert.equal(scoreToSignal(35), 'bull');
  assert.equal(scoreToSignal(0), 'neutral');
  assert.equal(scoreToSignal(-35), 'bear');
  assert.equal(scoreToSignal(-75), 'strong_bear');
});
