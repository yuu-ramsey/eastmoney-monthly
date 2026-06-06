// Trend indicator test + Moutai reconciliation
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sma, ema, macd } from '../../lib/indicators/trend.js';

// Moutai last 5 daily close: 1361.33, 1354.55, 1344.09, 1342.17, 1332.95
const MAOTAI_5 = [1361.33, 1354.55, 1344.09, 1342.17, 1332.95];

test('sma: Moutai MA5 manual verification', () => {
  const result = sma(MAOTAI_5, 5);
  // (1361.33 + 1354.55 + 1344.09 + 1342.17 + 1332.95) / 5
  const expected = 1347.018;
  assert.ok(Math.abs(result[4] - expected) < 0.01, `MA5=${result[4]} expected=${expected}`);
});

test('sma: less than period all null', () => {
  const r = sma([100, 200], 5);
  assert.equal(r[0], null);
  assert.equal(r[1], null);
});

test('ema: 12-period EMA', () => {
  const closes = Array(30).fill(100);
  closes[15] = 110; closes[20] = 90;
  const result = ema(closes, 12);
  assert.ok(result[29] != null);
  // should be near 100 (deviation smoothed out)
  assert.ok(Math.abs(result[29] - 100) < 5, `EMA12=${result[29]}`);
});

test('macd: 50 bars MACD has values', () => {
  // MACD(12,26,9) needs 26 + 9 = 35 bars before DEA appears
  const closes = [];
  for (let i = 0; i < 50; i++) closes.push(100 + Math.sin(i * 0.5) * 10);
  const r = macd(closes);
  assert.ok(r.dif.some(v => v != null), 'DIF should have values');
  assert.ok(r.dea.some(v => v != null), 'DEA should have values');
  assert.ok(r.hist.some(v => v != null), 'HIST should have values');
});

test('macd: < slow all null', () => {
  const r = macd([100, 200], 12, 26, 9);
  assert.equal(r.dif.find(v => v != null), undefined);
});
