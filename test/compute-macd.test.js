import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeMACD } from '../lib/compute-macd.js';

// Helper: manual EMA calculation for cross-validation
function emaRef(values, period) {
  const out = new Array(values.length).fill(null);
  if (values.length < period) return out;
  let sum = 0;
  for (let i = 0; i < period; i++) sum += values[i];
  out[period - 1] = sum / period;
  const k = 2 / (period + 1);
  for (let i = period; i < values.length; i++) {
    out[i] = values[i] * k + out[i - 1] * (1 - k);
  }
  return out;
}

test('computeMACD: when data fewer than slow period, dif/dea/hist are all null', () => {
  const closes = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
  const { dif, dea, hist } = computeMACD(closes, 3, 12, 5);
  assert.equal(dif.length, closes.length);
  assert.equal(dea.length, closes.length);
  assert.equal(hist.length, closes.length);
  for (let i = 0; i < closes.length; i++) {
    assert.equal(dif[i], null);
    assert.equal(dea[i], null);
    assert.equal(hist[i], null);
  }
});

test('computeMACD: leading null ranges of dif/dea/hist are correct', () => {
  // slow=5, signal=3 -> dif[0..3] null, dif[4] has value; dea[4..5] null, dea[6] has value
  const closes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const { dif, dea, hist } = computeMACD(closes, 2, 5, 3);

  // dif: slow=5, first value at index 4
  for (let i = 0; i < 4; i++) assert.equal(dif[i], null);
  assert.notEqual(dif[4], null);

  // dea: dif has values from index 4, signal=3, so index 4,5 are null (fewer than 3), index 6 is SMA initial value
  for (let i = 0; i < 6; i++) assert.equal(dea[i], null);
  assert.notEqual(dea[6], null);

  // hist: dea has values from index 6, so hist[0..5] null, hist[6] has value
  for (let i = 0; i < 6; i++) assert.equal(hist[i], null);
  assert.notEqual(hist[6], null);
});

test('computeMACD: cross-validation with manual EMA', () => {
  const closes = [10, 10.5, 10.3, 10.8, 11, 11.2, 11.5, 11.3, 11, 10.8, 10.5, 10.7,
    11, 11.3, 11.6, 12, 12.3, 12.1, 11.8, 11.5, 11.3, 11, 11.2, 11.5, 11.8, 12,
    12.5, 13, 13.5, 14];
  const { dif, dea, hist } = computeMACD(closes);

  // Manual EMA calculation
  const ema12 = emaRef(closes, 12);
  const ema26 = emaRef(closes, 26);

  // Verify DIF
  for (let i = 0; i < closes.length; i++) {
    if (ema12[i] !== null && ema26[i] !== null) {
      assert.ok(Math.abs(dif[i] - (ema12[i] - ema26[i])) < 1e-10,
        `DIF[${i}]: expected ${ema12[i] - ema26[i]}, got ${dif[i]}`);
    } else {
      assert.equal(dif[i], null);
    }
  }

  // Verify DEA: EMA(9) on non-null DIF
  const difDense = dif.filter((v) => v !== null);
  const deaExpected = emaRef(difDense, 9);
  let di = 0;
  for (let i = 0; i < closes.length; i++) {
    if (dif[i] !== null) {
      if (deaExpected[di] !== null) {
        assert.ok(Math.abs(dea[i] - deaExpected[di]) < 1e-10,
          `DEA[${i}]: expected ${deaExpected[di]}, got ${dea[i]}`);
      } else {
        assert.equal(dea[i], null);
      }
      di++;
    }
  }

  // Verify HIST
  for (let i = 0; i < closes.length; i++) {
    if (dif[i] !== null && dea[i] !== null) {
      const expectedHist = 2 * (dif[i] - dea[i]);
      assert.ok(Math.abs(hist[i] - expectedHist) < 1e-10,
        `HIST[${i}]: expected ${expectedHist}, got ${hist[i]}`);
    } else {
      assert.equal(hist[i], null);
    }
  }
});

test('computeMACD: invalid input returns three empty arrays', () => {
  assert.deepEqual(computeMACD(null), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD(undefined), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD('foo'), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD([]), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD([1, 2, 3], 0, 26, 9), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD([1, 2, 3], -1, 26, 9), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD([1, 2, 3], 1.5, 26, 9), { dif: [], dea: [], hist: [] });
});

test('computeMACD: fast >= slow returns empty array', () => {
  assert.deepEqual(computeMACD([1, 2, 3], 12, 12, 9), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD([1, 2, 3], 26, 12, 9), { dif: [], dea: [], hist: [] });
});

test('computeMACD: default params fast=12 slow=26 signal=9', () => {
  const closes = Array.from({ length: 35 }, (_, i) => 20 + i * 0.5);
  const { dif, dea, hist } = computeMACD(closes);
  assert.equal(dif.length, 35);
  assert.equal(dea.length, 35);
  assert.equal(hist.length, 35);
  // slow=26, first dif at index 25
  assert.notEqual(dif[25], null);
  for (let i = 0; i < 25; i++) assert.equal(dif[i], null);
});
