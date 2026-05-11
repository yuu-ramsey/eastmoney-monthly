import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeMACD } from '../lib/compute-macd.js';

// 辅助:手工计算 EMA,用于交叉验证
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

test('computeMACD: 数据少于 slow 周期时 dif/dea/hist 全为 null', () => {
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

test('computeMACD: dif/dea/hist 前部 null 区间正确', () => {
  // slow=5,信号=3 → dif[0..3] null,dif[4] 有值;dea[4..5] null,dea[6] 有值
  const closes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const { dif, dea, hist } = computeMACD(closes, 2, 5, 3);

  // dif: slow=5,首个值在 index 4
  for (let i = 0; i < 4; i++) assert.equal(dif[i], null);
  assert.notEqual(dif[4], null);

  // dea:dif 从 index 4 才有值,信号=3,所以 index 4,5 为 null(不足 3 个),index 6 为 SMA 初值
  for (let i = 0; i < 6; i++) assert.equal(dea[i], null);
  assert.notEqual(dea[6], null);

  // hist:dea 从 index 6 才有值,所以 hist[0..5] null,hist[6] 有值
  for (let i = 0; i < 6; i++) assert.equal(hist[i], null);
  assert.notEqual(hist[6], null);
});

test('computeMACD: 与手工 EMA 交叉验证', () => {
  const closes = [10, 10.5, 10.3, 10.8, 11, 11.2, 11.5, 11.3, 11, 10.8, 10.5, 10.7,
    11, 11.3, 11.6, 12, 12.3, 12.1, 11.8, 11.5, 11.3, 11, 11.2, 11.5, 11.8, 12,
    12.5, 13, 13.5, 14];
  const { dif, dea, hist } = computeMACD(closes);

  // 手工计算 EMA
  const ema12 = emaRef(closes, 12);
  const ema26 = emaRef(closes, 26);

  // 验证 DIF
  for (let i = 0; i < closes.length; i++) {
    if (ema12[i] !== null && ema26[i] !== null) {
      assert.ok(Math.abs(dif[i] - (ema12[i] - ema26[i])) < 1e-10,
        `DIF[${i}]:期望 ${ema12[i] - ema26[i]},实际 ${dif[i]}`);
    } else {
      assert.equal(dif[i], null);
    }
  }

  // 验证 DEA:对非 null DIF 取 EMA(9)
  const difDense = dif.filter((v) => v !== null);
  const deaExpected = emaRef(difDense, 9);
  let di = 0;
  for (let i = 0; i < closes.length; i++) {
    if (dif[i] !== null) {
      if (deaExpected[di] !== null) {
        assert.ok(Math.abs(dea[i] - deaExpected[di]) < 1e-10,
          `DEA[${i}]:期望 ${deaExpected[di]},实际 ${dea[i]}`);
      } else {
        assert.equal(dea[i], null);
      }
      di++;
    }
  }

  // 验证 HIST
  for (let i = 0; i < closes.length; i++) {
    if (dif[i] !== null && dea[i] !== null) {
      const expectedHist = 2 * (dif[i] - dea[i]);
      assert.ok(Math.abs(hist[i] - expectedHist) < 1e-10,
        `HIST[${i}]:期望 ${expectedHist},实际 ${hist[i]}`);
    } else {
      assert.equal(hist[i], null);
    }
  }
});

test('computeMACD: 非法输入返回三个空数组', () => {
  assert.deepEqual(computeMACD(null), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD(undefined), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD('foo'), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD([]), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD([1, 2, 3], 0, 26, 9), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD([1, 2, 3], -1, 26, 9), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD([1, 2, 3], 1.5, 26, 9), { dif: [], dea: [], hist: [] });
});

test('computeMACD: fast >= slow 返回空数组', () => {
  assert.deepEqual(computeMACD([1, 2, 3], 12, 12, 9), { dif: [], dea: [], hist: [] });
  assert.deepEqual(computeMACD([1, 2, 3], 26, 12, 9), { dif: [], dea: [], hist: [] });
});

test('computeMACD: 默认参数 fast=12 slow=26 signal=9', () => {
  const closes = Array.from({ length: 35 }, (_, i) => 20 + i * 0.5);
  const { dif, dea, hist } = computeMACD(closes);
  assert.equal(dif.length, 35);
  assert.equal(dea.length, 35);
  assert.equal(hist.length, 35);
  // slow=26,首个 dif 在 index 25
  assert.notEqual(dif[25], null);
  for (let i = 0; i < 25; i++) assert.equal(dif[i], null);
});
