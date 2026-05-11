import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeMA } from '../lib/compute-ma.js';

test('computeMA: MA5 标准计算', () => {
  // [1,2,3,4,5,6,7] 的 MA5:i=4 时 (1+2+3+4+5)/5=3,i=5 时 (2+3+4+5+6)/5=4,i=6 时 5
  assert.deepEqual(computeMA([1, 2, 3, 4, 5, 6, 7], 5), [null, null, null, null, 3, 4, 5]);
});

test('computeMA: 数据少于 period 全 null', () => {
  assert.deepEqual(computeMA([1, 2, 3], 5), [null, null, null]);
});

test('computeMA: 空数组', () => {
  assert.deepEqual(computeMA([], 5), []);
});

test('computeMA: period=1 返回原值', () => {
  assert.deepEqual(computeMA([1, 2, 3], 1), [1, 2, 3]);
});

test('computeMA: 非法输入返回空数组', () => {
  assert.deepEqual(computeMA(null, 5), []);
  assert.deepEqual(computeMA([1, 2], 0), []);
  assert.deepEqual(computeMA([1, 2], -1), []);
  assert.deepEqual(computeMA([1, 2], 1.5), []);
});

test('computeMA: MA20 在第 20 项首次有值', () => {
  const values = Array.from({ length: 25 }, (_, i) => i + 1);
  const out = computeMA(values, 20);
  assert.equal(out[18], null);
  assert.equal(out[19], (1 + 20) / 2); // (1+2+...+20)/20 = 10.5
  assert.equal(out[24], (6 + 25) / 2); // (6+7+...+25)/20 = 15.5
});
