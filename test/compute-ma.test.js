import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeMA } from '../lib/compute-ma.js';

test('computeMA: MA5 standard calculation', () => {
  // [1,2,3,4,5,6,7]: MA5 at i=4 is (1+2+3+4+5)/5=3, at i=5 is (2+3+4+5+6)/5=4, at i=6 is 5
  assert.deepEqual(computeMA([1, 2, 3, 4, 5, 6, 7], 5), [null, null, null, null, 3, 4, 5]);
});

test('computeMA: data fewer than period returns all null', () => {
  assert.deepEqual(computeMA([1, 2, 3], 5), [null, null, null]);
});

test('computeMA: empty array', () => {
  assert.deepEqual(computeMA([], 5), []);
});

test('computeMA: period=1 returns original values', () => {
  assert.deepEqual(computeMA([1, 2, 3], 1), [1, 2, 3]);
});

test('computeMA: invalid input returns empty array', () => {
  assert.deepEqual(computeMA(null, 5), []);
  assert.deepEqual(computeMA([1, 2], 0), []);
  assert.deepEqual(computeMA([1, 2], -1), []);
  assert.deepEqual(computeMA([1, 2], 1.5), []);
});

test('computeMA: MA20 first has value at 20th item', () => {
  const values = Array.from({ length: 25 }, (_, i) => i + 1);
  const out = computeMA(values, 20);
  assert.equal(out[18], null);
  assert.equal(out[19], (1 + 20) / 2); // (1+2+...+20)/20 = 10.5
  assert.equal(out[24], (6 + 25) / 2); // (6+7+...+25)/20 = 15.5
});
