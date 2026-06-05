// 核心工具测试
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sum, avg, max, min, std, emaInit, smaSmoothing } from '../../lib/indicators/core.js';

test('sum: 空数组', () => {
  assert.deepEqual(sum([], 5), []);
});

test('sum: 滚动求和', () => {
  const arr = [1, 2, 3, 4, 5];
  const s = sum(arr, 3);
  assert.equal(s[0], null);
  assert.equal(s[1], null);
  assert.equal(s[2], 6); // 1+2+3
  assert.equal(s[3], 9); // 2+3+4
  assert.equal(s[4], 12); // 3+4+5
});

test('avg: 标准计算', () => {
  const arr = [10, 20, 30, 40, 50];
  const a = avg(arr, 3);
  assert.equal(a[2], 20); // (10+20+30)/3
  assert.equal(a[3], 30);
  assert.equal(a[4], 40);
});

test('avg: 数据少于 period', () => {
  const a = avg([1, 2], 5);
  assert.deepEqual(a, [null, null]);
});

test('max: 滚动最高', () => {
  const arr = [3, 1, 4, 1, 5, 9, 2, 6];
  const m = max(arr, 3);
  assert.equal(m[2], 4);
  assert.equal(m[4], 5);
  assert.equal(m[5], 9);
});

test('min: 滚动最低', () => {
  const arr = [5, 3, 8, 6, 1, 4];
  const m = min(arr, 3);
  assert.equal(m[2], 3);
  assert.equal(m[4], 1);
});

test('std: 常数数组 std=0', () => {
  const arr = [5, 5, 5, 5, 5];
  const s = std(arr, 3);
  for (let i = 2; i < 5; i++) assert.ok(Math.abs(s[i]) < 0.001);
});

test('emaInit: 正常计算', () => {
  assert.equal(emaInit([10, 20, 30], 3), 20);
  assert.equal(emaInit([5, 5, 5, 5, 5], 5), 5);
});

test('emaInit: 空数组返回 null', () => {
  assert.equal(emaInit([], 5), null);
});

test('smaSmoothing: 通达信 SMA(X,3,1)', () => {
  // M=1: 1/3 * current + 2/3 * prev
  const r = smaSmoothing(50, 100, 3, 1);
  assert.ok(Math.abs(r - 66.6667) < 0.01);
});
