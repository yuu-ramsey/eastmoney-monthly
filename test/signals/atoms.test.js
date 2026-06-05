import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cross, exist, count, hhv, llv, every } from '../../lib/signals/atoms.js';

test('cross: A上穿B', () => {
  const a = [1, 2, 1, 1, 3], b = [2, 2, 2, 2, 2];
  const r = cross(a, b);
  assert.equal(r.crossed, true);
  assert.equal(r.direction, 'up');
  assert.equal(r.atIndex, 4);
});

test('cross: A下穿B', () => {
  const a = [5, 4, 5, 5, 1], b = [3, 3, 3, 3, 3];
  const r = cross(a, b);
  assert.equal(r.crossed, true);
  assert.equal(r.direction, 'down');
  assert.equal(r.atIndex, 4);
});

test('cross: 未穿越', () => {
  const r = cross([1, 2, 3], [0, 0, 0]);
  assert.equal(r.crossed, false);
});

test('cross: 空数组', () => {
  const r = cross([], []);
  assert.equal(r.crossed, false);
});

test('cross: lookback=3', () => {
  const a = [1, 2, 1, 1, 3], b = [1.5, 1.5, 1.5, 1.5, 1.5];
  const r = cross(a, b, 3);
  assert.equal(r.crossed, true);
});

test('exist: 存在满足条件', () => {
  assert.equal(exist([false, false, true, false], 4), true);
});

test('exist: 不存在', () => {
  assert.equal(exist([false, false, false], 3), false);
});

test('count: 统计满足次数', () => {
  assert.equal(count([true, false, true, true], 4), 3);
});

test('count: n限制', () => {
  assert.equal(count([true, true, true, true, true], 2), 2);
});

test('hhv: 最高值', () => {
  assert.equal(hhv([10, 5, 20, 15, 8], 5), 20);
});

test('hhv: n=2', () => {
  assert.equal(hhv([10, 5, 20, 15, 8], 2), 15);
});

test('llv: 最低值', () => {
  assert.equal(llv([10, 5, 20, 15, 8], 5), 5);
});

test('every: 全部满足', () => {
  assert.equal(every([true, true, true], 3), true);
});

test('every: 不满足', () => {
  assert.equal(every([true, false, true], 3), false);
});
