import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseKlines } from '../lib/parse-klines.js';

test('parseKlines: single row parsing', () => {
  const out = parseKlines([
    '2024-01-31,1700.00,1750.50,1780.00,1690.00,1234567,12345678.90,5.0,2.5,40.0,1.5',
  ]);
  assert.equal(out.length, 1);
  assert.equal(out[0].date, '2024-01-31');
  assert.equal(out[0].open, 1700);
  assert.equal(out[0].close, 1750.5);
  assert.equal(out[0].high, 1780);
  assert.equal(out[0].low, 1690);
  assert.equal(out[0].changePercent, 2.5);
  assert.equal(out[0].turnoverRate, 1.5);
});

test('parseKlines: 空数组', () => {
  assert.deepEqual(parseKlines([]), []);
});

test('parseKlines: 非数组输入', () => {
  assert.deepEqual(parseKlines(null), []);
  assert.deepEqual(parseKlines(undefined), []);
  assert.deepEqual(parseKlines('foo'), []);
});

test('parseKlines: 字段不够的行被丢弃', () => {
  assert.deepEqual(parseKlines(['2024-01,1.0,2.0']), []);
});

test('parseKlines: 多行', () => {
  const out = parseKlines([
    '2024-01-31,1700,1750,1780,1690,100,1000,5,2,40,1',
    '2024-02-29,1750,1800,1820,1740,200,2000,4,3,50,2',
  ]);
  assert.equal(out.length, 2);
  assert.equal(out[0].close, 1750);
  assert.equal(out[1].close, 1800);
});

test('parseKlines: open/close 非数字的行丢弃', () => {
  const out = parseKlines([
    'X,abc,def,1,1,1,1,1,1,1,1',
    '2024-01,1,2,3,4,5,6,7,8,9,10',
  ]);
  assert.equal(out.length, 1);
  assert.equal(out[0].date, '2024-01');
});
