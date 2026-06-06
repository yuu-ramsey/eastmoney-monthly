// K-line data health check test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { checkKlines } from '../../lib/data-validation/validate-klines.js';

function makeKline(date, o, c, h, l, v = 1000) {
  return { date, open: o, close: c, high: h, low: l, volume: v };
}

test('normal data returns ok', () => {
  const k = Array(20).fill(null).map((_, i) => makeKline('2026-' + String(i+1).padStart(2,'0'), 100 + i, 105 + i, 108 + i, 98 + i));
  const r = checkKlines(k);
  assert.equal(r.valid, true);
  assert.equal(r.severity, 'ok');
});

test('empty array returns error', () => {
  const r = checkKlines([]);
  assert.equal(r.valid, false);
  assert.equal(r.severity, 'error');
});

test('< 12 bars returns warn', () => {
  const k = Array(5).fill(null).map((_, i) => makeKline('2026-0'+(i+1), 100, 105, 110, 95));
  const r = checkKlines(k);
  assert.equal(r.severity, 'warn');
});

test('close <= 0 返回 error', () => {
  const k = Array(15).fill(null).map((_, i) => makeKline('2026-' + String(i+1).padStart(2,'0'), 100, i === 3 ? 0 : 105, 110, 95));
  const r = checkKlines(k);
  assert.equal(r.severity, 'error');
});

test('volume < 0 返回 error', () => {
  const k = Array(15).fill(null).map((_, i) => makeKline('2026-'+String(i+1).padStart(2,'0'), 100, 105, 110, 95, i === 5 ? -100 : 1000));
  const r = checkKlines(k);
  assert.equal(r.severity, 'error');
});

test('low > high 返回 error', () => {
  const k = Array(20).fill(null).map((_, i) => makeKline('2026-'+String(i+1).padStart(2,'0'), 100, 105, i === 3 ? 50 : 110, i === 3 ? 200 : 95));
  const r = checkKlines(k);
  assert.ok(r.issues.some(i => i.severity === 'error' && i.field === 'low/high'));
});

test('single-period change% > 50% returns warn', () => {
  const k = Array(15).fill(null).map((_, i) => makeKline('2026-'+String(i+1).padStart(2,'0'), 100, i === 5 ? 200 : 105, 110, 95));
  const r = checkKlines(k);
  assert.ok(r.issues.some(i => i.severity === 'warn' && i.message.includes('50%')));
});

test('monthly: cross-month gap detection', () => {
  const k = [
    makeKline('2026-01', 100, 105, 110, 95),
    makeKline('2026-04', 105, 110, 115, 100), // skip 02, 03
  ];
  const r = checkKlines(k, 'monthly');
  assert.ok(r.issues.some(i => i.field === 'date' && i.message.includes('缺失')));
});
