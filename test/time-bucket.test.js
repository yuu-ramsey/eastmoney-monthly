import { test } from 'node:test';
import assert from 'node:assert/strict';

// timeBucket 和 isoWeekFromDate 的纯函数副本（与 background.js 同款）
function timeBucket(dateStr, period) {
  switch (period) {
    case 'daily':
      return String(dateStr).slice(0, 10);
    case 'weekly':
      return isoWeekFromDate(dateStr);
    case 'monthly':
    default:
      return String(dateStr).slice(0, 7);
  }
}

function isoWeekFromDate(dateStr) {
  const d = new Date(dateStr.replace(/\s.*$/, '') + 'T00:00:00');
  const dayOfWeek = d.getDay() || 7;
  d.setDate(d.getDate() + 4 - dayOfWeek);
  const year = d.getFullYear();
  const yearStart = new Date(year, 0, 1);
  const weekNum = Math.ceil(((d - yearStart) / 86400000 + 1) / 7);
  return `${year}-W${String(weekNum).padStart(2, '0')}`;
}

test('timeBucket: monthly 返回 YYYY-MM', () => {
  assert.equal(timeBucket('2026-04-30', 'monthly'), '2026-04');
  assert.equal(timeBucket('2026-01-01', 'monthly'), '2026-01');
  assert.equal(timeBucket('2026-12-31 15:30:00', 'monthly'), '2026-12');
});

test('timeBucket: daily 返回 YYYY-MM-DD', () => {
  assert.equal(timeBucket('2026-04-30', 'daily'), '2026-04-30');
  assert.equal(timeBucket('2026-01-01 00:00:00', 'daily'), '2026-01-01');
});

test('timeBucket: weekly 返回 ISO 周', () => {
  // 2026-04-30 是周四 → 2026 年的第 18 周
  assert.equal(timeBucket('2026-04-30', 'weekly'), '2026-W18');
  // 2026-01-01 是周四 → 2026-W01
  assert.equal(timeBucket('2026-01-01', 'weekly'), '2026-W01');
});

test('isoWeekFromDate: 年底跨年归属', () => {
  // 2025-12-31 是周三,属于 2026 年的第 1 周
  assert.equal(isoWeekFromDate('2025-12-31'), '2026-W01');
});

test('isoWeekFromDate: 年初归属', () => {
  // 2026-01-01 是周四,属于 2026-W01
  assert.equal(isoWeekFromDate('2026-01-01'), '2026-W01');
  // 2026-01-04 是周日,属于 2026-W01
  assert.equal(isoWeekFromDate('2026-01-04'), '2026-W01');
  // 2026-01-05 是周一,属于 2026-W02
  assert.equal(isoWeekFromDate('2026-01-05'), '2026-W02');
});

test('timeBucket: 未知 period 默认走 monthly', () => {
  assert.equal(timeBucket('2026-04-30', 'unknown'), '2026-04');
  assert.equal(timeBucket('2026-04-30'), '2026-04');
});
