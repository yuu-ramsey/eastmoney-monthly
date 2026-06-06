import { test } from 'node:test';
import assert from 'node:assert/strict';

// timeBucket and isoWeekFromDate pure function copy (same as background.js)
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

test('timeBucket: monthly returns YYYY-MM', () => {
  assert.equal(timeBucket('2026-04-30', 'monthly'), '2026-04');
  assert.equal(timeBucket('2026-01-01', 'monthly'), '2026-01');
  assert.equal(timeBucket('2026-12-31 15:30:00', 'monthly'), '2026-12');
});

test('timeBucket: daily returns YYYY-MM-DD', () => {
  assert.equal(timeBucket('2026-04-30', 'daily'), '2026-04-30');
  assert.equal(timeBucket('2026-01-01 00:00:00', 'daily'), '2026-01-01');
});

test('timeBucket: weekly returns ISO week', () => {
  // 2026-04-30 is Thursday -> week 18 of 2026
  assert.equal(timeBucket('2026-04-30', 'weekly'), '2026-W18');
  // 2026-01-01 is Thursday -> 2026-W01
  assert.equal(timeBucket('2026-01-01', 'weekly'), '2026-W01');
});

test('isoWeekFromDate: year-end cross-year attribution', () => {
  // 2025-12-31 is Wednesday, belongs to week 1 of 2026
  assert.equal(isoWeekFromDate('2025-12-31'), '2026-W01');
});

test('isoWeekFromDate: year-start attribution', () => {
  // 2026-01-01 is Thursday, belongs to 2026-W01
  assert.equal(isoWeekFromDate('2026-01-01'), '2026-W01');
  // 2026-01-04 is Sunday, belongs to 2026-W01
  assert.equal(isoWeekFromDate('2026-01-04'), '2026-W01');
  // 2026-01-05 is Monday, belongs to 2026-W02
  assert.equal(isoWeekFromDate('2026-01-05'), '2026-W02');
});

test('timeBucket: unknown period defaults to monthly', () => {
  assert.equal(timeBucket('2026-04-30', 'unknown'), '2026-04');
  assert.equal(timeBucket('2026-04-30'), '2026-04');
});
