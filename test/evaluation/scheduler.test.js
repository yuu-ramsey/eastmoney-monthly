// Scheduling router + safety switch test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { getScheduleForDate } from '../../lib/scanner/scheduler.js';

test('May 3 (Sun, week 1) -> HS300 + watchlist + daily report', () => {
  const d = new Date('2026-05-03T08:00:00Z');
  const s = getScheduleForDate(d);
  assert.equal(s.runHs300, true);
  assert.equal(s.runWatchlist, true);
  assert.equal(s.runDailyReport, true);
  assert.ok(s.reason.includes('HS300week'));
});

test('May 10 (Sun, week 2) -> watchlist only + daily report', () => {
  const d = new Date('2026-05-10T08:00:00Z');
  const s = getScheduleForDate(d);
  assert.equal(s.runHs300, false);
  assert.equal(s.runWatchlist, true);
  assert.equal(s.runDailyReport, true);
  assert.ok(s.reason.includes('watchlist week'));
});

test('May 17 (Sun, week 3) -> HS300 + watchlist + daily report', () => {
  const d = new Date('2026-05-17T08:00:00Z');
  const s = getScheduleForDate(d);
  assert.equal(s.runHs300, true);
  assert.equal(s.runWatchlist, true);
  assert.equal(s.runDailyReport, true);
});

test('May 14 (Wed) -> no batch scan triggered', () => {
  const d = new Date('2026-05-14T08:00:00Z');
  const s = getScheduleForDate(d);
  assert.equal(s.runHs300, false);
  assert.equal(s.runWatchlist, false);
  assert.equal(s.runDailyReport, false);
  assert.ok(s.reason.includes('evaluation'));
});

test('non-Sunday (Mon-Sat) all only run evaluation', () => {
  // 2026-05-04 = Monday, Mon-Sat all do not trigger batch scan
  const base = new Date('2026-05-04T08:00:00Z');
  for (let offset = 0; offset <= 5; offset++) {
    const d = new Date(base);
    d.setDate(d.getDate() + offset);
    const dow = d.getDay();
    if (dow === 0) continue; // skip Sunday
    const s = getScheduleForDate(d);
    assert.equal(s.runHs300, false, `${d.toISOString().slice(0,10)} (dayOfWeek=${dow}): hs300 should be false`);
    assert.equal(s.runWatchlist, false, `${d.toISOString().slice(0,10)}: watchlist should be false`);
    assert.equal(s.runDailyReport, false, `${d.toISOString().slice(0,10)}: report should be false`);
  }
});
