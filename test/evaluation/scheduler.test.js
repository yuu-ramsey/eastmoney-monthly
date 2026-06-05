// 调度路由 + 安全开关测试
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { getScheduleForDate } from '../../lib/scanner/scheduler.js';

test('5月3日(周日,第1周)→HS300+自选股+日报', () => {
  const d = new Date('2026-05-03T08:00:00Z');
  const s = getScheduleForDate(d);
  assert.equal(s.runHs300, true);
  assert.equal(s.runWatchlist, true);
  assert.equal(s.runDailyReport, true);
  assert.ok(s.reason.includes('HS300周'));
});

test('5月10日(周日,第2周)→仅自选股+日报', () => {
  const d = new Date('2026-05-10T08:00:00Z');
  const s = getScheduleForDate(d);
  assert.equal(s.runHs300, false);
  assert.equal(s.runWatchlist, true);
  assert.equal(s.runDailyReport, true);
  assert.ok(s.reason.includes('自选股周'));
});

test('5月17日(周日,第3周)→HS300+自选股+日报', () => {
  const d = new Date('2026-05-17T08:00:00Z');
  const s = getScheduleForDate(d);
  assert.equal(s.runHs300, true);
  assert.equal(s.runWatchlist, true);
  assert.equal(s.runDailyReport, true);
});

test('5月14日(周三)→不触发批量扫描', () => {
  const d = new Date('2026-05-14T08:00:00Z');
  const s = getScheduleForDate(d);
  assert.equal(s.runHs300, false);
  assert.equal(s.runWatchlist, false);
  assert.equal(s.runDailyReport, false);
  assert.ok(s.reason.includes('evaluation'));
});

test('非周日(周一~周六)都只跑evaluation', () => {
  // 2026-05-04 = Monday, 一周 Mon-Sat 都不触发批量扫描
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
