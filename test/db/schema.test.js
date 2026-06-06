// schema test - table structure + index verification
import { test } from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { initSchema, tableForPeriod, PERIOD_TABLES } from '../../lib/db/schema.js';

function newDb() {
  const db = new Database(':memory:');
  initSchema(db);
  return db;
}

test('all tables created successfully', () => {
  const db = newDb();
  const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").all().map((r) => r.name);
  assert.ok(tables.includes('stocks'));
  assert.ok(tables.includes('monthly_klines'));
  assert.ok(tables.includes('weekly_klines'));
  assert.ok(tables.includes('daily_klines'));
  assert.ok(tables.includes('kline_60min'));
  assert.ok(tables.includes('adjust_events'));
});

test('all indexes created successfully', () => {
  const db = newDb();
  const count = db.prepare("SELECT count(*) as c FROM sqlite_master WHERE type='index'").get();
  assert.ok(count.c >= 8);
});

test('tableForPeriod returns correct table names', () => {
  assert.equal(tableForPeriod('monthly'), 'monthly_klines');
  assert.equal(tableForPeriod('weekly'), 'weekly_klines');
  assert.equal(tableForPeriod('daily'), 'daily_klines');
  assert.equal(tableForPeriod('60min'), 'kline_60min');
});

test('tableForPeriod unknown period throws error', () => {
  assert.throws(() => tableForPeriod('yearly'), /未知周期/);
});

test('PERIOD_TABLES contains 4 periods', () => {
  assert.equal(Object.keys(PERIOD_TABLES).length, 4);
  assert.equal(PERIOD_TABLES.monthly, 'monthly_klines');
  assert.equal(PERIOD_TABLES['60min'], 'kline_60min');
});

test('K-line tables have correct columns', () => {
  const db = newDb();
  const cols = db.prepare('PRAGMA table_info(monthly_klines)').all().map((c) => c.name);
  assert.ok(cols.includes('code'));
  assert.ok(cols.includes('date'));
  assert.ok(cols.includes('open'));
  assert.ok(cols.includes('close'));
  assert.ok(cols.includes('high'));
  assert.ok(cols.includes('low'));
  assert.ok(cols.includes('volume'));
  assert.ok(cols.includes('amount'));
  assert.ok(cols.includes('amplitude'));
  assert.ok(cols.includes('change_percent'));
  assert.ok(cols.includes('change_amount'));
  assert.ok(cols.includes('turnover_rate'));
  assert.ok(cols.includes('adjust'));
});

test('stocks table primary key constraint', () => {
  const db = newDb();
  db.prepare("INSERT INTO stocks (code, market, name) VALUES ('600519', '1', '茅台')").run();
  assert.throws(() => {
    db.prepare("INSERT INTO stocks (code, market, name) VALUES ('600519', '1', '重复')").run();
  }, /UNIQUE/);
});

test(':memory: database journal_mode is memory', () => {
  const db = newDb();
  const jm = db.prepare('PRAGMA journal_mode').get();
  assert.equal(jm.journal_mode, 'memory'); // :memory: does not support WAL
});
