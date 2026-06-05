// db-init 测试 — 进度持久化 + 数据流
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import Database from 'better-sqlite3';
import { initSchema } from '../../lib/db/schema.js';
import { saveKlinesSync } from '../../lib/db/klines-repo.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const TMP_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'test-tmp');

function setupTempDir() {
  if (!fs.existsSync(TMP_DIR)) fs.mkdirSync(TMP_DIR, { recursive: true });
}

function cleanupTempDir() {
  if (fs.existsSync(TMP_DIR)) fs.rmSync(TMP_DIR, { recursive: true, force: true });
}

test('进度文件初始加载（不存在时返回默认）', () => {
  // 模拟 loadProgress 逻辑
  const progressPath = path.join(TMP_DIR, 'nonexistent.json');
  let progress;
  try {
    progress = JSON.parse(fs.readFileSync(progressPath, 'utf-8'));
  } catch (_) {
    progress = { done: [], failed: 0 };
  }
  assert.deepEqual(progress.done, []);
  assert.equal(progress.failed, 0);
});

test('进度文件存盘后再读', () => {
  setupTempDir();
  const progressPath = path.join(TMP_DIR, 'test-progress.json');
  const data = { done: ['600519', '000001'], failed: 2, updatedAt: new Date().toISOString() };
  fs.writeFileSync(progressPath, JSON.stringify(data, null, 2), 'utf-8');

  const loaded = JSON.parse(fs.readFileSync(progressPath, 'utf-8'));
  assert.equal(loaded.done.length, 2);
  assert.equal(loaded.failed, 2);
  assert.ok(loaded.done.includes('600519'));

  cleanupTempDir();
});

test('断点续传：done 中的股票被过滤', () => {
  const stockList = [
    { code: '600519', market: '1', name: '茅台' },
    { code: '000001', market: '0', name: '平安' },
    { code: '600036', market: '1', name: '招商' },
  ];
  const done = new Set(['600519', '000001']);
  const pending = stockList.filter((s) => !done.has(s.code));
  assert.equal(pending.length, 1);
  assert.equal(pending[0].code, '600036');
});

test('并发批次正确分组', () => {
  const stockList = Array.from({ length: 12 }, (_, i) => ({ code: String(i) }));
  const concurrency = 5;
  const batches = [];
  for (let i = 0; i < stockList.length; i += concurrency) {
    batches.push(stockList.slice(i, i + concurrency));
  }
  assert.equal(batches.length, 3);
  assert.equal(batches[0].length, 5);
  assert.equal(batches[1].length, 5);
  assert.equal(batches[2].length, 2);
});

test('空股票清单不报错', () => {
  const stockList = [];
  const done = new Set();
  const pending = stockList.filter((s) => !done.has(s.code));
  assert.equal(pending.length, 0);
});

test('实际写入 SQLite 验证', () => {
  const db = new Database(':memory:');
  initSchema(db);

  const mockKlines = [
    { date: '2026-03', open: 100, close: 110, high: 115, low: 95, volume: 1e6, amount: 1e8, amplitude: 20, changePercent: 10, change: 10, turnoverRate: 0.5 },
    { date: '2026-04', open: 110, close: 120, high: 125, low: 105, volume: 1.2e6, amount: 1.3e8, amplitude: 18, changePercent: 9, change: 10, turnoverRate: 0.6 },
  ];

  saveKlinesSync(db, 'monthly_klines', '600519', '1', '贵州茅台', mockKlines);
  saveKlinesSync(db, 'weekly_klines', '600519', '1', '贵州茅台', mockKlines);

  // 验证 stocks
  const stock = db.prepare('SELECT * FROM stocks WHERE code = ?').get('600519');
  assert.equal(stock.name, '贵州茅台');
  assert.equal(stock.market, '1');

  // 验证 klines 分开存储
  const monthlyCount = db.prepare('SELECT count(*) as c FROM monthly_klines').get();
  const weeklyCount = db.prepare('SELECT count(*) as c FROM weekly_klines').get();
  assert.equal(monthlyCount.c, 2);
  assert.equal(weeklyCount.c, 2);

  // 验证字段映射（kline 对象 → SQLite 列）
  const row = db.prepare('SELECT * FROM monthly_klines WHERE date = ?').get('2026-03');
  assert.equal(row.open, 100);
  assert.equal(row.close, 110);
  assert.equal(row.amplitude, 20);
  assert.equal(row.change_percent, 10);
  assert.equal(row.change_amount, 10);
  assert.equal(row.turnover_rate, 0.5);
  assert.equal(row.adjust, 1.0);
});
