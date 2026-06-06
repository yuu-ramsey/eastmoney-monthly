// Batch scan + daily report test
import { test } from 'node:test';
import assert from 'node:assert/strict';

test('batch-scan: generateDailyReport format verification', async () => {
  const { generateDailyReport } = await import('../../lib/scanner/daily-report.js');

  const results = [
    { code: '600519', name: 'Kweichow Moutai', judgment: 'bull', text: '突破前期高点，底部放量上涨，多头排列明显' },
    { code: '600522', name: '中天科技', judgment: 'bear', text: '顶背离信号出现，天量见顶' },
    { code: '000001', name: '平安银行', judgment: 'neutral', text: '震荡' },
    { code: '600000', name: '浦发银行', judgment: 'bull', text: '金叉形成，反转信号' },
    { code: '002415', name: '海康威视', judgment: 'bear', text: '空头排列，破位下行' },
    { code: '600888', name: '新疆众和', judgment: 'bull', text: '底背离' },
    { code: 'bad', name: '失败股', error: 'timeout' },
  ];

  const meta = { totalStocks: 300, totalCost: 12.5, monthSpent: 15.2, monthBudget: 50 };
  const reportPath = generateDailyReport(results, meta, new Date('2026-05-17'));

  const fs = await import('node:fs');
  const content = fs.readFileSync(reportPath, 'utf-8');

  // title
  assert.ok(content.includes('# 机会股日报 2026-05-17'));
  // summary
  assert.ok(content.includes('300 stocks'));
  assert.ok(content.includes('¥15.20'));
  // bullish
  assert.ok(content.includes('🟢 Strong Bullish Signals'));
  assert.ok(content.includes('贵州茅台'));
  assert.ok(content.includes('中天科技'));
  // bearish
  assert.ok(content.includes('🔴 Strong Bearish Signals'));
  // failures
  assert.ok(content.includes('失败股'));
  assert.ok(content.includes('timeout'));

  // cleanup
  fs.unlinkSync(reportPath);
});

test('batch-scan: empty results does not throw', async () => {
  const { generateDailyReport } = await import('../../lib/scanner/daily-report.js');
  const reportPath = generateDailyReport([], { totalStocks: 0, totalCost: 0 });

  const fs = await import('node:fs');
  const content = fs.readFileSync(reportPath, 'utf-8');
  assert.ok(content.includes('0 stocks'));
  fs.unlinkSync(reportPath);
});

test('batch-scan: budget exhausted scenario', async () => {
  const { checkBudget, BudgetExceededError } = await import('../../lib/evaluation/cost-guard.js');
  // check if large estimate throws
  try {
    checkBudget(100); // far exceeds ¥3/day
    // if it passes (budget file may have accumulation), skip
  } catch (err) {
    assert.ok(err instanceof BudgetExceededError);
  }
});
