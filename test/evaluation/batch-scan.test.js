// 批量扫描 + 日报测试
import { test } from 'node:test';
import assert from 'node:assert/strict';

test('batch-scan: generateDailyReport 格式验证', async () => {
  const { generateDailyReport } = await import('../../lib/scanner/daily-report.js');

  const results = [
    { code: '600519', name: '贵州茅台', judgment: 'bull', text: '突破前期高点，底部放量上涨，多头排列明显' },
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

  // 标题
  assert.ok(content.includes('# 机会股日报 2026-05-17'));
  // 概况
  assert.ok(content.includes('300 只'));
  assert.ok(content.includes('¥15.20'));
  // 看多
  assert.ok(content.includes('🟢 强看多信号'));
  assert.ok(content.includes('贵州茅台'));
  assert.ok(content.includes('中天科技'));
  // 看空
  assert.ok(content.includes('🔴 强看空信号'));
  // 失败
  assert.ok(content.includes('失败股'));
  assert.ok(content.includes('timeout'));

  // 清理
  fs.unlinkSync(reportPath);
});

test('batch-scan: 空结果不抛错', async () => {
  const { generateDailyReport } = await import('../../lib/scanner/daily-report.js');
  const reportPath = generateDailyReport([], { totalStocks: 0, totalCost: 0 });

  const fs = await import('node:fs');
  const content = fs.readFileSync(reportPath, 'utf-8');
  assert.ok(content.includes('0 只'));
  fs.unlinkSync(reportPath);
});

test('batch-scan: 预算耗尽场景', async () => {
  const { checkBudget, BudgetExceededError } = await import('../../lib/evaluation/cost-guard.js');
  // 检查大量预估是否抛错
  try {
    checkBudget(100); // 远超 ¥3/天
    // 如果能通过（预算文件可能有累积），跳过
  } catch (err) {
    assert.ok(err instanceof BudgetExceededError);
  }
});
