// Cross-level consistency check test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { checkCrossLevelConsistency } from '../lib/cross-level-check.js';

const monthly = {
  period: 'monthly',
  centralZone: { lower: 1400.00, upper: 1700.00, exists: true },
  keySupport: [1400.00, 1350.00, 1300.00],
  keyResistance: [1700.00, 1750.00, 1800.00],
  trend: 'up',
};

const weekly = {
  period: 'weekly',
  centralZone: { lower: 1500.00, upper: 1650.00, exists: true },
  keySupport: [1500.00, 1450.00],
  keyResistance: [1650.00, 1700.00],
  trend: 'sideways',
};

test('checkCrossLevelConsistency: parent=null 返回空 warnings', () => {
  const result = checkCrossLevelConsistency(weekly, null);
  assert.deepEqual(result.warnings, []);
});

test('checkCrossLevelConsistency: 中枢嵌套正常时无 warning', () => {
  const result = checkCrossLevelConsistency(weekly, monthly);
  assert.equal(result.warnings.length, 0);
});

test('checkCrossLevelConsistency: 中枢嵌套异常——小级别中枢超上沿', () => {
  const badWeekly = {
    ...weekly,
    centralZone: { lower: 1500.00, upper: 1800.00, exists: true },
  };
  const result = checkCrossLevelConsistency(badWeekly, monthly);
  assert.ok(result.warnings.length >= 1);
  assert.match(result.warnings[0], /中枢嵌套异常/);
  assert.match(result.warnings[0], /超出上级周期中枢/);
});

test('checkCrossLevelConsistency: 中枢嵌套异常——小级别中枢超下沿', () => {
  const badWeekly = {
    ...weekly,
    centralZone: { lower: 1300.00, upper: 1600.00, exists: true },
  };
  const result = checkCrossLevelConsistency(badWeekly, monthly);
  assert.ok(result.warnings.length >= 1);
  assert.match(result.warnings[0], /中枢嵌套异常/);
});

test('checkCrossLevelConsistency: 趋势反转冲突——上级 up 下级 reversal_top', () => {
  const reversalWeekly = { ...weekly, trend: 'reversal_top' };
  const result = checkCrossLevelConsistency(reversalWeekly, monthly);
  assert.ok(result.warnings.length >= 1);
  assert.match(result.warnings[0], /趋势冲突/);
  assert.match(result.warnings[0], /顶部反转/);
});

test('checkCrossLevelConsistency: 趋势反转冲突——上级 down 下级 reversal_bottom', () => {
  const downMonthly = { ...monthly, trend: 'down' };
  const reversalWeekly = { ...weekly, trend: 'reversal_bottom' };
  const result = checkCrossLevelConsistency(reversalWeekly, downMonthly);
  assert.ok(result.warnings.length >= 1);
  assert.match(result.warnings[0], /趋势冲突/);
  assert.match(result.warnings[0], /底部反转/);
});

test('checkCrossLevelConsistency: 无冲突趋势不走警告（up→sideways）', () => {
  const result = checkCrossLevelConsistency(weekly, monthly);
  // weekly=sideways, monthly=up → 无冲突
  assert.equal(result.warnings.length, 0);
});

test('checkCrossLevelConsistency: 多个 warning 同时返回', () => {
  const badWeekly = {
    ...weekly,
    centralZone: { lower: 1300.00, upper: 1800.00, exists: true }, // 超范围
    trend: 'reversal_top', // 趋势冲突
  };
  const result = checkCrossLevelConsistency(badWeekly, monthly);
  assert.ok(result.warnings.length >= 2, `应有至少2个warning,实际${result.warnings.length}个`);
  const hasZoneWarning = result.warnings.some((w) => /中枢嵌套异常/.test(w));
  const hasTrendWarning = result.warnings.some((w) => /趋势冲突/.test(w));
  assert.ok(hasZoneWarning, '应包含中枢警告');
  assert.ok(hasTrendWarning, '应包含趋势警告');
});

test('checkCrossLevelConsistency: 上级周期或当前周期 centralZone 不存在时跳过中枢检查', () => {
  const noZoneCurrent = { ...weekly, centralZone: { lower: null, upper: null, exists: false } };
  let result = checkCrossLevelConsistency(noZoneCurrent, monthly);
  assert.equal(result.warnings.length, 0);

  const noZoneParent = { ...monthly, centralZone: { lower: null, upper: null, exists: false } };
  result = checkCrossLevelConsistency(weekly, noZoneParent);
  assert.equal(result.warnings.length, 0);
});

test('checkCrossLevelConsistency: 越界 10%（容差内）→ 不报警', () => {
  // monthly zone [1400, 1700], height=300, tolerance=45
  // 越界 10% = 30，在容差 45 内
  const withinTolerance = {
    ...weekly,
    centralZone: { lower: 1370.00, upper: 1700.00, exists: true },
  };
  const result = checkCrossLevelConsistency(withinTolerance, monthly);
  assert.equal(result.warnings.length, 0, '10%越界应在容差内，不报警');
});

test('checkCrossLevelConsistency: 越界 20%（超容差）→ 报警', () => {
  // monthly zone [1400, 1700], height=300, tolerance=45
  // 越界 20% = 60，超出容差 45
  const beyondTolerance = {
    ...weekly,
    centralZone: { lower: 1340.00, upper: 1700.00, exists: true },
  };
  const result = checkCrossLevelConsistency(beyondTolerance, monthly);
  assert.ok(result.warnings.length >= 1, '20%越界应报警');
  assert.match(result.warnings[0], /中枢嵌套异常/);
  assert.match(result.warnings[0], /15% 容差/);
});

test('checkCrossLevelConsistency: NaN 价位值被跳过不报异常', () => {
  const badZoneCurrent = {
    ...weekly,
    centralZone: { lower: 'not-a-number', upper: 1600.00, exists: true },
  };
  const result = checkCrossLevelConsistency(badZoneCurrent, monthly);
  // 价位转换失败→跳过，不抛异常
  assert.equal(result.warnings.length, 0);
});
