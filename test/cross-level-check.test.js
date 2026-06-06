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

test('checkCrossLevelConsistency: parent=null returns empty warnings', () => {
  const result = checkCrossLevelConsistency(weekly, null);
  assert.deepEqual(result.warnings, []);
});

test('checkCrossLevelConsistency: normal central zone nesting produces no warning', () => {
  const result = checkCrossLevelConsistency(weekly, monthly);
  assert.equal(result.warnings.length, 0);
});

test('checkCrossLevelConsistency: zone nesting anomaly -- child zone exceeds upper bound', () => {
  const badWeekly = {
    ...weekly,
    centralZone: { lower: 1500.00, upper: 1800.00, exists: true },
  };
  const result = checkCrossLevelConsistency(badWeekly, monthly);
  assert.ok(result.warnings.length >= 1);
  assert.match(result.warnings[0], /中枢嵌套异常/);
  assert.match(result.warnings[0], /超出上级周期中枢/);
});

test('checkCrossLevelConsistency: zone nesting anomaly -- child zone exceeds lower bound', () => {
  const badWeekly = {
    ...weekly,
    centralZone: { lower: 1300.00, upper: 1600.00, exists: true },
  };
  const result = checkCrossLevelConsistency(badWeekly, monthly);
  assert.ok(result.warnings.length >= 1);
  assert.match(result.warnings[0], /中枢嵌套异常/);
});

test('checkCrossLevelConsistency: trend reversal conflict -- parent up, child reversal_top', () => {
  const reversalWeekly = { ...weekly, trend: 'reversal_top' };
  const result = checkCrossLevelConsistency(reversalWeekly, monthly);
  assert.ok(result.warnings.length >= 1);
  assert.match(result.warnings[0], /趋势冲突/);
  assert.match(result.warnings[0], /顶部反转/);
});

test('checkCrossLevelConsistency: trend reversal conflict -- parent down, child reversal_bottom', () => {
  const downMonthly = { ...monthly, trend: 'down' };
  const reversalWeekly = { ...weekly, trend: 'reversal_bottom' };
  const result = checkCrossLevelConsistency(reversalWeekly, downMonthly);
  assert.ok(result.warnings.length >= 1);
  assert.match(result.warnings[0], /趋势冲突/);
  assert.match(result.warnings[0], /底部反转/);
});

test('checkCrossLevelConsistency: non-conflicting trend does not trigger warning (up to sideways)', () => {
  const result = checkCrossLevelConsistency(weekly, monthly);
  // weekly=sideways, monthly=up -> no conflict
  assert.equal(result.warnings.length, 0);
});

test('checkCrossLevelConsistency: multiple warnings returned simultaneously', () => {
  const badWeekly = {
    ...weekly,
    centralZone: { lower: 1300.00, upper: 1800.00, exists: true }, // out of range
    trend: 'reversal_top', // trend conflict
  };
  const result = checkCrossLevelConsistency(badWeekly, monthly);
  assert.ok(result.warnings.length >= 2, `expected at least 2 warnings, got ${result.warnings.length}`);
  const hasZoneWarning = result.warnings.some((w) => /中枢嵌套异常/.test(w));
  const hasTrendWarning = result.warnings.some((w) => /趋势冲突/.test(w));
  assert.ok(hasZoneWarning, 'should contain central zone warning');
  assert.ok(hasTrendWarning, 'should contain trend warning');
});

test('checkCrossLevelConsistency: skips zone check when parent or current centralZone does not exist', () => {
  const noZoneCurrent = { ...weekly, centralZone: { lower: null, upper: null, exists: false } };
  let result = checkCrossLevelConsistency(noZoneCurrent, monthly);
  assert.equal(result.warnings.length, 0);

  const noZoneParent = { ...monthly, centralZone: { lower: null, upper: null, exists: false } };
  result = checkCrossLevelConsistency(weekly, noZoneParent);
  assert.equal(result.warnings.length, 0);
});

test('checkCrossLevelConsistency: 10% boundary breach (within tolerance) -> no alert', () => {
  // monthly zone [1400, 1700], height=300, tolerance=45
  // 10% breach = 30, within tolerance of 45
  const withinTolerance = {
    ...weekly,
    centralZone: { lower: 1370.00, upper: 1700.00, exists: true },
  };
  const result = checkCrossLevelConsistency(withinTolerance, monthly);
  assert.equal(result.warnings.length, 0, '10% breach should be within tolerance, no alert');
});

test('checkCrossLevelConsistency: 20% boundary breach (exceeds tolerance) -> alert', () => {
  // monthly zone [1400, 1700], height=300, tolerance=45
  // 20% breach = 60, exceeds tolerance of 45
  const beyondTolerance = {
    ...weekly,
    centralZone: { lower: 1340.00, upper: 1700.00, exists: true },
  };
  const result = checkCrossLevelConsistency(beyondTolerance, monthly);
  assert.ok(result.warnings.length >= 1, '20% breach should trigger alert');
  assert.match(result.warnings[0], /中枢嵌套异常/);
  assert.match(result.warnings[0], /15% 容差/);
});

test('checkCrossLevelConsistency: NaN price values are skipped without throwing exception', () => {
  const badZoneCurrent = {
    ...weekly,
    centralZone: { lower: 'not-a-number', upper: 1600.00, exists: true },
  };
  const result = checkCrossLevelConsistency(badZoneCurrent, monthly);
  // Price conversion fails -> skip, no exception thrown
  assert.equal(result.warnings.length, 0);
});
