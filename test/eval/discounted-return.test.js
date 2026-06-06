// Discounted multi-month return unit test
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { computeDiscountedReturn, discretizeDiscountedReturn, mapSignalToNumber } from '../../lib/eval/discounted-return.js';

describe('computeDiscountedReturn', () => {
  // build monthly klines: prices 10, 11, 12, 13, 14 (monthly gain 10%/20%/30%/40%)
  const upKlines = [
    { close: 10 }, // idx 0
    { close: 11 }, // idx 1: +10%
    { close: 12 }, // idx 2: +20%
    { close: 13 }, // idx 3: +30%
    { close: 14 }, // idx 4: +40%
  ];

  it('full 3-month forward return', () => {
    const r = computeDiscountedReturn(upKlines, 0);
    // r1 = (11-10)/10 = 0.1, r2 = 0.2, r3 = 0.3
    // discounted = 0.1 + 0.9*0.2 + 0.81*0.3 = 0.1 + 0.18 + 0.243 = 0.523
    assert.strictEqual(r.monthsAvailable, 3);
    assert.ok(Math.abs(r.discountedReturn - 0.523) < 0.001);
    assert.strictEqual(r.individualReturns.length, 3);
  });

  it('only 2 months available (tail insufficient)', () => {
    const r = computeDiscountedReturn(upKlines, 2);
    // r1 = (13-12)/12 = 0.0833, r2 = (14-12)/12 = 0.1667
    // discounted = 0.0833 + 0.9*0.1667 = 0.2333
    assert.strictEqual(r.monthsAvailable, 2);
    assert.ok(Math.abs(r.discountedReturn - 0.2333) < 0.01);
  });

  it('gamma=1 equivalent to undiscounted cumsum', () => {
    const r = computeDiscountedReturn(upKlines, 0, { gamma: 1.0, months: 3 });
    // 0.1 + 0.2 + 0.3 = 0.6
    assert.ok(Math.abs(r.discountedReturn - 0.6) < 0.001);
  });

  it('downtrend sequence', () => {
    const downKlines = [
      { close: 10 }, { close: 9 }, { close: 8 }, { close: 7 }, { close: 6 },
    ];
    const r = computeDiscountedReturn(downKlines, 0);
    // r1=-0.1, r2=-0.2, r3=-0.3 -> -0.1 + 0.9*(-0.2) + 0.81*(-0.3) = -0.1 -0.18 -0.243 = -0.523
    assert.ok(r.discountedReturn < -0.5);
  });
});

describe('discretizeDiscountedReturn', () => {
  it('strong_bull boundary', () => {
    assert.strictEqual(discretizeDiscountedReturn(0.20), 'strong_bull');
    assert.strictEqual(discretizeDiscountedReturn(0.15), 'strong_bull');
  });
  it('bull boundary', () => {
    assert.strictEqual(discretizeDiscountedReturn(0.06), 'bull');
    assert.strictEqual(discretizeDiscountedReturn(0.10), 'bull');
  });
  it('neutral', () => {
    assert.strictEqual(discretizeDiscountedReturn(0.04), 'neutral');
    assert.strictEqual(discretizeDiscountedReturn(0.00), 'neutral');
    assert.strictEqual(discretizeDiscountedReturn(-0.04), 'neutral');
    assert.strictEqual(discretizeDiscountedReturn(-0.09), 'neutral');
  });
  it('bear', () => {
    assert.strictEqual(discretizeDiscountedReturn(-0.10), 'bear');
    assert.strictEqual(discretizeDiscountedReturn(-0.15), 'bear');
  });
  it('strong_bear', () => {
    assert.strictEqual(discretizeDiscountedReturn(-0.20), 'strong_bear');
    assert.strictEqual(discretizeDiscountedReturn(-0.30), 'strong_bear');
  });
});

describe('mapSignalToNumber', () => {
  it('mapping correct', () => {
    assert.strictEqual(mapSignalToNumber('strong_bull'), 2);
    assert.strictEqual(mapSignalToNumber('bull'), 1);
    assert.strictEqual(mapSignalToNumber('neutral'), 0);
    assert.strictEqual(mapSignalToNumber('bear'), -1);
    assert.strictEqual(mapSignalToNumber('strong_bear'), -2);
  });
});
