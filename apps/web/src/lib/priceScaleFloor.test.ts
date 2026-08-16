import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  PRICE_FLOOR,
  PRICE_SCALE_MARGIN_TOP,
  PRICE_SCALE_MARGINS_NORMAL,
  applyPriceFloorToAutoscaleInfo,
  autoscaleWithPriceFloor,
  bakeBottomPadding,
  clampVisiblePriceRange,
  projectedVisibleMin,
} from './priceScaleFloor.ts';

describe('projectedVisibleMin', () => {
  it('subtracts bottom margin as a fraction of the data span', () => {
    const min = projectedVisibleMin({ minValue: 50, maxValue: 100 }, PRICE_SCALE_MARGINS_NORMAL);
    assert.ok(min < 50);
    assert.ok(min > 0);
  });
});

describe('autoscaleWithPriceFloor', () => {
  it('keeps normal margins when padding would stay above 0', () => {
    const result = autoscaleWithPriceFloor({ minValue: 50, maxValue: 100 });
    assert.ok(result);
    assert.equal(result.scaleMargins.top, PRICE_SCALE_MARGINS_NORMAL.top);
    assert.equal(result.scaleMargins.bottom, PRICE_SCALE_MARGINS_NORMAL.bottom);
    assert.equal(result.priceRange.minValue, 50);
    assert.equal(result.priceRange.maxValue, 100);
    assert.ok(projectedVisibleMin(result.priceRange, result.scaleMargins) >= PRICE_FLOOR);
  });

  it('sits on 0 and drops bottom margin when padding would cross 0', () => {
    const data = { minValue: 10, maxValue: 200 };
    assert.ok(projectedVisibleMin(data, PRICE_SCALE_MARGINS_NORMAL) < PRICE_FLOOR);
    const result = autoscaleWithPriceFloor(data);
    assert.ok(result);
    assert.equal(result.priceRange.minValue, PRICE_FLOOR);
    assert.equal(result.priceRange.maxValue, 200);
    assert.equal(result.scaleMargins.top, PRICE_SCALE_MARGIN_TOP);
    assert.equal(result.scaleMargins.bottom, 0);
    assert.equal(projectedVisibleMin(result.priceRange, result.scaleMargins), PRICE_FLOOR);
  });

  it('does not raise minValue above the data minimum', () => {
    const result = autoscaleWithPriceFloor({ minValue: 10, maxValue: 200 });
    assert.ok(result);
    assert.ok(result.priceRange.minValue <= 10);
  });

  it('floors when data min is already 0', () => {
    const result = autoscaleWithPriceFloor({ minValue: 0, maxValue: 100 });
    assert.ok(result);
    assert.equal(result.priceRange.minValue, PRICE_FLOOR);
    assert.equal(result.scaleMargins.bottom, 0);
  });

  it('floors when all values are 0', () => {
    const result = autoscaleWithPriceFloor({ minValue: 0, maxValue: 0 });
    assert.ok(result);
    assert.equal(result.priceRange.minValue, PRICE_FLOOR);
    assert.equal(result.priceRange.maxValue, PRICE_FLOOR);
    assert.equal(result.scaleMargins.bottom, 0);
  });

  it('clamps a negative data min up to the floor', () => {
    const result = autoscaleWithPriceFloor({ minValue: -8, maxValue: 40 });
    assert.ok(result);
    assert.equal(result.priceRange.minValue, PRICE_FLOOR);
    assert.equal(result.scaleMargins.bottom, 0);
  });
});

describe('bakeBottomPadding', () => {
  it('encodes far-from-zero bottom padding into minValue for bottom:0 charts', () => {
    const result = autoscaleWithPriceFloor({ minValue: 50, maxValue: 100 });
    assert.ok(result);
    const baked = bakeBottomPadding(result);
    assert.equal(baked.maxValue, 100);
    assert.equal(baked.minValue, projectedVisibleMin(result.priceRange, result.scaleMargins));
    assert.ok(baked.minValue < 50);
    assert.ok(baked.minValue >= PRICE_FLOOR);
  });

  it('leaves a floored range at 0', () => {
    const result = autoscaleWithPriceFloor({ minValue: 10, maxValue: 200 });
    assert.ok(result);
    assert.deepEqual(bakeBottomPadding(result), { minValue: 0, maxValue: 200 });
  });
});

describe('applyPriceFloorToAutoscaleInfo', () => {
  it('passes through null / missing priceRange', () => {
    assert.equal(applyPriceFloorToAutoscaleInfo(null), null);
    const empty = { priceRange: null };
    assert.equal(applyPriceFloorToAutoscaleInfo(empty), empty);
  });

  it('returns a baked range that never projects below 0 with chart margins', () => {
    const info = applyPriceFloorToAutoscaleInfo({
      priceRange: { minValue: 10, maxValue: 200 },
    });
    assert.ok(info?.priceRange);
    assert.equal(info.priceRange.minValue, PRICE_FLOOR);
    assert.ok(
      projectedVisibleMin(info.priceRange, { top: PRICE_SCALE_MARGIN_TOP, bottom: 0 }) >=
        PRICE_FLOOR,
    );
  });
});

describe('clampVisiblePriceRange', () => {
  it('cuts the negative side without shifting the top', () => {
    assert.deepEqual(clampVisiblePriceRange({ from: -12, to: 80 }), { from: 0, to: 80 });
  });

  it('leaves a range already above the floor', () => {
    const range = { from: 40, to: 90 };
    assert.equal(clampVisiblePriceRange(range), range);
  });

  it('returns null for a missing range', () => {
    assert.equal(clampVisiblePriceRange(null), null);
  });
});
