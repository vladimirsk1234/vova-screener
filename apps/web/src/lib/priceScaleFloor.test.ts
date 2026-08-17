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
  it('leaves min unchanged when bottom margin is 0', () => {
    const min = projectedVisibleMin({ minValue: 0, maxValue: 100 }, PRICE_SCALE_MARGINS_NORMAL);
    assert.equal(min, 0);
  });
});

describe('autoscaleWithPriceFloor', () => {
  it('pins minValue to 0 even when data is far above the floor', () => {
    const result = autoscaleWithPriceFloor({ minValue: 50, maxValue: 100 });
    assert.ok(result);
    assert.equal(result.priceRange.minValue, PRICE_FLOOR);
    assert.equal(result.priceRange.maxValue, 100);
    assert.equal(result.scaleMargins.top, PRICE_SCALE_MARGIN_TOP);
    assert.equal(result.scaleMargins.bottom, 0);
    assert.equal(projectedVisibleMin(result.priceRange, result.scaleMargins), PRICE_FLOOR);
  });

  it('pins minValue to 0 when padding would have crossed 0 under the old policy', () => {
    const result = autoscaleWithPriceFloor({ minValue: 10, maxValue: 200 });
    assert.ok(result);
    assert.equal(result.priceRange.minValue, PRICE_FLOOR);
    assert.equal(result.priceRange.maxValue, 200);
    assert.equal(result.scaleMargins.bottom, 0);
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
  it('leaves a floored range at 0 when bottom margin is 0', () => {
    const result = autoscaleWithPriceFloor({ minValue: 50, maxValue: 100 });
    assert.ok(result);
    assert.deepEqual(bakeBottomPadding(result), { minValue: 0, maxValue: 100 });
  });

  it('leaves a floored range at 0 for near-zero data', () => {
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

  it('returns a baked range pinned to 0 with chart margins', () => {
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

  it('pins from to 0 even when the range is already above the floor', () => {
    assert.deepEqual(clampVisiblePriceRange({ from: 40, to: 90 }), { from: 0, to: 90 });
  });

  it('returns null for a missing range', () => {
    assert.equal(clampVisiblePriceRange(null), null);
  });
});
