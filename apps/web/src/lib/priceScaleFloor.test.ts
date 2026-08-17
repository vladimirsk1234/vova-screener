import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_STRETCH,
  PRICE_FLOOR,
  PRICE_SCALE_MARGIN_TOP,
  PRICE_SCALE_MARGINS_NORMAL,
  STRETCH_MAX,
  STRETCH_MIN,
  applyPriceFloorToAutoscaleInfo,
  autoscaleWithPriceFloor,
  bakeBottomPadding,
  clampStretchFactor,
  projectedVisibleMin,
} from './priceScaleFloor.ts';

describe('projectedVisibleMin', () => {
  it('leaves min unchanged when bottom margin is 0', () => {
    const min = projectedVisibleMin({ minValue: 0, maxValue: 100 }, PRICE_SCALE_MARGINS_NORMAL);
    assert.equal(min, 0);
  });
});

describe('clampStretchFactor', () => {
  it('keeps a factor inside the band', () => {
    assert.equal(clampStretchFactor(2), 2);
  });

  it('clamps to the band edges', () => {
    assert.equal(clampStretchFactor(STRETCH_MIN / 10), STRETCH_MIN);
    assert.equal(clampStretchFactor(STRETCH_MAX * 10), STRETCH_MAX);
  });

  it('falls back to 1 for invalid input', () => {
    assert.equal(clampStretchFactor(Number.NaN), DEFAULT_STRETCH);
    assert.equal(clampStretchFactor(0), DEFAULT_STRETCH);
    assert.equal(clampStretchFactor(-3), DEFAULT_STRETCH);
    assert.equal(clampStretchFactor(Number.POSITIVE_INFINITY), DEFAULT_STRETCH);
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

  it('clamps a negative data min up to the floor', () => {
    const result = autoscaleWithPriceFloor({ minValue: -8, maxValue: 40 });
    assert.ok(result);
    assert.equal(result.priceRange.minValue, PRICE_FLOOR);
    assert.equal(result.priceRange.maxValue, 40);
  });

  it('keeps the floor at 0 when all values are 0', () => {
    const result = autoscaleWithPriceFloor({ minValue: 0, maxValue: 0 });
    assert.ok(result);
    assert.equal(result.priceRange.minValue, PRICE_FLOOR);
    assert.ok(result.priceRange.maxValue > PRICE_FLOOR);
  });

  it('swaps an inverted range before flooring', () => {
    const result = autoscaleWithPriceFloor({ minValue: 90, maxValue: 10 });
    assert.ok(result);
    assert.equal(result.priceRange.minValue, PRICE_FLOOR);
    assert.equal(result.priceRange.maxValue, 90);
  });

  it('returns null for a non-finite range', () => {
    assert.equal(autoscaleWithPriceFloor({ minValue: Number.NaN, maxValue: 10 }), null);
    assert.equal(autoscaleWithPriceFloor(null), null);
  });

  it('puts the whole stretch on top and never moves the floor', () => {
    const doubled = autoscaleWithPriceFloor({ minValue: 10, maxValue: 100 }, PRICE_FLOOR, 2);
    assert.ok(doubled);
    assert.equal(doubled.priceRange.minValue, PRICE_FLOOR);
    assert.equal(doubled.priceRange.maxValue, 200);

    const halved = autoscaleWithPriceFloor({ minValue: 10, maxValue: 100 }, PRICE_FLOOR, 0.5);
    assert.ok(halved);
    assert.equal(halved.priceRange.minValue, PRICE_FLOOR);
    assert.equal(halved.priceRange.maxValue, 50);
  });

  it('keeps the floor at 0 across the whole stretch band', () => {
    for (const stretch of [STRETCH_MIN, 0.5, 1, 3, STRETCH_MAX, 1000, -5, Number.NaN]) {
      const result = autoscaleWithPriceFloor({ minValue: -20, maxValue: 240 }, PRICE_FLOOR, stretch);
      assert.ok(result);
      assert.equal(result.priceRange.minValue, PRICE_FLOOR, `stretch=${stretch}`);
      assert.ok(result.priceRange.maxValue > PRICE_FLOOR, `stretch=${stretch}`);
      assert.equal(
        projectedVisibleMin(result.priceRange, result.scaleMargins),
        PRICE_FLOOR,
        `stretch=${stretch}`,
      );
    }
  });
});

describe('bakeBottomPadding', () => {
  it('leaves a floored range at 0', () => {
    const result = autoscaleWithPriceFloor({ minValue: 50, maxValue: 100 });
    assert.ok(result);
    assert.deepEqual(bakeBottomPadding(result), { minValue: 0, maxValue: 100 });
  });
});

describe('applyPriceFloorToAutoscaleInfo', () => {
  it('passes through null / missing priceRange', () => {
    assert.equal(applyPriceFloorToAutoscaleInfo(null), null);
    const empty = { priceRange: null };
    assert.equal(applyPriceFloorToAutoscaleInfo(empty), empty);
  });

  it('returns a range pinned to 0 with chart margins', () => {
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

  it('drops the pixel margin below so the floor stays on the bottom edge', () => {
    const info = applyPriceFloorToAutoscaleInfo({
      priceRange: { minValue: 10, maxValue: 200 },
      margins: { above: 12, below: 30 },
    });
    assert.ok(info?.margins);
    assert.equal(info.margins.below, 0);
    assert.equal(info.margins.above, 12);
  });

  it('applies the stretch to the top only', () => {
    const info = applyPriceFloorToAutoscaleInfo(
      { priceRange: { minValue: 10, maxValue: 120 } },
      PRICE_FLOOR,
      2,
    );
    assert.ok(info?.priceRange);
    assert.equal(info.priceRange.minValue, PRICE_FLOOR);
    assert.equal(info.priceRange.maxValue, 240);
  });
});
