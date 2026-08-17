/**
 * Fundamentals Y-axis policy: the visible minimum is always 0, sitting on the bottom
 * edge of the plot. Vertical stretch is expressed as a factor on the top of the range,
 * so the chart never needs to disable auto-scaling (which is what lets Lightweight
 * Charts scale around the range centre and produce negative prices).
 */

export const PRICE_FLOOR = 0;
export const PRICE_SCALE_MARGIN_TOP = 0.1;
export const PRICE_SCALE_MARGIN_BOTTOM = 0;

export const PRICE_SCALE_MARGINS_NORMAL = {
  top: PRICE_SCALE_MARGIN_TOP,
  bottom: PRICE_SCALE_MARGIN_BOTTOM,
} as const;

/** Chart-applied margins: bottom is 0 so the floor sits on the plot edge. */
export const PRICE_SCALE_MARGINS_CHART = {
  top: PRICE_SCALE_MARGIN_TOP,
  bottom: 0,
} as const;

export const DEFAULT_STRETCH = 1;
export const STRETCH_MIN = 0.2;
export const STRETCH_MAX = 8;

/** Keeps a degenerate 0..0 range from collapsing the scale. */
const MIN_SPAN = 1e-6;

export type PriceRange = { minValue: number; maxValue: number };
export type ScaleMargins = { top: number; bottom: number };

export type PriceFloorAutoscale = {
  priceRange: PriceRange;
  scaleMargins: ScaleMargins;
};

export type AutoscaleInfoLike = {
  priceRange: PriceRange | null;
  margins?: { above: number; below: number };
};

function dataFraction(margins: ScaleMargins): number {
  return Math.max(1e-9, 1 - margins.top - margins.bottom);
}

/** Visible min after Lightweight Charts applies `scaleMargins` to `priceRange`. */
export function projectedVisibleMin(range: PriceRange, margins: ScaleMargins): number {
  const k = margins.bottom / dataFraction(margins);
  return range.minValue - (range.maxValue - range.minValue) * k;
}

/** Clamp a stretch factor into the supported band; invalid input falls back to 1. */
export function clampStretchFactor(factor: number): number {
  if (!Number.isFinite(factor) || factor <= 0) return DEFAULT_STRETCH;
  return Math.min(STRETCH_MAX, Math.max(STRETCH_MIN, factor));
}

function normalizeRange(range: PriceRange, floor: number): PriceRange | null {
  let { minValue, maxValue } = range;
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) return null;
  if (maxValue < minValue) {
    const swap = minValue;
    minValue = maxValue;
    maxValue = swap;
  }
  return {
    minValue: Math.max(minValue, floor),
    maxValue: Math.max(maxValue, floor),
  };
}

/**
 * Pin minValue to `floor` and put the whole stretch on top, so 0 stays on the
 * bottom edge and only the ceiling moves.
 */
export function autoscaleWithPriceFloor(
  range: PriceRange | null,
  floor = PRICE_FLOOR,
  stretch = DEFAULT_STRETCH,
): PriceFloorAutoscale | null {
  if (!range) return null;
  const normalized = normalizeRange(range, floor);
  if (!normalized) return null;

  const factor = clampStretchFactor(stretch);
  const span = Math.max((normalized.maxValue - floor) * factor, MIN_SPAN);
  return {
    priceRange: { minValue: floor, maxValue: floor + span },
    scaleMargins: { top: PRICE_SCALE_MARGIN_TOP, bottom: 0 },
  };
}

/**
 * Encode bottom padding into minValue so the chart can keep `scaleMargins.bottom = 0`.
 * With the pinned-floor policy this is a no-op (bottom is already 0).
 */
export function bakeBottomPadding(result: PriceFloorAutoscale): PriceRange {
  if (result.scaleMargins.bottom === 0) return result.priceRange;
  return {
    minValue: projectedVisibleMin(result.priceRange, result.scaleMargins),
    maxValue: result.priceRange.maxValue,
  };
}

export function applyPriceFloorToAutoscaleInfo(
  info: AutoscaleInfoLike | null,
  floor = PRICE_FLOOR,
  stretch = DEFAULT_STRETCH,
): AutoscaleInfoLike | null {
  if (!info?.priceRange) return info;
  const result = autoscaleWithPriceFloor(info.priceRange, floor, stretch);
  if (!result) return info;
  return {
    ...info,
    priceRange: bakeBottomPadding(result),
    // Pixel margins would lift the floor off the bottom edge.
    margins: { above: info.margins?.above ?? 0, below: 0 },
  };
}
