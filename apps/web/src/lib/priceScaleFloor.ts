/** Pin Fundamentals Y-axis so visible min is always 0 at the bottom of the chart. */

export const PRICE_FLOOR = 0;
export const PRICE_SCALE_MARGIN_TOP = 0.1;
export const PRICE_SCALE_MARGIN_BOTTOM = 0;

export const PRICE_SCALE_MARGINS_NORMAL = {
  top: PRICE_SCALE_MARGIN_TOP,
  bottom: PRICE_SCALE_MARGIN_BOTTOM,
} as const;

/** Chart-applied margins: bottom is 0 so floor sits on the plot edge. */
export const PRICE_SCALE_MARGINS_CHART = {
  top: PRICE_SCALE_MARGIN_TOP,
  bottom: 0,
} as const;

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
 * Always pin minValue to `floor` so 0 sits on the bottom of the chart box.
 * Top margin stays for headroom above the data max.
 */
export function autoscaleWithPriceFloor(
  range: PriceRange | null,
  floor = PRICE_FLOOR,
): PriceFloorAutoscale | null {
  if (!range) return null;
  const normalized = normalizeRange(range, floor);
  if (!normalized) return null;

  return {
    priceRange: { minValue: floor, maxValue: normalized.maxValue },
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
): AutoscaleInfoLike | null {
  if (!info?.priceRange) return info;
  const result = autoscaleWithPriceFloor(info.priceRange, floor);
  if (!result) return info;
  return {
    ...info,
    priceRange: bakeBottomPadding(result),
  };
}

/** Always pin `from` to the floor; stretch only moves the top. */
export function clampVisiblePriceRange(
  range: { from: number; to: number } | null,
  floor = PRICE_FLOOR,
): { from: number; to: number } | null {
  if (!range) return null;
  return { from: floor, to: Math.max(range.to, floor) };
}
