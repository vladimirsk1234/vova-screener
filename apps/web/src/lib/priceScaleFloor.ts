/** One-sided Y floor for Fundamentals: auto-scale and zoom must not go below 0. */

export const PRICE_FLOOR = 0;
export const PRICE_SCALE_MARGIN_TOP = 0.1;
export const PRICE_SCALE_MARGIN_BOTTOM = 0.08;

export const PRICE_SCALE_MARGINS_NORMAL = {
  top: PRICE_SCALE_MARGIN_TOP,
  bottom: PRICE_SCALE_MARGIN_BOTTOM,
} as const;

/** Chart-applied margins: bottom padding is baked into minValue when not floored. */
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
 * Decide priceRange + scaleMargins so the visible axis never crosses `floor`.
 * Does not raise minValue above the data min (except clamping negatives up to the floor).
 */
export function autoscaleWithPriceFloor(
  range: PriceRange | null,
  floor = PRICE_FLOOR,
): PriceFloorAutoscale | null {
  if (!range) return null;
  const normalized = normalizeRange(range, floor);
  if (!normalized) return null;

  const projected = projectedVisibleMin(normalized, PRICE_SCALE_MARGINS_NORMAL);
  if (normalized.minValue <= floor || projected < floor) {
    return {
      priceRange: { minValue: floor, maxValue: normalized.maxValue },
      scaleMargins: { top: PRICE_SCALE_MARGIN_TOP, bottom: 0 },
    };
  }
  return {
    priceRange: normalized,
    scaleMargins: { ...PRICE_SCALE_MARGINS_NORMAL },
  };
}

/**
 * Encode bottom padding into minValue so the chart can keep `scaleMargins.bottom = 0`
 * (avoids margin-induced negatives and per-series margin oscillation).
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

export function clampVisiblePriceRange(
  range: { from: number; to: number } | null,
  floor = PRICE_FLOOR,
): { from: number; to: number } | null {
  if (!range) return null;
  if (range.from >= floor) return range;
  return { from: floor, to: Math.max(range.to, floor) };
}
