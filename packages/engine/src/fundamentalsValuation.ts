/**
 * Fast Graphs–style valuation: Normal P/E is median historical price/metric;
 * Fair Value uses PE 15 or Peter Lynch PEG=1 from 5-year EPS CAGR.
 * Pure math — no I/O. Data comes from FMP (or any provider) via the API layer.
 */

export type ValuationMetric = 'eps' | 'revenue' | 'fcf' | 'ownerEarnings';

export type FairValueRule = 'pe15' | 'lynch_peg' | 'none';

/** Chart / Normal P/E window. `null` = MAX (all complete years). */
export type ValuationWindowYears = 5 | 10 | null;

/** Below this 5y CAGR, fair value uses a 15× multiple instead of PEG=1. */
export const GROWTH_PE_FLOOR = 15;

export type AnnualFundamentalPoint = {
  /** Fiscal year end date YYYY-MM-DD */
  date: string;
  year: number;
  /** Year-end (or nearest) adjusted close */
  price: number | null;
  eps: number | null;
  revenuePerShare: number | null;
  fcfPerShare: number | null;
  ownerEarningsPerShare: number | null;
  pe: number | null;
  revenue: number | null;
  netIncome: number | null;
  operatingCashFlow: number | null;
  freeCashFlow: number | null;
  dividend?: number | null;
  /** Diluted weighted average shares from the filing (ordinary, not ADS). */
  dilutedShares?: number | null;
};

/**
 * Forward analyst estimate for the selected metric. Fast Graphs reads its Growth Rate off the
 * whole displayed window, estimates included, so these years drive both the ratio and the anchor.
 */
export type ForwardMetricPoint = {
  year: number;
  metric: number | null;
};

export type ValuationSeriesPoint = {
  date: string;
  year: number;
  price: number | null;
  /** Selected per-share metric */
  metric: number | null;
  /**
   * Metric × fair-value ratio (or Normal P/E if growth is N/A) — green earnings
   * on the price axis.
   */
  earningsPower: number | null;
  /** Metric × fair-value ratio (15× or Lynch). Absent when growth is N/A. */
  fairValue: number | null;
  /** Metric × Normal P/E (median historical price/metric). */
  normalValue: number | null;
  pe: number | null;
  estimated?: boolean;
};

export type ValuationSummary = {
  metric: ValuationMetric;
  /** Median historical price/metric — Normal P/E, not fair value. */
  normalMultiple: number;
  normalMultipleSource: 'median_pe' | 'fallback';
  currentPrice: number | null;
  latestMetric: number | null;
  fairValue: number | null;
  /** Metric the fair value is anchored on: first forward estimate, else `latestMetric`. */
  fairValueAnchor: number | null;
  fairValueAnchorYear: number | null;
  /** (price − fairValue) / fairValue */
  premiumPct: number | null;
  currentPe: number | null;
  /** Full-span CAGR of the selected metric. */
  metricCagrPct: number | null;
  /** CAGR behind the PE15 / Lynch rule (positive years only). */
  growthRatePct: number | null;
  /** `forward` spans the window through the last estimate; `trailing` is history only. */
  growthSource: 'trailing' | 'forward';
  fairValueRatio: number | null;
  fairValueRule: FairValueRule;
  years: number;
  /** 5, 10, or null for MAX. */
  windowYears: ValuationWindowYears;
};

function finite(n: unknown): n is number {
  return typeof n === 'number' && Number.isFinite(n);
}

function median(vals: number[]): number | null {
  if (!vals.length) return null;
  const s = [...vals].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

export function pickMetric(point: AnnualFundamentalPoint, metric: ValuationMetric): number | null {
  switch (metric) {
    case 'eps':
      return finite(point.eps) ? point.eps : null;
    case 'revenue':
      return finite(point.revenuePerShare) ? point.revenuePerShare : null;
    case 'fcf':
      return finite(point.fcfPerShare) ? point.fcfPerShare : null;
    case 'ownerEarnings':
      return finite(point.ownerEarningsPerShare) ? point.ownerEarningsPerShare : null;
    default:
      return null;
  }
}

/**
 * Normal multiple: median of historical price/metric for years with positive metric.
 * Falls back to 15 for EPS-like metrics and 3 for sales (P/S-style).
 */
export function computeNormalMultiple(
  points: AnnualFundamentalPoint[],
  metric: ValuationMetric,
): { multiple: number; source: 'median_pe' | 'fallback' } {
  const ratios: number[] = [];
  for (const p of points) {
    const m = pickMetric(p, metric);
    if (!finite(p.price) || !finite(m) || m <= 0 || p.price <= 0) continue;
    const r = p.price / m;
    // Guard absurd multiples from near-zero EPS years.
    if (r > 0 && r < 120) ratios.push(r);
  }
  const med = median(ratios);
  if (med != null && med > 0) return { multiple: roundMultiple(med), source: 'median_pe' };
  const fallback = metric === 'revenue' ? 3 : 15;
  return { multiple: fallback, source: 'fallback' };
}

/** 1 decimal at ≥1× so 6.29 → 6.3; extra digits below 1 so 0.28 does not become 0.0. */
export function roundMultiple(n: number): number {
  if (!Number.isFinite(n) || n <= 0) return n;
  if (n >= 1) return Math.round(n * 10) / 10;
  if (n >= 0.1) return Math.round(n * 100) / 100;
  return Math.round(n * 1000) / 1000;
}

/** Drop fiscal years whose period-end date is still in the future (stubs / current FY). */
export function completeFiscalYears(
  points: AnnualFundamentalPoint[],
  asOfIso = new Date().toISOString().slice(0, 10),
): AnnualFundamentalPoint[] {
  const byYear = new Map<number, AnnualFundamentalPoint>();
  for (const p of points) {
    if (p.date.slice(0, 10) > asOfIso) continue;
    const prev = byYear.get(p.year);
    if (!prev || p.date > prev.date) byYear.set(p.year, p);
  }
  return [...byYear.values()].sort((a, b) => a.year - b.year);
}

/**
 * Keep enough complete FYs for a true `windowYears` span ending at the latest FY.
 * A 5Y window means lastYear − 5 … lastYear (6 points, 5-year CAGR), not 5 points / 4 years.
 * `null` keeps the full history (MAX).
 */
export function sliceToWindow(
  points: AnnualFundamentalPoint[],
  windowYears: ValuationWindowYears,
): AnnualFundamentalPoint[] {
  const sorted = completeFiscalYears(points);
  if (windowYears == null || windowYears <= 0 || !sorted.length) return sorted;
  const lastYear = sorted[sorted.length - 1]!.year;
  const minYear = lastYear - windowYears;
  return sorted.filter((p) => p.year >= minYear);
}

export function cagrPct(first: number, last: number, years: number): number | null {
  if (!finite(first) || !finite(last) || years <= 0 || first <= 0 || last <= 0) return null;
  return (Math.pow(last / first, 1 / years) - 1) * 100;
}

/**
 * 5-year (or `lookbackYears`) CAGR using the last positive point and the
 * positive point at or before `last.year - lookbackYears`. Years with metric ≤ 0
 * are skipped. Needs at least a 2-year span.
 */
export function trailingMetricCagr(
  points: AnnualFundamentalPoint[],
  metric: ValuationMetric,
  lookbackYears = 5,
): number | null {
  const withM = [...points]
    .sort((a, b) => a.year - b.year)
    .filter((p) => {
      const m = pickMetric(p, metric);
      return finite(m) && m > 0;
    });
  if (withM.length < 2) return null;
  const last = withM[withM.length - 1]!;
  const targetYear = last.year - lookbackYears;
  let first = withM[0]!;
  for (const p of withM) {
    if (p.year <= targetYear) first = p;
  }
  const span = last.year - first.year;
  if (span < 2) return null;
  const a = pickMetric(first, metric);
  const b = pickMetric(last, metric);
  if (!finite(a) || !finite(b)) return null;
  return cagrPct(a, b, span);
}

/**
 * Fast Graphs–style growth: base is the earliest positive year still in the window, end is the
 * last positive analyst estimate. A trough-year base inflates this the same way it does in FG —
 * the rate is a property of the displayed window, not of the trailing history alone.
 */
export function forwardMetricCagr(
  windowed: AnnualFundamentalPoint[],
  forward: ForwardMetricPoint[],
  metric: ValuationMetric,
): number | null {
  const base = [...windowed]
    .sort((a, b) => a.year - b.year)
    .find((p) => {
      const m = pickMetric(p, metric);
      return finite(m) && m > 0;
    });
  if (!base) return null;
  const end = [...forward]
    .sort((a, b) => a.year - b.year)
    .filter((p) => finite(p.metric) && (p.metric as number) > 0)
    .pop();
  if (!end) return null;
  const span = end.year - base.year;
  if (span < 2) return null;
  const a = pickMetric(base, metric);
  if (!finite(a)) return null;
  return cagrPct(a, end.metric as number, span);
}

/** Earliest positive forward estimate — what Fast Graphs anchors its fair value line on. */
function firstForwardPositive(forward: ForwardMetricPoint[]): ForwardMetricPoint | null {
  return (
    [...forward]
      .sort((a, b) => a.year - b.year)
      .find((p) => finite(p.metric) && (p.metric as number) > 0) ?? null
  );
}

/**
 * Fair Value Ratio: growth < 15% → 15×; growth ≥ 15% → PEG=1 (ratio = growth %).
 */
export function fairValueRatioFromGrowth(growthPct: number | null): {
  ratio: number | null;
  rule: FairValueRule;
} {
  if (growthPct == null || !finite(growthPct)) return { ratio: null, rule: 'none' };
  if (growthPct < GROWTH_PE_FLOOR) return { ratio: GROWTH_PE_FLOOR, rule: 'pe15' };
  return { ratio: Math.round(growthPct * 100) / 100, rule: 'lynch_peg' };
}

export function yoyChgPct(curr: number | null | undefined, prev: number | null | undefined): number | null {
  if (!finite(curr) || !finite(prev) || prev === 0) return null;
  return ((curr - prev) / Math.abs(prev)) * 100;
}

/** Price CAGR over `years` looking back from the last bar. */
export function annualizedPriceReturnPct(
  bars: Array<{ date: string; close: number }>,
  years: number,
): number | null {
  if (!bars.length || years <= 0) return null;
  const last = bars[bars.length - 1]!;
  const lastT = Date.parse(last.date);
  if (!Number.isFinite(lastT) || last.close <= 0) return null;
  const target = lastT - years * 365.25 * 86_400_000;
  let first = bars[0]!;
  for (const b of bars) {
    const t = Date.parse(b.date);
    if (!Number.isFinite(t)) continue;
    if (t <= target) first = b;
    else break;
  }
  const spanYears = (lastT - Date.parse(first.date)) / (365.25 * 86_400_000);
  if (!Number.isFinite(spanYears) || spanYears < years * 0.75 || first.close <= 0) return null;
  return cagrPct(first.close, last.close, spanYears);
}

export function buildValuationSeries(
  points: AnnualFundamentalPoint[],
  metric: ValuationMetric,
  opts: {
    normalMultiple?: number;
    currentPrice?: number | null;
    /** Visible years for the chart and Normal P/E. Default MAX (all complete years). */
    windowYears?: ValuationWindowYears;
    /** Analyst estimates for the same metric. Present → FG-style forward growth and anchor. */
    forward?: ForwardMetricPoint[];
    /**
     * Trailing-twelve-month metric (EPS for the EPS view). Used as the fair-value anchor when
     * there is no forward estimate — closer to “current earnings” than the last closed FY.
     */
    ttmMetric?: number | null;
  } = {},
): { series: ValuationSeriesPoint[]; summary: ValuationSummary } {
  const windowYears = opts.windowYears === undefined ? null : opts.windowYears;
  const sorted = sliceToWindow(points, windowYears);
  const { multiple, source } =
    opts.normalMultiple != null && opts.normalMultiple > 0
      ? { multiple: opts.normalMultiple, source: 'median_pe' as const }
      : computeNormalMultiple(sorted, metric);

  const forward = opts.forward ?? [];
  const forwardGrowth = forwardMetricCagr(sorted, forward, metric);
  const growthSource: 'trailing' | 'forward' = forwardGrowth != null ? 'forward' : 'trailing';
  const growthRatePct = forwardGrowth ?? trailingMetricCagr(sorted, metric, 5);
  const { ratio: fairValueRatio, rule: fairValueRule } = fairValueRatioFromGrowth(growthRatePct);
  const earningsScale = fairValueRatio ?? multiple;

  const series: ValuationSeriesPoint[] = sorted.map((p) => {
    const m = pickMetric(p, metric);
    const positive = finite(m) && m > 0;
    const earningsPower = positive ? m * earningsScale : null;
    const fairValue = positive && fairValueRatio != null ? m * fairValueRatio : null;
    const normalValue = positive ? m * multiple : null;
    return {
      date: p.date,
      year: p.year,
      price: finite(p.price) ? p.price : null,
      metric: m,
      earningsPower,
      fairValue,
      normalValue,
      pe: finite(p.pe) ? p.pe : finite(p.price) && positive ? p.price / m : null,
      estimated: false,
    };
  });

  const withMetric = series.filter((s) => finite(s.metric) && (s.metric as number) > 0);
  const first = withMetric[0];
  const last = withMetric[withMetric.length - 1];
  const yearsSpan =
    first && last && last.year > first.year ? last.year - first.year : withMetric.length - 1;
  const metricCagrPct =
    first && last && finite(first.metric) && finite(last.metric)
      ? cagrPct(first.metric, last.metric, Math.max(1, yearsSpan))
      : null;

  const latestMetric = last?.metric ?? null;
  // Anchor order: first forward estimate (FG) → TTM (microcaps without coverage) → last FY.
  const anchorPoint = firstForwardPositive(forward);
  const ttm =
    opts.ttmMetric != null && finite(opts.ttmMetric) && opts.ttmMetric > 0 ? opts.ttmMetric : null;
  const fairValueAnchor = anchorPoint?.metric ?? ttm ?? latestMetric;
  const fairValueAnchorYear = anchorPoint?.year ?? (ttm != null ? null : last?.year ?? null);
  const fairValue =
    finite(fairValueAnchor) && fairValueAnchor > 0 && fairValueRatio != null
      ? fairValueAnchor * fairValueRatio
      : null;
  const currentPrice =
    opts.currentPrice != null && finite(opts.currentPrice)
      ? opts.currentPrice
      : last?.price ?? null;
  const premiumPct =
    finite(currentPrice) && finite(fairValue) && fairValue > 0
      ? ((currentPrice - fairValue) / fairValue) * 100
      : null;

  const currentPe =
    metric === 'eps' && finite(currentPrice) && finite(latestMetric) && latestMetric > 0
      ? currentPrice / latestMetric
      : last?.pe ?? null;

  return {
    series,
    summary: {
      metric,
      normalMultiple: multiple,
      normalMultipleSource: source,
      currentPrice,
      latestMetric,
      fairValue,
      fairValueAnchor: finite(fairValueAnchor) ? fairValueAnchor : null,
      fairValueAnchorYear,
      premiumPct,
      currentPe,
      metricCagrPct,
      growthRatePct,
      growthSource,
      fairValueRatio,
      fairValueRule,
      years: withMetric.length,
      windowYears,
    },
  };
}
