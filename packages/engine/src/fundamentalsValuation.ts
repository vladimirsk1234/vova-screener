/**
 * Fast Graphs–style valuation: Normal P/E is median historical price/metric;
 * Fair Value uses PE 15 or Peter Lynch PEG=1 from trailing EPS CAGR in the
 * selected lookback window (1Y / 3Y / 5Y / 8Y / 10Y / MAX).
 * Pure math — no I/O. Data comes from FMP (or any provider) via the API layer.
 */

export type ValuationMetric = 'eps' | 'revenue' | 'fcf' | 'ownerEarnings';

export type FairValueRule = 'pe15' | 'lynch_peg' | 'none';

/** Chart / Normal P/E window. `null` = MAX (all complete years). */
export type ValuationWindowYears = 1 | 3 | 5 | 8 | 10 | null;

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
  /** Annual dividend per share (DPS) on the price axis, FAST Graphs–style. */
  dividend?: number | null;
  pe: number | null;
  estimated?: boolean;
  /** Forward analyst year — dashed chart segment, not the TTM today-point. */
  forecast?: boolean;
};

/** How many analyst years the Fundamentals chart projects as a dashed fair-value line. */
export const FORWARD_FAIR_VALUE_YEARS = 3;

/**
 * Skip a current-FY estimate when its FY-end is this close to the last solid
 * point so the dashed tail is the next year, not a 12-day wall.
 */
export const NEAR_FORECAST_FY_END_DAYS = 90;

export type QuarterlyMetricPoint = {
  date: string;
  eps?: number | null;
  metric?: number | null;
};

export type ForwardEstimatePoint = {
  year: number;
  date?: string;
  eps?: number | null;
  metric?: number | null;
};

export type ValuationSummary = {
  metric: ValuationMetric;
  /** Median historical price/metric — Normal P/E, not fair value. */
  normalMultiple: number;
  normalMultipleSource: 'median_pe' | 'fallback';
  currentPrice: number | null;
  latestMetric: number | null;
  fairValue: number | null;
  /** Metric the fair value is anchored on: TTM, else last complete FY. */
  fairValueAnchor: number | null;
  fairValueAnchorYear: number | null;
  /** (price − fairValue) / fairValue */
  premiumPct: number | null;
  currentPe: number | null;
  /** Full-span CAGR of the selected metric. */
  metricCagrPct: number | null;
  /** Trailing CAGR behind the PE15 / Lynch rule (positive years in the window only). */
  growthRatePct: number | null;
  /** Always trailing — analyst estimates do not set the growth rate. */
  growthSource: 'trailing' | 'forward';
  fairValueRatio: number | null;
  fairValueRule: FairValueRule;
  years: number;
  /** 1, 3, 5, 8, 10, or null for MAX. */
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
 * Window-span CAGR using the last positive point and the positive point at or
 * before `last.year - lookbackYears`. Years with metric ≤ 0 are skipped.
 * A 1-year window is a simple YoY; longer windows need at least that span.
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
  if (span < 1) return null;
  const a = pickMetric(first, metric);
  const b = pickMetric(last, metric);
  if (!finite(a) || !finite(b)) return null;
  return cagrPct(a, b, span);
}

/** Lookback for trailing CAGR: 1 / 3 / 5 / 8 / 10, or a long span for MAX. */
export function windowLookbackYears(windowYears: ValuationWindowYears): number {
  return windowYears ?? 1000;
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
    /** Kept for callers; estimates no longer set growth or the fair-value anchor. */
    forward?: ForwardMetricPoint[];
    /**
     * Trailing-twelve-month metric (EPS for the EPS view). Preferred fair-value
     * anchor — closer to “current earnings” than the last closed FY.
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

  const growthSource = 'trailing' as const;
  const growthRatePct = trailingMetricCagr(sorted, metric, windowLookbackYears(windowYears));
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
      dividend: finite(p.dividend) ? p.dividend : null,
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
  // Anchor: TTM (current earnings) → last complete FY. Estimates do not set today's FV.
  const ttm =
    opts.ttmMetric != null && finite(opts.ttmMetric) && opts.ttmMetric > 0 ? opts.ttmMetric : null;
  const fairValueAnchor = ttm ?? latestMetric;
  const fairValueAnchorYear = ttm != null ? null : last?.year ?? null;
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

/**
 * Chart series whose last fair-value point equals `summary.fairValue`.
 * Historical years stay `metric × ratio`. When `pinToday` is true (Sales /
 * Owner Earnings), a today-point is added so the solid line reaches today.
 * EPS / FCF pass `pinToday: false` and then `appendNextQuarterEstimate`.
 */
export function seriesForFairValueChart(
  series: ValuationSeriesPoint[],
  summary: ValuationSummary,
  asOfIso = new Date().toISOString().slice(0, 10),
  options?: { pinToday?: boolean },
): ValuationSeriesPoint[] {
  if (options?.pinToday === false) return series;
  if (summary.fairValue == null || !finite(summary.fairValue) || summary.fairValue <= 0) {
    return series;
  }
  const last = series[series.length - 1];
  const lastDate = last?.date.slice(0, 10);
  if (
    lastDate === asOfIso &&
    last?.fairValue != null &&
    finite(last.fairValue) &&
    Math.abs(last.fairValue - summary.fairValue) < 1e-6
  ) {
    return series;
  }
  if (lastDate && lastDate > asOfIso) return series;
  const todayPoint: ValuationSeriesPoint = {
    date: asOfIso,
    year: Number(asOfIso.slice(0, 4)),
    price: summary.currentPrice,
    metric: summary.fairValueAnchor,
    earningsPower: summary.fairValue,
    fairValue: summary.fairValue,
    normalValue: null,
    dividend: null,
    pe: null,
    estimated: true,
  };
  if (lastDate === asOfIso && last) {
    return [...series.slice(0, -1), { ...last, ...todayPoint }];
  }
  return [...series, todayPoint];
}

export function fairValueFromEstimate(
  eps: number | null | undefined,
  ratio: number | null | undefined,
): number | null {
  if (!finite(eps) || !finite(ratio) || eps <= 0 || ratio <= 0) return null;
  return eps * ratio;
}

function fiscalYearEndMd(lastHistDate?: string): string {
  const md = lastHistDate && /^\d{4}-(\d{2}-\d{2})/.exec(lastHistDate.slice(0, 10));
  return md?.[1] ?? '12-31';
}

function fiscalYearEndIso(year: number, lastHistDate?: string): string {
  return `${year}-${fiscalYearEndMd(lastHistDate)}`;
}

/** Prefer the last historical FY-end over a stale FMP publish date. */
function estimateIsoDate(year: number, date?: string, lastHistDate?: string): string {
  const fyEnd = fiscalYearEndIso(year, lastHistDate);
  if (date && /^\d{4}-\d{2}-\d{2}/.test(date)) {
    const iso = date.slice(0, 10);
    if (iso.slice(5, 10) === fiscalYearEndMd(lastHistDate)) return iso;
  }
  return fyEnd;
}

export function nextIsoDate(iso: string): string {
  const ms = Date.parse(`${iso.slice(0, 10)}T00:00:00Z`);
  if (!Number.isFinite(ms)) return iso.slice(0, 10);
  return new Date(ms + 86_400_000).toISOString().slice(0, 10);
}

/** ~one fiscal quarter after `lastSolidIso` when FMP has no next-earnings date. */
const QUARTER_FALLBACK_DAYS = 91;

export function nextQuarterIso(
  lastSolidIso: string,
  nextEarningsDate?: string | null,
): string {
  const last = Date.parse(`${lastSolidIso.slice(0, 10)}T00:00:00Z`);
  const earnIso = nextEarningsDate?.slice(0, 10) ?? '';
  const fromEarnings = /^\d{4}-\d{2}-\d{2}$/.test(earnIso)
    ? Date.parse(`${earnIso}T00:00:00Z`)
    : Number.NaN;
  if (Number.isFinite(fromEarnings) && fromEarnings > last + 7 * 86_400_000) {
    return earnIso;
  }
  const d = new Date(last);
  d.setUTCDate(d.getUTCDate() + QUARTER_FALLBACK_DAYS);
  return d.toISOString().slice(0, 10);
}

/**
 * First dashed point: last actual quarter → next earnings (or +91d),
 * interpolating toward the next annual estimate × ratio.
 * Same estimate as the yearly tail — drawn to the next print, not today.
 */
export function appendNextQuarterEstimate(
  series: ValuationSeriesPoint[],
  nextEarningsDate: string | null | undefined,
  estimates: ForwardEstimatePoint[],
  fairValueRatio: number | null,
  normalMultiple?: number | null,
): ValuationSeriesPoint[] {
  if (series.length === 0 || !estimates.length) return series;
  const hasFv = finite(fairValueRatio) && fairValueRatio > 0;
  const hasNpe = finite(normalMultiple) && normalMultiple > 0;
  if (!hasFv && !hasNpe) return series;
  const last = series[series.length - 1]!;
  const lastFv = last.fairValue;
  if (lastFv == null || !finite(lastFv) || lastFv <= 0) return series;
  const lastHist = [...series].reverse().find((p) => !p.estimated && !p.forecast) ?? last;
  const nextEst = [...estimates]
    .filter((e) => Number.isFinite(e.year) && e.year > (lastHist.year ?? 0))
    .sort((a, b) => a.year - b.year)[0];
  if (!nextEst) return series;
  const metric = finite(nextEst.eps) ? nextEst.eps : finite(nextEst.metric) ? nextEst.metric : null;
  if (metric == null || metric <= 0) return series;
  const nextQ = nextQuarterIso(last.date, nextEarningsDate);
  const lastT = Date.parse(`${last.date.slice(0, 10)}T00:00:00Z`);
  const nextQT = Date.parse(`${nextQ}T00:00:00Z`);
  if (!Number.isFinite(nextQT) || nextQT <= lastT) return series;
  const estIso = estimateIsoDate(nextEst.year, nextEst.date, lastHist.date);
  const estT = Date.parse(`${estIso}T00:00:00Z`);
  const t =
    Number.isFinite(estT) && estT > lastT
      ? Math.min(1, (nextQT - lastT) / (estT - lastT))
      : 0.25;
  const targetFv = hasFv ? metric * fairValueRatio : null;
  const fv =
    targetFv != null && finite(targetFv) && targetFv > 0
      ? lastFv + (targetFv - lastFv) * t
      : lastFv;
  if (!finite(fv) || fv <= 0) return series;
  if (Math.abs(fv - lastFv) < 0.005) return series;
  const lastMetric = last.metric;
  const interpMetric =
    lastMetric != null && finite(lastMetric) ? lastMetric + (metric - lastMetric) * t : metric;
  const lastNpe = last.normalValue;
  const targetNpe = hasNpe ? metric * normalMultiple : null;
  const interpNpe =
    targetNpe != null && lastNpe != null && finite(lastNpe)
      ? lastNpe + (targetNpe - lastNpe) * t
      : targetNpe;
  return [
    ...series,
    {
      date: nextQ,
      year: Number(nextQ.slice(0, 4)),
      price: null,
      metric: interpMetric,
      earningsPower: hasFv ? fv : null,
      fairValue: hasFv ? fv : null,
      normalValue: interpNpe,
      dividend: null,
      pe: null,
      estimated: true,
      forecast: true,
    },
  ];
}

/** Signed calendar-day gap (UTC). */
export function isoDayDiff(fromIso: string, toIso: string): number {
  const a = Date.parse(`${fromIso.slice(0, 10)}T00:00:00Z`);
  const b = Date.parse(`${toIso.slice(0, 10)}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return Number.POSITIVE_INFINITY;
  return (b - a) / 86_400_000;
}

function quarterValue(q: QuarterlyMetricPoint): number | null {
  if (finite(q.eps)) return q.eps;
  if (finite(q.metric)) return q.metric;
  return null;
}

/**
 * Trailing-twelve-month metric from the last four completed quarters on or
 * before `asOfIso`. Null when fewer than four prints exist or the span is
 * wider than ~13 months (a missing quarter).
 */
export function ttmFromQuarterly(
  quarters: QuarterlyMetricPoint[],
  asOfIso = new Date().toISOString().slice(0, 10),
): { ttm: number | null; asOf: string | null } {
  const completed = quarters
    .map((q) => ({ date: q.date?.slice(0, 10) ?? '', value: quarterValue(q) }))
    .filter((q) => q.date.length === 10 && q.date <= asOfIso && finite(q.value))
    .sort((a, b) => a.date.localeCompare(b.date));
  if (completed.length < 4) return { ttm: null, asOf: null };
  const last4 = completed.slice(-4);
  if (isoDayDiff(last4[0]!.date, last4[3]!.date) > 400) return { ttm: null, asOf: null };
  const ttm = last4.reduce((sum, q) => sum + (q.value as number), 0);
  return { ttm, asOf: last4[3]!.date };
}

/**
 * Solid intra-year steps after the last complete FY: each reported quarter
 * gets FV = rolling TTM × ratio so the line moves with real prints.
 */
export function appendIntraYearTtmSteps(
  series: ValuationSeriesPoint[],
  quarters: QuarterlyMetricPoint[],
  fairValueRatio: number | null,
  asOfIso = new Date().toISOString().slice(0, 10),
  normalMultiple?: number | null,
): ValuationSeriesPoint[] {
  const hasFv = finite(fairValueRatio) && fairValueRatio > 0;
  const hasNpe = finite(normalMultiple) && normalMultiple > 0;
  if ((!hasFv && !hasNpe) || !quarters.length) return series;
  const lastHist = [...series].reverse().find((p) => !p.estimated && !p.forecast);
  const lastHistDate = lastHist?.date.slice(0, 10) ?? '';
  const existing = new Set(series.map((p) => p.date.slice(0, 10)));
  const dates = [
    ...new Set(
      quarters
        .map((q) => q.date.slice(0, 10))
        .filter((d) => d.length === 10 && d <= asOfIso && (!lastHistDate || d > lastHistDate)),
    ),
  ].sort();
  if (!dates.length) return series;

  const extra: ValuationSeriesPoint[] = [];
  for (const date of dates) {
    if (existing.has(date)) continue;
    const { ttm } = ttmFromQuarterly(quarters, date);
    if (ttm == null || ttm <= 0) continue;
    extra.push({
      date,
      year: Number(date.slice(0, 4)),
      price: null,
      metric: ttm,
      earningsPower: hasFv ? ttm * fairValueRatio : null,
      fairValue: hasFv ? ttm * fairValueRatio : null,
      normalValue: hasNpe ? ttm * normalMultiple : null,
      dividend: null,
      pe: null,
      estimated: true,
    });
  }
  if (!extra.length) return series;
  return [...series, ...extra].sort((a, b) => a.date.localeCompare(b.date));
}

function ensureDateAfter(date: string, after: string): string {
  if (!after || date > after) return date;
  return nextIsoDate(after);
}

/**
 * Append up to `horizonYears` forward points: FV = EPS × ratio, Normal P/E = EPS × multiple.
 * Skips years already in the historical / TTM series. Dates are forced strictly
 * after the previous point so lightweight-charts can plot them.
 */
export function appendForwardFairValue(
  series: ValuationSeriesPoint[],
  estimates: ForwardEstimatePoint[],
  fairValueRatio: number | null,
  horizonYears = FORWARD_FAIR_VALUE_YEARS,
  normalMultiple?: number | null,
): ValuationSeriesPoint[] {
  const hasFv = finite(fairValueRatio) && fairValueRatio > 0;
  const hasNpe = finite(normalMultiple) && normalMultiple > 0;
  if ((!hasFv && !hasNpe) || horizonYears <= 0) return series;
  const lastHist = [...series].reverse().find((p) => !p.estimated && !p.forecast) ?? series[series.length - 1];
  const lastHistYear = lastHist?.year ?? 0;
  const lastHistDate = lastHist?.date.slice(0, 10);
  const candidates = [...estimates]
    .filter((e) => Number.isFinite(e.year) && e.year > lastHistYear)
    .sort((a, b) => a.year - b.year);
  if (!candidates.length) return series;

  const out = series.map((p) => ({ ...p }));
  let prevDate = out[out.length - 1]?.date.slice(0, 10) ?? '';
  const lastSolid =
    [...out].reverse().find((p) => !p.forecast) ?? out[out.length - 1];
  const lastSolidDate = lastSolid?.date.slice(0, 10) ?? prevDate;
  let added = 0;
  for (const est of candidates) {
    if (added >= horizonYears) break;
    const metric = finite(est.eps) ? est.eps : finite(est.metric) ? est.metric : null;
    const positive = metric != null && metric > 0;
    const fv = positive && hasFv ? metric * fairValueRatio : null;
    const npe = positive && hasNpe ? metric * normalMultiple : null;
    const rawDate = estimateIsoDate(est.year, est.date, lastHistDate);
    if (prevDate && rawDate <= prevDate) continue;
    const date = ensureDateAfter(rawDate, prevDate);
    if (lastSolidDate && isoDayDiff(lastSolidDate, date) <= NEAR_FORECAST_FY_END_DAYS) {
      continue;
    }
    out.push({
      date,
      year: est.year,
      price: null,
      metric,
      earningsPower: fv,
      fairValue: fv,
      normalValue: npe,
      dividend: null,
      pe: null,
      estimated: true,
      forecast: true,
    });
    prevDate = date;
    added += 1;
  }
  return out;
}
