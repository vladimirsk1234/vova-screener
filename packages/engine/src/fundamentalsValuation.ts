/**
 * Fast Graphs–style valuation: Normal P/E is median historical price/metric;
 * Fair Value (orange line) uses three FG formulas from trailing metric CAGR in
 * the selected lookback window (1Y … 19Y / MAX):
 *   GDF          — growth < 5%     → P/E = 8.5 + 2g (classic Graham-Dodd)
 *   GDF…P/E=G    — 5% ≤ growth < 15% → P/E = 15 (blend toward 15)
 *   P/E=G        — growth ≥ 15%    → P/E = growth % (Lynch PEG=1)
 * The multiple is fixed for the whole window; each point is metric × that ratio.
 * Default EPS is FMP GAAP diluted — live FMP vs FG shows this matches
 * “Adjusted (Operating) Earnings” in most AAPL/MSFT years. `operatingEps`
 * (NOPAT / shares) is kept as a secondary series; it overshoots FG (AAPL
 * FY25 8.87 vs 7.46). FMP has no FactSet-adjusted operating-EPS field.
 * Historical orange-box growth is trailing in the selected window.
 * Forecasting uses a separate Street-to-Street CAGR and can flip the rule.
 * Pure math — no I/O. Data comes from FMP (or any provider) via the API layer.
 */

export type ValuationMetric = 'eps' | 'operatingEps' | 'revenue' | 'fcf' | 'ownerEarnings';

/** FAST Graphs orange-box labels: GDF / GDF…P/E=G / P/E=G. */
export type FairValueRule = 'gdf' | 'gdf_pe_g' | 'pe_g' | 'none';

/** Chart / Normal P/E window in fiscal years. `null` = MAX (all complete years). */
export type ValuationWindowYears = number | null;

/** FG Historical chips: MAX, 19Y … 1Y. */
export const VALUATION_LOOKBACK_MAX = 19;

export const VALUATION_WINDOW_CHIPS: Array<ValuationWindowYears> = [
  null,
  ...Array.from({ length: VALUATION_LOOKBACK_MAX }, (_, i) => VALUATION_LOOKBACK_MAX - i),
];

/** Growth below this uses classic Graham-Dodd (8.5 + 2g). */
export const GRAHAM_GROWTH_MAX = 5;
/** Graham no-growth multiple. */
export const GRAHAM_DODD_CONSTANT = 8.5;
/** Graham growth coefficient (P/E = 8.5 + 2g). */
export const GRAHAM_DODD_GROWTH_COEF = 2;
/** Floor so a deep earnings decline cannot produce a negative multiple. */
export const GRAHAM_DODD_PE_MIN = 1;
/** Growth at/above this uses P/E = growth % (Lynch PEG=1). */
export const LYNCH_GROWTH_MIN = 15;
/** Fixed fair-value multiple for the 5–15% GDF…P/E=G band. */
export const FAIR_VALUE_PE = 15;
/** @deprecated Use FAIR_VALUE_PE / LYNCH_GROWTH_MIN. Kept for callers. */
export const GROWTH_PE_FLOOR = FAIR_VALUE_PE;

export type AnnualFundamentalPoint = {
  /** Fiscal year end date YYYY-MM-DD */
  date: string;
  year: number;
  /** Year-end (or nearest) adjusted close */
  price: number | null;
  /** Default EPS view: FMP GAAP diluted (closest FMP match to FG operating). */
  eps: number | null;
  /** NOPAT / diluted shares — secondary; overshoots FG adjusted operating. */
  operatingEps?: number | null;
  /** Explicit GAAP diluted when `eps` is used as the default series. */
  gaapEps?: number | null;
  /** Filing-currency operating income (EBIT-like). Used to rebuild operating EPS. */
  operatingIncome?: number | null;
  incomeBeforeTax?: number | null;
  incomeTaxExpense?: number | null;
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
 * Forward analyst estimate for the selected metric. Street EPS is often non-GAAP;
 * do not mix these points with GAAP history when computing growth.
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
  /** Metric × fair-value ratio (GDF / 15× / Lynch). Absent when growth is N/A. */
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
  /** Trailing CAGR behind the GDF / P/E=G rule (positive years in the window only). */
  growthRatePct: number | null;
  /**
   * Calendar years between the first and last positive points used for CAGR.
   * May be shorter than `windowYears` (IPO / turnaround). Null when CAGR is N/A.
   */
  growthSpanYears: number | null;
  /** Historical orange box is always trailing. Forecasting has its own Street CAGR. */
  growthSource: 'trailing' | 'forward';
  fairValueRatio: number | null;
  fairValueRule: FairValueRule;
  years: number;
  /** 1–19, or null for MAX. */
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
      if (finite(point.gaapEps)) return point.gaapEps;
      return finite(point.eps) ? point.eps : null;
    case 'operatingEps':
      if (finite(point.operatingEps)) return point.operatingEps;
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

const DAY_MS = 86_400_000;

export type ValuationChartRangeInput = {
  firstBarDate: string;
  lastBarDate: string;
  windowYears: ValuationWindowYears;
  /** First non-forecast FY / TTM date — used so the fiscal window start is not clipped. */
  firstHistoricalDate?: string | null;
  /** First dashed forecast point. */
  firstForecastDate?: string | null;
  /** Last dashed forecast point — included in `to` so the 3y tail stays on screen. */
  lastForecastDate?: string | null;
  /** Last extra series date (DCF). Always included in `to` when set. */
  lastExtraDate?: string | null;
};

function isoDateMs(iso: string): number {
  return Date.parse(`${iso.slice(0, 10)}T00:00:00Z`);
}

function msToIsoDate(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

export function firstNonForecastDate(
  series: Array<{ date: string; forecast?: boolean }>,
): string | null {
  const hist = series
    .filter((p) => !p.forecast)
    .map((p) => p.date.slice(0, 10))
    .sort();
  return hist[0] ?? null;
}

export function firstForecastDate(
  series: Array<{ date: string; forecast?: boolean }>,
): string | null {
  const fwd = series
    .filter((p) => p.forecast)
    .map((p) => p.date.slice(0, 10))
    .sort();
  return fwd[0] ?? null;
}

export function lastForecastDate(
  series: Array<{ date: string; forecast?: boolean }>,
): string | null {
  const fwd = series
    .filter((p) => p.forecast)
    .map((p) => p.date.slice(0, 10))
    .sort();
  return fwd[fwd.length - 1] ?? null;
}

export function lastSeriesDate(series: Array<{ date: string }>): string | null {
  let last: string | null = null;
  for (const p of series) {
    const d = p.date.slice(0, 10);
    if (!last || d > last) last = d;
  }
  return last;
}

/**
 * Visible Fundamentals range: N years of history (and the first FY in the window),
 * plus the dashed 3y forecast tail.
 */
export function valuationChartRange(input: ValuationChartRangeInput): { from: string; to: string } {
  const firstBar = input.firstBarDate.slice(0, 10);
  const lastBar = input.lastBarDate.slice(0, 10);
  const firstBarMs = isoDateMs(firstBar);
  const lastBarMs = isoDateMs(lastBar);
  const fallback = { from: firstBar, to: lastBar };
  if (!Number.isFinite(firstBarMs) || !Number.isFinite(lastBarMs)) return fallback;

  let fromMs: number;
  if (input.windowYears == null || input.windowYears <= 0) {
    fromMs = firstBarMs;
  } else {
    const calendarFrom = lastBarMs - input.windowYears * 365.25 * DAY_MS;
    const hist = input.firstHistoricalDate?.slice(0, 10);
    const histMs = hist ? isoDateMs(hist) : Number.NaN;
    fromMs = Number.isFinite(histMs) ? Math.min(calendarFrom, histMs) : calendarFrom;
    fromMs = Math.max(fromMs, firstBarMs);
  }

  let toMs = lastBarMs;
  const lastFc = (input.lastForecastDate ?? input.firstForecastDate)?.slice(0, 10);
  if (lastFc) {
    const fcMs = isoDateMs(lastFc);
    if (Number.isFinite(fcMs) && fcMs > toMs) toMs = fcMs;
  }
  const extra = input.lastExtraDate?.slice(0, 10);
  if (extra) {
    const extraMs = isoDateMs(extra);
    if (Number.isFinite(extraMs) && extraMs > toMs) toMs = extraMs;
  }

  return { from: msToIsoDate(fromMs), to: msToIsoDate(toMs) };
}

/**
 * Map a date range onto bar indices. `fromIdx` is never negative so the
 * chart does not open with an empty gutter left of the first candle.
 */
export function valuationChartLogicalRange(
  timesMs: number[],
  range: { from: string; to: string },
  padMs = 0,
): { fromIdx: number; toIdx: number } {
  if (!timesMs.length) return { fromIdx: 0, toIdx: 0 };
  const fromMs = isoDateMs(range.from);
  const toMs = isoDateMs(range.to) + Math.max(0, padMs);
  let fromIdx = 0;
  if (Number.isFinite(fromMs)) {
    for (let i = 0; i < timesMs.length; i++) {
      if (timesMs[i]! <= fromMs) fromIdx = i;
      else break;
    }
  }
  let toIdx = timesMs.length - 1;
  if (Number.isFinite(toMs)) {
    for (let i = 0; i < timesMs.length; i++) {
      if (timesMs[i]! >= toMs) {
        toIdx = i;
        break;
      }
    }
  }
  fromIdx = Math.max(0, fromIdx);
  toIdx = Math.max(fromIdx, toIdx);
  return { fromIdx, toIdx };
}

export function cagrPct(first: number, last: number, years: number): number | null {
  if (!finite(first) || !finite(last) || years <= 0 || first <= 0 || last <= 0) return null;
  return (Math.pow(last / first, 1 / years) - 1) * 100;
}

export type TrailingCagrResult = {
  growthPct: number | null;
  /** Calendar years between the CAGR endpoints; null when growth is N/A. */
  spanYears: number | null;
};

/**
 * Window-span CAGR using the last positive point and the positive point at or
 * before `last.year - lookbackYears`. Years with metric ≤ 0 are skipped.
 * A 1-year window is a simple YoY; if history is shorter than lookback, span
 * is the actual years between the earliest and latest positive points.
 */
export function trailingMetricCagrDetail(
  points: AnnualFundamentalPoint[],
  metric: ValuationMetric,
  lookbackYears = 5,
): TrailingCagrResult {
  const withM = [...points]
    .sort((a, b) => a.year - b.year)
    .filter((p) => {
      const m = pickMetric(p, metric);
      return finite(m) && m > 0;
    });
  if (withM.length < 2) return { growthPct: null, spanYears: null };
  const last = withM[withM.length - 1]!;
  const targetYear = last.year - lookbackYears;
  let first = withM[0]!;
  for (const p of withM) {
    if (p.year <= targetYear) first = p;
  }
  const span = last.year - first.year;
  if (span < 1) return { growthPct: null, spanYears: null };
  const a = pickMetric(first, metric);
  const b = pickMetric(last, metric);
  if (!finite(a) || !finite(b)) return { growthPct: null, spanYears: null };
  return { growthPct: cagrPct(a, b, span), spanYears: span };
}

export function trailingMetricCagr(
  points: AnnualFundamentalPoint[],
  metric: ValuationMetric,
  lookbackYears = 5,
): number | null {
  return trailingMetricCagrDetail(points, metric, lookbackYears).growthPct;
}

/** Lookback for trailing CAGR: selected N years, or a long span for MAX. */
export function windowLookbackYears(windowYears: ValuationWindowYears): number {
  return windowYears != null && windowYears > 0 ? windowYears : 1000;
}

/** Classic Graham-Dodd: 8.5 + 2g, floored so declining names stay plottable. */
export function grahamDoddMultiple(growthPct: number): number {
  const pe = GRAHAM_DODD_CONSTANT + GRAHAM_DODD_GROWTH_COEF * growthPct;
  if (!Number.isFinite(pe)) return FAIR_VALUE_PE;
  return Math.max(GRAHAM_DODD_PE_MIN, Math.round(pe * 100) / 100);
}

/**
 * Forward growth from the Street estimate chain only.
 * Starting at a GAAP year and ending at a Street estimate invents a fake jump
 * (Adobe FY2025 GAAP $16.70 → FY2026 Street $24.41 = +46%).
 * `_windowed` / `_metric` stay on the signature so existing callers compile.
 */
export function forwardMetricCagrDetail(
  _windowed: AnnualFundamentalPoint[],
  forward: ForwardMetricPoint[],
  _metric?: ValuationMetric,
): TrailingCagrResult {
  const pts = [...forward]
    .sort((a, b) => a.year - b.year)
    .filter((p) => finite(p.metric) && (p.metric as number) > 0);
  if (pts.length < 2) return { growthPct: null, spanYears: null };
  const first = pts[0]!;
  const last = pts[pts.length - 1]!;
  const span = last.year - first.year;
  if (span < 1) return { growthPct: null, spanYears: null };
  return {
    growthPct: cagrPct(first.metric as number, last.metric as number, span),
    spanYears: span,
  };
}

export function forwardMetricCagr(
  windowed: AnnualFundamentalPoint[],
  forward: ForwardMetricPoint[],
  metric: ValuationMetric,
): number | null {
  return forwardMetricCagrDetail(windowed, forward, metric).growthPct;
}

export type ForecastGrowthBox = {
  growthRatePct: number | null;
  growthSpanYears: number | null;
  fairValueRatio: number | null;
  fairValueRule: FairValueRule;
};

/**
 * FG Forecasting Graph Key: Street-to-Street CAGR (first estimate → last),
 * then the same GDF / 15× / P/E=G rule. Not mixed with trailing history.
 * A 1-year estimate span is allowed to use Lynch (same as Historical 1Y).
 */
export function forecastGrowthFromEstimates(
  estimates: Array<{ year: number; eps?: number | null; metric?: number | null }>,
): ForecastGrowthBox {
  const forward: ForwardMetricPoint[] = estimates.map((e) => ({
    year: e.year,
    metric: finite(e.metric) ? e.metric : finite(e.eps) ? e.eps : null,
  }));
  const { growthPct, spanYears } = forwardMetricCagrDetail([], forward);
  const { ratio, rule } = fairValueRatioFromGrowth(growthPct, {
    spanYears,
    windowYears: 1,
  });
  return {
    growthRatePct: growthPct,
    growthSpanYears: spanYears,
    fairValueRatio: ratio,
    fairValueRule: rule,
  };
}

export type FairValueRatioOpts = {
  /** Calendar years behind the CAGR. Null when growth is N/A (e.g. one profitable FY). */
  spanYears?: number | null;
  /** Selected chart window; `1` allows Lynch on a 1-year YoY. */
  windowYears?: ValuationWindowYears;
};

/**
 * Fair Value Ratio from trailing growth (FAST Graphs orange line):
 *   g < 5%           → 8.5 + 2g (GDF, classic Graham-Dodd)
 *   5% ≤ g < 15%     → 15× (GDF…P/E=G — blend toward 15)
 *   g ≥ 15%          → P/E = g (P/E=G / Lynch)
 * When `windowYears` / `spanYears` are supplied: Lynch is blocked if the window
 * is not 1Y and the CAGR span is shorter than 2 years (IPO / GAAP turnaround
 * like LYFT 0.06→6.81) — still draw FV at 15×.
 */
export function fairValueRatioFromGrowth(
  growthPct: number | null,
  opts: FairValueRatioOpts = {},
): {
  ratio: number | null;
  rule: FairValueRule;
} {
  const hasSpanGuard = opts.spanYears !== undefined || opts.windowYears !== undefined;
  if (hasSpanGuard) {
    const spanYears = opts.spanYears ?? null;
    const windowYears = opts.windowYears === undefined ? null : opts.windowYears;
    const allowLynch = windowYears === 1 || (spanYears != null && spanYears >= 2);
    if (!allowLynch) {
      return { ratio: FAIR_VALUE_PE, rule: 'gdf_pe_g' };
    }
  }
  if (growthPct == null || !finite(growthPct)) return { ratio: null, rule: 'none' };
  if (growthPct < GRAHAM_GROWTH_MAX) return { ratio: grahamDoddMultiple(growthPct), rule: 'gdf' };
  if (growthPct < LYNCH_GROWTH_MIN) return { ratio: FAIR_VALUE_PE, rule: 'gdf_pe_g' };
  return { ratio: Math.round(growthPct * 100) / 100, rule: 'pe_g' };
}

/** Map a cash date onto a fiscal year ending in `fyEndMonth` (1–12). */
export function fiscalYearForDate(iso: string, fyEndMonth: number): number | null {
  const y = Number(iso.slice(0, 4));
  const m = Number(iso.slice(5, 7));
  if (!Number.isFinite(y) || !Number.isFinite(m) || m < 1 || m > 12) return null;
  const end = fyEndMonth >= 1 && fyEndMonth <= 12 ? fyEndMonth : 12;
  return m <= end ? y : y + 1;
}

function numField(row: Record<string, unknown>, ...keys: string[]): number | null {
  for (const key of keys) {
    const n = row[key];
    if (typeof n === 'number' && Number.isFinite(n)) return n;
    if (typeof n === 'string' && n.trim()) {
      const v = Number(n);
      if (Number.isFinite(v)) return v;
    }
  }
  return null;
}

/**
 * FMP `/owner-earnings` uses `ownersEarnings` / `ownersEarningsPerShare`
 * (plural). Older payloads used `ownerEarnings`. Prefer the per-share field.
 */
export function ownerEarningsPerShareFromRow(row: Record<string, unknown>): number | null {
  const perShare = numField(row, 'ownersEarningsPerShare', 'ownerEarningsPerShare', 'oeps');
  if (perShare != null) return perShare;
  const oe = numField(row, 'ownersEarnings', 'ownerEarnings');
  const shares = numField(
    row,
    'averageSharesOutstanding',
    'weightedAverageShsOutDil',
    'weightedAverageShsOut',
    'shares',
  );
  if (oe == null || shares == null || shares <= 0) return null;
  return oe / shares;
}

/** Sum FMP `adjDividend` into fiscal-year DPS buckets (FG table), not calendar years. */
export function sumDividendsByFiscalYear(
  rows: Array<Record<string, unknown>>,
  fyEndMonth: number,
): Map<number, number> {
  const out = new Map<number, number>();
  for (const row of rows) {
    const rawDate = String(row.date ?? '').slice(0, 10);
    const y =
      rawDate.length === 10
        ? fiscalYearForDate(rawDate, fyEndMonth)
        : Number.isFinite(Number(rawDate.slice(0, 4)))
          ? Number(rawDate.slice(0, 4))
          : null;
    const d = numField(row, 'adjDividend', 'dividend');
    if (y == null || d == null) continue;
    out.set(y, (out.get(y) ?? 0) + d);
  }
  return out;
}

export function yoyChgPct(curr: number | null | undefined, prev: number | null | undefined): number | null {
  if (!finite(curr) || !finite(prev) || prev === 0) return null;
  return ((curr - prev) / Math.abs(prev)) * 100;
}

/**
 * Year-over-year % on the estimate chain only. The first row is always null —
 * do not compare the first Street year to the last GAAP FY.
 */
export function estimateChainChgPct(
  estimates: Array<{ year: number; eps?: number | null; metric?: number | null }>,
): Array<number | null> {
  const sorted = [...estimates].sort((a, b) => a.year - b.year);
  return sorted.map((row, i) => {
    if (i === 0) return null;
    const curr = finite(row.metric) ? row.metric : row.eps;
    const prev = finite(sorted[i - 1]!.metric) ? sorted[i - 1]!.metric : sorted[i - 1]!.eps;
    return yoyChgPct(curr, prev);
  });
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
    /**
     * Street analyst estimates — used by Forecasting (`forecastGrowthFromEstimates`),
     * not by this Historical orange box. Kept on the signature for callers.
     */
    forward?: ForwardMetricPoint[];
    /**
     * Trailing-twelve-month metric (EPS for the EPS view). Preferred fair-value
     * anchor — closer to “current earnings” than the last closed FY.
     */
    ttmMetric?: number | null;
    /**
     * When set, skip this metric's own CAGR and use these values for the
     * orange-box ratio. FCF may borrow Historical EPS trailing this way.
     */
    growthRatePct?: number | null;
    growthSpanYears?: number | null;
    growthSource?: 'trailing' | 'forward';
  } = {},
): { series: ValuationSeriesPoint[]; summary: ValuationSummary } {
  const windowYears = opts.windowYears === undefined ? null : opts.windowYears;
  const sorted = sliceToWindow(points, windowYears);
  const { multiple, source } =
    opts.normalMultiple != null && opts.normalMultiple > 0
      ? { multiple: opts.normalMultiple, source: 'median_pe' as const }
      : computeNormalMultiple(sorted, metric);

  const trailing = trailingMetricCagrDetail(
    sorted,
    metric,
    windowLookbackYears(windowYears),
  );
  const useOverride = opts.growthRatePct !== undefined;
  const growthSource: 'trailing' | 'forward' = useOverride
    ? (opts.growthSource ?? 'trailing')
    : 'trailing';
  const growthRatePct = useOverride
    ? opts.growthRatePct != null && finite(opts.growthRatePct)
      ? opts.growthRatePct
      : null
    : trailing.growthPct;
  const growthSpanYears = useOverride
    ? (opts.growthSpanYears ?? null)
    : trailing.spanYears;
  const { ratio: fairValueRatio, rule: fairValueRule } = fairValueRatioFromGrowth(growthRatePct, {
    spanYears: growthSpanYears,
    windowYears,
  });
  const earningsScale = fairValueRatio ?? multiple;

  const series: ValuationSeriesPoint[] = sorted.map((p) => {
    const m = pickMetric(p, metric);
    const positive = finite(m) && m > 0;
    const earningsPower = positive ? m * earningsScale : 0;
    const fairValue =
      fairValueRatio != null ? (positive ? m * fairValueRatio : 0) : null;
    const normalValue = multiple > 0 ? (positive ? m * multiple : 0) : null;
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
      growthSpanYears,
      growthSource,
      fairValueRatio,
      fairValueRule,
      years: withMetric.length,
      windowYears,
    },
  };
}

/** Copy EPS orange-box growth onto FCF without mixing FCF/EPS CAGR endpoints. */
export function growthOverrideFromSummary(
  summary:
    | Pick<ValuationSummary, 'growthRatePct' | 'growthSpanYears' | 'growthSource'>
    | null
    | undefined,
): {
  growthRatePct?: number;
  growthSpanYears?: number | null;
  growthSource?: 'trailing' | 'forward';
} {
  if (summary?.growthRatePct == null || !finite(summary.growthRatePct)) return {};
  return {
    growthRatePct: summary.growthRatePct,
    growthSpanYears: summary.growthSpanYears,
    growthSource: summary.growthSource,
  };
}

/**
 * Project the selected metric forward at `growthPct`: metric_t = last × (1+g)^Δt.
 * Used for FCF (FMP has no FCF estimates) so the dashed FV stays in FCF dollars.
 */
export function projectMetricByGrowth(opts: {
  lastMetric: number | null | undefined;
  lastYear: number;
  growthPct: number | null | undefined;
  years: Array<{ year: number; date?: string }>;
  horizonYears?: number;
}): ForwardEstimatePoint[] {
  const lastMetric = opts.lastMetric;
  const growthPct = opts.growthPct;
  if (!finite(lastMetric) || lastMetric <= 0 || !finite(growthPct)) return [];
  const g = 1 + growthPct / 100;
  if (!(g > 0)) return [];
  const horizon = opts.horizonYears ?? FORWARD_FAIR_VALUE_YEARS;
  const fromYears: Array<{ year: number; date?: string }> = [...opts.years]
    .filter((e) => Number.isFinite(e.year) && e.year > opts.lastYear)
    .sort((a, b) => a.year - b.year);
  const src: Array<{ year: number; date?: string }> = fromYears.length
    ? fromYears
    : Array.from({ length: horizon }, (_, i) => ({ year: opts.lastYear + i + 1 }));
  const out: ForwardEstimatePoint[] = [];
  for (const y of src) {
    if (out.length >= horizon) break;
    const dt = y.year - opts.lastYear;
    if (dt <= 0) continue;
    const metric = lastMetric * Math.pow(g, dt);
    if (!finite(metric) || metric <= 0) continue;
    out.push({ year: y.year, date: y.date, metric });
  }
  return out;
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

/** Prefer an explicit metric (FCF projection) over EPS so mixed estimate blobs stay in-unit. */
function estimatePointMetric(est: ForwardEstimatePoint): number | null {
  if (finite(est.metric)) return est.metric;
  if (finite(est.eps)) return est.eps;
  return null;
}

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
  const metric = estimatePointMetric(nextEst);
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
 * Append up to `horizonYears` forward points: FV = metric × ratio, Normal P/E = metric × multiple.
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
    const metric = estimatePointMetric(est);
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
