/**
 * Fast Graphs–style valuation helpers: normal multiple × per-share metric → fair value.
 * Pure math — no I/O. Data comes from FMP (or any provider) via the API layer.
 */

export type ValuationMetric = 'eps' | 'revenue' | 'fcf' | 'ownerEarnings';

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
};

export type ValuationSeriesPoint = {
  date: string;
  year: number;
  price: number | null;
  /** Selected per-share metric */
  metric: number | null;
  /** metric × normalMultiple (historical earnings power on the price axis) */
  earningsPower: number | null;
  /** Same as earningsPower historically; may extend with estimates */
  fairValue: number | null;
  pe: number | null;
};

export type ValuationSummary = {
  metric: ValuationMetric;
  normalMultiple: number;
  normalMultipleSource: 'median_pe' | 'fallback';
  currentPrice: number | null;
  latestMetric: number | null;
  fairValue: number | null;
  /** (price − fairValue) / fairValue */
  premiumPct: number | null;
  currentPe: number | null;
  metricCagrPct: number | null;
  years: number;
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
  if (med != null && med > 0) return { multiple: Math.round(med * 10) / 10, source: 'median_pe' };
  const fallback = metric === 'revenue' ? 3 : 15;
  return { multiple: fallback, source: 'fallback' };
}

export function cagrPct(first: number, last: number, years: number): number | null {
  if (!finite(first) || !finite(last) || years <= 0 || first <= 0 || last <= 0) return null;
  return (Math.pow(last / first, 1 / years) - 1) * 100;
}

export function buildValuationSeries(
  points: AnnualFundamentalPoint[],
  metric: ValuationMetric,
  opts: { normalMultiple?: number; currentPrice?: number | null } = {},
): { series: ValuationSeriesPoint[]; summary: ValuationSummary } {
  const sorted = [...points].sort((a, b) => a.year - b.year);
  const { multiple, source } =
    opts.normalMultiple != null && opts.normalMultiple > 0
      ? { multiple: opts.normalMultiple, source: 'median_pe' as const }
      : computeNormalMultiple(sorted, metric);

  const series: ValuationSeriesPoint[] = sorted.map((p) => {
    const m = pickMetric(p, metric);
    const earningsPower = finite(m) && m > 0 ? m * multiple : null;
    return {
      date: p.date,
      year: p.year,
      price: finite(p.price) ? p.price : null,
      metric: m,
      earningsPower,
      fairValue: earningsPower,
      pe: finite(p.pe) ? p.pe : finite(p.price) && finite(m) && m > 0 ? p.price / m : null,
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
  const fairValue = finite(latestMetric) && latestMetric > 0 ? latestMetric * multiple : null;
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
      premiumPct,
      currentPe,
      metricCagrPct,
      years: withMetric.length,
    },
  };
}
