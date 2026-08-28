import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  availableValuationWindows,
  buildFairValueChartSeries,
  buildValuationSeries,
  clampValuationWindow,
  coerceChartValuationMetric,
  DEFAULT_CHART_VALUATION_METRIC,
  DEFAULT_VALUATION_WINDOW,
  forecastGrowthFromEstimates,
  fundamentalsHistoryBounds,
  growthOverrideFromSummary,
  quarterlyPointsForMetric,
  usesStreetEpsHistory,
  type ChartValuationMetric,
  type ValuationSeriesPoint,
  type ValuationSummary,
  type ValuationWindowYears,
} from '@vova/engine';
import { api, type FundamentalsPayload } from './api';
import { isFundamentalsPendingError } from './apiError';

export type DividendHud = {
  yieldPct: number | null;
  dps: number | null;
  trend: 'growing' | 'falling' | 'flat';
};

function ttmMetricFor(metric: ChartValuationMetric, data: FundamentalsPayload): number | null {
  if (metric === 'operatingEps') return data.snapshot.ttmOperatingEps ?? null;
  if (metric === 'fcf') return data.snapshot.ttmFcf ?? null;
  return null;
}

function chartSeriesFromValuation(
  valuation: { series: ValuationSeriesPoint[]; summary: ValuationSummary },
  metric: ChartValuationMetric,
  data: FundamentalsPayload | undefined,
): ValuationSeriesPoint[] {
  return buildFairValueChartSeries({
    series: valuation.series,
    summary: valuation.summary,
    metric,
    quarters: quarterlyPointsForMetric(metric, data?.quarters, {
      streetEpsHistory: usesStreetEpsHistory(data?.scale),
    }),
    estimates: (data?.estimates ?? []).map((e) => ({
      year: e.year,
      date: e.date,
      eps: e.eps,
    })),
    nextEarningsDate: data?.snapshot.nextEarningsDate,
  });
}

function asPctPoints(n: number | null | undefined): number | null {
  if (n == null || !Number.isFinite(n)) return null;
  return Math.abs(n) <= 1.5 ? n * 100 : n;
}

function dividendHud(data: FundamentalsPayload | undefined): DividendHud | null {
  if (!data) return null;
  const yieldPct = asPctPoints(data.snapshot.dividendYieldTTM);
  const paid = data.incomeTrend
    .filter((row) => row.dividend != null && Number.isFinite(row.dividend) && row.dividend > 0)
    .sort((a, b) => a.year - b.year);
  const dps = paid.length ? paid[paid.length - 1].dividend : null;
  if ((yieldPct == null || yieldPct <= 0) && (dps == null || dps <= 0)) return null;
  let trend: DividendHud['trend'] = 'flat';
  if (paid.length >= 2) {
    const prev = paid[paid.length - 2].dividend as number;
    const last = paid[paid.length - 1].dividend as number;
    if (last > prev * 1.02) trend = 'growing';
    else if (last < prev * 0.98) trend = 'falling';
  }
  return { yieldPct: yieldPct != null && yieldPct > 0 ? yieldPct : null, dps, trend };
}

export function useFundamentalsValuation(ticker: string, enabled: boolean) {
  const [metric, setMetricState] = useState<ChartValuationMetric>(DEFAULT_CHART_VALUATION_METRIC);
  const setMetric = (next: ChartValuationMetric) => setMetricState(coerceChartValuationMetric(next));
  const [windowYears, setWindowYears] = useState<ValuationWindowYears>(DEFAULT_VALUATION_WINDOW);

  const fundQ = useQuery({
    queryKey: ['fundamentals', ticker],
    queryFn: () => api.fundamentals(ticker, 'eps'),
    enabled: enabled && Boolean(ticker),
    staleTime: 60_000,
    retry: (count: number, err: Error) =>
      isFundamentalsPendingError(err) ? count < 40 : count < 1,
    retryDelay: (_count: number, err: Error) =>
      isFundamentalsPendingError(err) ? 4_000 : 1_000,
    refetchInterval: (q: { state: { status: string; error: unknown } }) =>
      q.state.status === 'error' && isFundamentalsPendingError(q.state.error) ? 8_000 : false,
  });

  const history = useMemo(
    () => fundamentalsHistoryBounds(fundQ.data?.annual ?? []),
    [fundQ.data],
  );
  const historySpanYears = fundQ.data ? history.spanYears : null;
  const effectiveWindowYears = clampValuationWindow(windowYears, historySpanYears);
  const windowChipOptions = useMemo(
    () => availableValuationWindows(historySpanYears).map((w) => (w == null ? 'max' : String(w))),
    [historySpanYears],
  );

  const valuation = useMemo(() => {
    if (!fundQ.data) return null;
    const epsForward = (fundQ.data.estimates ?? []).map((e) => ({ year: e.year, metric: e.eps }));
    const common = {
      currentPrice: fundQ.data.profile.price,
      windowYears: effectiveWindowYears,
    };
    if (metric !== 'fcf') {
      return buildValuationSeries(fundQ.data.annual, metric, {
        ...common,
        forward: [],
        ttmMetric: ttmMetricFor(metric, fundQ.data),
      });
    }
    const epsVal = buildValuationSeries(fundQ.data.annual, 'eps', {
      ...common,
      forward: epsForward,
      ttmMetric: fundQ.data.snapshot.ttmEps,
    });
    return buildValuationSeries(fundQ.data.annual, 'fcf', {
      ...common,
      forward: [],
      ttmMetric: fundQ.data.snapshot.ttmFcf ?? null,
      ...growthOverrideFromSummary(epsVal.summary),
    });
  }, [fundQ.data, metric, effectiveWindowYears]);

  const chartSeries = useMemo(
    () => (valuation ? chartSeriesFromValuation(valuation, metric, fundQ.data) : []),
    [valuation, metric, fundQ.data],
  );

  const forecastValuation = useMemo(() => {
    if (!fundQ.data) return null;
    const box = forecastGrowthFromEstimates(fundQ.data.estimates ?? []);
    if (box.growthRatePct == null) return null;
    return buildValuationSeries(fundQ.data.annual, metric, {
      currentPrice: fundQ.data.profile.price,
      windowYears: effectiveWindowYears,
      ttmMetric: ttmMetricFor(metric, fundQ.data),
      growthRatePct: box.growthRatePct,
      growthSpanYears: box.growthSpanYears,
      growthSource: 'forward',
    });
  }, [fundQ.data, metric, effectiveWindowYears]);

  const forecastChartSeries = useMemo(
    () =>
      forecastValuation ? chartSeriesFromValuation(forecastValuation, metric, fundQ.data) : [],
    [forecastValuation, metric, fundQ.data],
  );

  const dividend = useMemo(() => dividendHud(fundQ.data), [fundQ.data]);

  return {
    metric,
    setMetric,
    windowYears: effectiveWindowYears,
    setWindowYears,
    windowChipOptions,
    historyStartDate: history.firstDate,
    fundQ,
    valuation,
    forecastValuation,
    chartSeries,
    forecastChartSeries,
    dividend,
  };
}
