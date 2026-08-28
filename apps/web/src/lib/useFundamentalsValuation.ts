import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  FORWARD_FAIR_VALUE_YEARS,
  appendForwardFairValue,
  appendIntraYearTtmSteps,
  appendNextQuarterEstimate,
  availableValuationWindows,
  buildValuationSeries,
  clampValuationWindow,
  DEFAULT_VALUATION_WINDOW,
  forecastGrowthFromEstimates,
  fundamentalsHistoryBounds,
  growthOverrideFromSummary,
  projectMetricByGrowth,
  seriesForFairValueChart,
  usesStreetEpsHistory,
  type ValuationMetric,
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

function quarterPoints(metric: ValuationMetric, data: FundamentalsPayload | undefined) {
  if (metric === 'eps') {
    if (usesStreetEpsHistory(data?.scale)) return [];
    return (data?.quarters ?? []).map((q) => ({ date: q.date, eps: q.eps }));
  }
  if (metric === 'operatingEps') {
    return (data?.quarters ?? []).map((q) => ({ date: q.date, eps: q.operatingEps ?? q.eps }));
  }
  if (metric === 'fcf') {
    return (data?.quarters ?? []).map((q) => ({ date: q.date, metric: q.fcfPerShare }));
  }
  return [];
}

function chartSeriesFromValuation(
  valuation: { series: ValuationSeriesPoint[]; summary: ValuationSummary },
  metric: ValuationMetric,
  data: FundamentalsPayload | undefined,
): ValuationSeriesPoint[] {
  const quarterPts = quarterPoints(metric, data);
  const withQuarters = quarterPts.length
    ? appendIntraYearTtmSteps(
        valuation.series,
        quarterPts,
        valuation.summary.fairValueRatio,
        undefined,
        valuation.summary.normalMultiple,
      )
    : valuation.series;
  const pinToday = metric !== 'eps' && metric !== 'operatingEps' && metric !== 'fcf';
  const historical = seriesForFairValueChart(withQuarters, valuation.summary, undefined, {
    pinToday,
  });
  const lastHist = [...historical].reverse().find((p) => !p.estimated && !p.forecast);
  const fcfEstimates =
    metric === 'fcf'
      ? projectMetricByGrowth({
          lastMetric: valuation.summary.fairValueAnchor ?? lastHist?.metric ?? null,
          lastYear: lastHist?.year ?? 0,
          growthPct: valuation.summary.growthRatePct,
          years: (data?.estimates ?? []).map((e) => ({ year: e.year, date: e.date })),
        })
      : [];
  const estimates =
    metric === 'fcf' ? fcfEstimates : metric === 'eps' ? (data?.estimates ?? []) : [];
  const towardNextPrint = pinToday
    ? historical
    : appendNextQuarterEstimate(
        historical,
        data?.snapshot.nextEarningsDate,
        estimates,
        valuation.summary.fairValueRatio,
        valuation.summary.normalMultiple,
      );
  return appendForwardFairValue(
    towardNextPrint,
    estimates,
    valuation.summary.fairValueRatio,
    FORWARD_FAIR_VALUE_YEARS,
    valuation.summary.normalMultiple,
  );
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
  const [metric, setMetric] = useState<ValuationMetric>('eps');
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
        forward: metric === 'eps' ? epsForward : [],
        ttmMetric:
          metric === 'eps'
            ? fundQ.data.snapshot.ttmEps
            : metric === 'operatingEps'
              ? fundQ.data.snapshot.ttmOperatingEps ?? null
              : null,
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
    if (metric !== 'eps' && metric !== 'operatingEps') return null;
    const box = forecastGrowthFromEstimates(fundQ.data.estimates ?? []);
    if (box.growthRatePct == null) return null;
    return buildValuationSeries(fundQ.data.annual, metric, {
      currentPrice: fundQ.data.profile.price,
      windowYears: effectiveWindowYears,
      ttmMetric:
        metric === 'eps'
          ? fundQ.data.snapshot.ttmEps
          : (fundQ.data.snapshot.ttmOperatingEps ?? null),
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
