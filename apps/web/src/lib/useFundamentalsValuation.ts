import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  FORWARD_FAIR_VALUE_YEARS,
  appendForwardFairValue,
  appendIntraYearTtmSteps,
  appendNextQuarterEstimate,
  buildValuationSeries,
  growthOverrideFromSummary,
  projectMetricByGrowth,
  seriesForFairValueChart,
  type ValuationMetric,
  type ValuationWindowYears,
} from '@vova/engine';
import { api, type FundamentalsPayload } from './api';

export type DividendHud = {
  yieldPct: number | null;
  dps: number | null;
  trend: 'growing' | 'falling' | 'flat';
};

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
  const [windowYears, setWindowYears] = useState<ValuationWindowYears>(5);

  const fundQ = useQuery({
    queryKey: ['fundamentals', ticker],
    queryFn: () => api.fundamentals(ticker, 'eps'),
    enabled: enabled && Boolean(ticker),
    staleTime: 60_000,
    retry: 1,
    refetchInterval: (q) => (q.state.status === 'error' ? 10_000 : false),
  });

  const valuation = useMemo(() => {
    if (!fundQ.data) return null;
    const epsForward = (fundQ.data.estimates ?? []).map((e) => ({ year: e.year, metric: e.eps }));
    const common = {
      currentPrice: fundQ.data.profile.price,
      windowYears,
    };
    if (metric !== 'fcf') {
      return buildValuationSeries(fundQ.data.annual, metric, {
        ...common,
        forward: metric === 'eps' ? epsForward : [],
        ttmMetric:
          metric === 'eps' ? fundQ.data.snapshot.ttmEps : null,
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
  }, [fundQ.data, metric, windowYears]);

  const chartSeries = useMemo(() => {
    if (!valuation) return [];
    const quarterPts =
      metric === 'eps'
        ? (fundQ.data?.quarters ?? []).map((q) => ({ date: q.date, eps: q.eps }))
        : metric === 'fcf'
          ? (fundQ.data?.quarters ?? []).map((q) => ({ date: q.date, metric: q.fcfPerShare }))
          : [];
    const withQuarters = quarterPts.length
      ? appendIntraYearTtmSteps(
          valuation.series,
          quarterPts,
          valuation.summary.fairValueRatio,
          undefined,
          valuation.summary.normalMultiple,
        )
      : valuation.series;
    const pinToday = metric !== 'eps' && metric !== 'fcf';
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
            years: (fundQ.data?.estimates ?? []).map((e) => ({ year: e.year, date: e.date })),
          })
        : [];
    const estimates = metric === 'fcf' ? fcfEstimates : (fundQ.data?.estimates ?? []);
    const towardNextPrint = pinToday
      ? historical
      : appendNextQuarterEstimate(
          historical,
          fundQ.data?.snapshot.nextEarningsDate,
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
  }, [
    valuation,
    fundQ.data?.estimates,
    fundQ.data?.quarters,
    fundQ.data?.snapshot.nextEarningsDate,
    metric,
  ]);

  const dividend = useMemo(() => dividendHud(fundQ.data), [fundQ.data]);

  return {
    metric,
    setMetric,
    windowYears,
    setWindowYears,
    fundQ,
    valuation,
    chartSeries,
    dividend,
  };
}
