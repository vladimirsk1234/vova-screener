import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  FORWARD_FAIR_VALUE_YEARS,
  appendForwardFairValue,
  buildValuationSeries,
  forwardEstimatesForMetric,
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
  });

  const valuation = useMemo(() => {
    if (!fundQ.data) return null;
    return buildValuationSeries(fundQ.data.annual, metric, {
      currentPrice: fundQ.data.profile.price,
      windowYears,
      ttmMetric: metric === 'eps' ? fundQ.data.snapshot.ttmEps : null,
    });
  }, [fundQ.data, metric, windowYears]);

  const chartSeries = useMemo(() => {
    if (!valuation) return [];
    const pinned = seriesForFairValueChart(valuation.series, valuation.summary);
    return appendForwardFairValue(
      pinned,
      forwardEstimatesForMetric(metric, fundQ.data?.estimates ?? []),
      valuation.summary.fairValueRatio,
      FORWARD_FAIR_VALUE_YEARS,
      valuation.summary.normalMultiple,
    );
  }, [valuation, fundQ.data?.estimates, metric]);

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
