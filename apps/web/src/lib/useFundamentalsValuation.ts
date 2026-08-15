import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  buildValuationSeries,
  type ValuationMetric,
  type ValuationWindowYears,
} from '@vova/engine';
import { api } from './api';

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
      // FMP only estimates EPS; the other metrics stay on trailing growth.
      forward:
        metric === 'eps'
          ? fundQ.data.estimates.map((e) => ({ year: e.year, metric: e.eps }))
          : [],
      ttmMetric: metric === 'eps' ? fundQ.data.snapshot.ttmEps : null,
    });
  }, [fundQ.data, metric, windowYears]);

  const chartSeries = useMemo(() => {
    if (!fundQ.data || !valuation) return [];
    const lastYear = valuation.series[valuation.series.length - 1]?.year ?? 0;
    const extra = fundQ.data.forecastSeries.filter((p) => p.estimated && p.year > lastYear);
    return [...valuation.series, ...extra];
  }, [fundQ.data, valuation]);

  return {
    metric,
    setMetric,
    windowYears,
    setWindowYears,
    fundQ,
    valuation,
    chartSeries,
  };
}
