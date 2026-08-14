/** Batch FMP card metrics for the tickers currently on Results / History. */
import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, type CardFundamentals } from './api';

function normalizeTickers(tickers: string[]): string[] {
  const unique = [...new Set(tickers.map((t) => t.trim().toUpperCase()).filter(Boolean))];
  unique.sort();
  return unique;
}

export function useCardFundamentals(tickers: string[]) {
  // Intentionally join for a stable dep when callers pass a fresh array of the same tickers.
  const key = useMemo(() => normalizeTickers(tickers), [tickers.join('|')]);

  return useQuery({
    queryKey: ['fundamentals-cards', key],
    queryFn: () => api.fundamentalsCards(key),
    enabled: key.length > 0,
    staleTime: 60 * 60_000,
  });
}

export type CardFundamentalsMap = Record<string, CardFundamentals>;
