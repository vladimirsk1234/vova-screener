/** User-facing timeframes. Daily is not a product timeframe. */

export type Timeframe = 'Weekly' | 'Monthly';
export type HistoryTf = Timeframe | 'All';
/** Exit-date grouping axis — not a scan timeframe. */
export type HistoryGroupBy = 'Day' | 'Weekly' | 'Monthly';

export const TIMEFRAMES = ['Weekly', 'Monthly'] as const satisfies readonly Timeframe[];
export const HISTORY_GROUP_BYS = ['Day', 'Weekly', 'Monthly'] as const satisfies readonly HistoryGroupBy[];
export const HISTORY_TFS = ['Weekly', 'Monthly', 'All'] as const satisfies readonly HistoryTf[];

export const DEFAULT_RESULTS_PATH = '/results/Stocks/Weekly/new';

/** Rewrite a remembered Results URL that still names the retired Daily timeframe. */
export function rewriteLegacyResultsPath(path: string): string {
  return path.replace(/^\/results\/(Stocks|ETF)\/Daily\//, '/results/$1/Weekly/');
}

export function isHistoryTf(value: string): value is HistoryTf {
  return (HISTORY_TFS as readonly string[]).includes(value);
}

export function isHistoryGroupBy(value: string): value is HistoryGroupBy {
  return (HISTORY_GROUP_BYS as readonly string[]).includes(value);
}

export function isUserTimeframe(value: string): value is Timeframe {
  return TIMEFRAMES.includes(value as Timeframe);
}

export type HistoryFilters = {
  universe: 'Stocks' | 'ETF';
  tf: HistoryTf;
  groupBy: HistoryGroupBy;
  range: string;
};

export function normalizeHistoryFilters(parsed: {
  universe?: string;
  tf?: string;
  groupBy?: string;
  range?: string;
}): { universe: 'Stocks' | 'ETF'; tf: HistoryTf; groupBy: HistoryGroupBy; range: string } {
  const universe = parsed.universe === 'ETF' ? 'ETF' : 'Stocks';
  const tf = typeof parsed.tf === 'string' && isHistoryTf(parsed.tf) ? parsed.tf : 'All';
  const rawGroup = parsed.groupBy === 'Daily' ? 'Day' : parsed.groupBy;
  const groupBy =
    typeof rawGroup === 'string' && isHistoryGroupBy(rawGroup)
      ? rawGroup
      : tf === 'All'
        ? 'Day'
        : tf;
  return { universe, tf, groupBy, range: parsed.range ?? 'all' };
}
