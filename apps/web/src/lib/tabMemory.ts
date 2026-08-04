import {
  BUCKETS,
  HISTORY_RANGES,
  TIMEFRAMES,
  UNIVERSES,
  type HistoryRange,
  type HistoryTf,
  type Timeframe,
  type Universe,
} from './api';

const RESULTS_KEY = 'vova.lastResultsPath';
const HISTORY_KEY = 'vova.historyFilters';

export const DEFAULT_RESULTS_PATH = '/results/Stocks/Daily/new';

const HISTORY_TFS = ['Daily', 'Weekly', 'Monthly', 'All'] as const satisfies readonly HistoryTf[];
const HISTORY_RANGE_OPTIONS = HISTORY_RANGES;

const RESULTS_PATH_RE = new RegExp(
  `^/results/(${UNIVERSES.join('|')})/(${TIMEFRAMES.join('|')})/(${BUCKETS.join('|')})(/|\\?|$)`,
);

function isHistoryTf(value: string): value is HistoryTf {
  return (HISTORY_TFS as readonly string[]).includes(value);
}

function isTimeframe(value: string): value is Timeframe {
  return TIMEFRAMES.includes(value as Timeframe);
}

/** Remember a concrete Results route so Status / Timeframe survive leaving the tab. */
export function rememberResultsPath(pathname: string, search = ''): void {
  if (!RESULTS_PATH_RE.test(pathname)) return;
  try {
    sessionStorage.setItem(RESULTS_KEY, `${pathname}${search}`);
  } catch {
    // Private mode / quota — selection still works within the mounted page.
  }
}

export function lastResultsPath(): string {
  try {
    const saved = sessionStorage.getItem(RESULTS_KEY);
    if (saved && RESULTS_PATH_RE.test(saved.split('?')[0] ?? '')) return saved;
  } catch {
    // ignore
  }
  return DEFAULT_RESULTS_PATH;
}

/** Rebuild a Results URL for another universe, keeping the last timeframe / bucket / sort. */
export function resultsPathForUniverse(universe: Universe): string {
  const last = lastResultsPath();
  const [path, query] = last.split('?');
  const parts = (path ?? '').split('/');
  // ['', 'results', universe, tf, bucket]
  if (parts.length >= 5) {
    parts[2] = universe;
    return query ? `${parts.join('/')}?${query}` : parts.join('/');
  }
  return `/results/${universe}/Daily/new`;
}

export type HistoryFilters = {
  universe: Universe;
  tf: HistoryTf;
  groupBy: Timeframe;
  range: HistoryRange;
};

function isUniverse(value: string): value is Universe {
  return UNIVERSES.includes(value as Universe);
}

function isHistoryRange(value: string): value is HistoryRange {
  return (HISTORY_RANGE_OPTIONS as readonly string[]).includes(value);
}

export function loadHistoryFilters(): HistoryFilters {
  try {
    const raw = sessionStorage.getItem(HISTORY_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<HistoryFilters>;
      const universe =
        typeof parsed.universe === 'string' && isUniverse(parsed.universe)
          ? parsed.universe
          : 'Stocks';
      const tf = typeof parsed.tf === 'string' && isHistoryTf(parsed.tf) ? parsed.tf : 'Daily';
      const groupBy =
        typeof parsed.groupBy === 'string' && isTimeframe(parsed.groupBy)
          ? parsed.groupBy
          : tf === 'All'
            ? 'Daily'
            : tf;
      const range =
        typeof parsed.range === 'string' && isHistoryRange(parsed.range) ? parsed.range : 'all';
      return { universe, tf, groupBy, range };
    }
  } catch {
    // ignore
  }
  return { universe: 'Stocks', tf: 'Daily', groupBy: 'Daily', range: 'all' };
}

export function saveHistoryFilters(filters: HistoryFilters): void {
  try {
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(filters));
  } catch {
    // ignore
  }
}
