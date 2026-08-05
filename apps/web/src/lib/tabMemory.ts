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
const HISTORY_PATH_KEY = 'vova.lastHistoryPath';
const HISTORY_KEY = 'vova.historyFilters';

export const DEFAULT_RESULTS_PATH = '/results/Stocks/Daily/new';
export const DEFAULT_HISTORY_PATH = '/history/Stocks';

const HISTORY_TFS = ['Daily', 'Weekly', 'Monthly', 'All'] as const satisfies readonly HistoryTf[];
const HISTORY_RANGE_OPTIONS = HISTORY_RANGES;

const RESULTS_PATH_RE = new RegExp(
  `^/results/(${UNIVERSES.join('|')})/(${TIMEFRAMES.join('|')})/(${BUCKETS.join('|')})(/|\\?|$)`,
);

const HISTORY_PATH_RE = new RegExp(`^/history/(${UNIVERSES.join('|')})(/|\\?|$)`);

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

/** Remember Stocks/ETF on History so the bottom-nav returns to the same universe. */
export function rememberHistoryPath(pathname: string): void {
  if (!HISTORY_PATH_RE.test(pathname)) return;
  try {
    sessionStorage.setItem(HISTORY_PATH_KEY, pathname);
  } catch {
    // Private mode / quota — selection still works within the mounted page.
  }
}

export function lastHistoryPath(): string {
  try {
    const saved = sessionStorage.getItem(HISTORY_PATH_KEY);
    if (saved && HISTORY_PATH_RE.test(saved.split('?')[0] ?? '')) return saved;
    // Migrate from the older filters blob that stored universe in session JSON.
    const raw = sessionStorage.getItem(HISTORY_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<HistoryFilters & { universe?: string }>;
      if (typeof parsed.universe === 'string' && isUniverse(parsed.universe)) {
        return `/history/${parsed.universe}`;
      }
    }
  } catch {
    // ignore
  }
  return DEFAULT_HISTORY_PATH;
}

/** Filters that live outside the URL (universe is the path segment). */
export type HistoryFilters = {
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
      const parsed = JSON.parse(raw) as Partial<HistoryFilters & { universe?: string }>;
      const tf = typeof parsed.tf === 'string' && isHistoryTf(parsed.tf) ? parsed.tf : 'Daily';
      const groupBy =
        typeof parsed.groupBy === 'string' && isTimeframe(parsed.groupBy)
          ? parsed.groupBy
          : tf === 'All'
            ? 'Daily'
            : tf;
      const range =
        typeof parsed.range === 'string' && isHistoryRange(parsed.range) ? parsed.range : 'all';
      return { tf, groupBy, range };
    }
  } catch {
    // ignore
  }
  return { tf: 'Daily', groupBy: 'Daily', range: 'all' };
}

export function saveHistoryFilters(filters: HistoryFilters): void {
  try {
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(filters));
  } catch {
    // ignore
  }
}
