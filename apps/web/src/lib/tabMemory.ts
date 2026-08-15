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
const APP_KEY = 'vova.lastAppPath';
const HISTORY_KEY = 'vova.historyFilters';

export const DEFAULT_RESULTS_PATH = '/results/Stocks/Daily/new';

const HISTORY_TFS = ['Daily', 'Weekly', 'Monthly', 'All'] as const satisfies readonly HistoryTf[];
const HISTORY_RANGE_OPTIONS = HISTORY_RANGES;

const RESULTS_PATH_RE = new RegExp(
  `^/results/(${UNIVERSES.join('|')})/(${TIMEFRAMES.join('|')})/(${BUCKETS.join('|')})$`,
);

const MANUAL_PATH_RE = /^\/results\/manual(\/rejected\/[^/]+)?$/;
const HISTORY_PATH_RE = /^\/history$/;
const CHART_PATH_RE = /^\/chart\/[^/]+$/;
const LEGACY_FUNDAMENTALS_PATH_RE = /^\/fundamentals\/([^/]+)$/;

function isHistoryTf(value: string): value is HistoryTf {
  return (HISTORY_TFS as readonly string[]).includes(value);
}

function isTimeframe(value: string): value is Timeframe {
  return TIMEFRAMES.includes(value as Timeframe);
}

function isUniverse(value: string): value is Universe {
  return UNIVERSES.includes(value as Universe);
}

function isHistoryRange(value: string): value is HistoryRange {
  return (HISTORY_RANGE_OPTIONS as readonly string[]).includes(value);
}

function readStorage(store: Storage, key: string): string | null {
  try {
    return store.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(store: Storage, key: string, value: string): void {
  try {
    store.setItem(key, value);
  } catch {
    // Private mode / quota — selection still works within the mounted page.
  }
}

/** One-shot: copy legacy sessionStorage values into localStorage when local is empty. */
function migrateFromSession(key: string): string | null {
  const local = readStorage(localStorage, key);
  if (local != null && local !== '') return local;
  const session = readStorage(sessionStorage, key);
  if (session != null && session !== '') {
    writeStorage(localStorage, key, session);
    return session;
  }
  return null;
}

function splitPathSearch(saved: string): { path: string; search: string } {
  const q = saved.indexOf('?');
  if (q === -1) return { path: saved, search: '' };
  return { path: saved.slice(0, q), search: saved.slice(q) };
}

function isValidResultsPath(path: string): boolean {
  return RESULTS_PATH_RE.test(path);
}

function isCanonicalAppPath(path: string): boolean {
  return (
    isValidResultsPath(path) ||
    MANUAL_PATH_RE.test(path) ||
    HISTORY_PATH_RE.test(path) ||
    CHART_PATH_RE.test(path)
  );
}

function withSearchPrefix(search: string): string {
  if (!search) return '';
  return search.startsWith('?') ? search : `?${search}`;
}

/** Rewrite legacy `/fundamentals/:ticker` to the unified chart window. */
function normalizeAppLocation(
  pathname: string,
  search = '',
): { path: string; search: string } | null {
  const legacy = pathname.match(LEGACY_FUNDAMENTALS_PATH_RE);
  if (legacy) {
    const params = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search);
    params.set('view', 'fundamentals');
    const q = params.toString();
    return { path: `/chart/${legacy[1]}`, search: q ? `?${q}` : '' };
  }
  if (!isCanonicalAppPath(pathname)) return null;
  return { path: pathname, search: withSearchPrefix(search) };
}

/** Remember a concrete Results route so Status / Timeframe survive leaving the tab. */
export function rememberResultsPath(pathname: string, search = ''): void {
  if (!isValidResultsPath(pathname)) return;
  writeStorage(localStorage, RESULTS_KEY, `${pathname}${search}`);
}

export function lastResultsPath(): string {
  const saved = migrateFromSession(RESULTS_KEY);
  if (saved) {
    const { path } = splitPathSearch(saved);
    if (isValidResultsPath(path)) return saved;
  }
  return DEFAULT_RESULTS_PATH;
}

/** Remember any primary app route so / restores where the user left off across sessions. */
export function rememberAppPath(pathname: string, search = ''): void {
  const normalized = normalizeAppLocation(pathname, search);
  if (!normalized) return;
  writeStorage(localStorage, APP_KEY, `${normalized.path}${normalized.search}`);
}

export function lastAppPath(): string {
  const saved = migrateFromSession(APP_KEY);
  if (saved) {
    const { path, search } = splitPathSearch(saved);
    const normalized = normalizeAppLocation(path, search);
    if (normalized) return `${normalized.path}${normalized.search}`;
  }
  // Fall back to last Results path (may itself migrate from sessionStorage).
  return lastResultsPath();
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

export function loadHistoryFilters(): HistoryFilters {
  try {
    const raw = migrateFromSession(HISTORY_KEY);
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
  writeStorage(localStorage, HISTORY_KEY, JSON.stringify(filters));
}
