import {
  BUCKETS,
  HISTORY_RANGES,
  UNIVERSES,
  type HistoryRange,
  type Universe,
} from './api';
import {
  DEFAULT_RESULTS_PATH,
  TIMEFRAMES,
  normalizeHistoryFilters,
  rewriteLegacyResultsPath,
  type HistoryGroupBy,
  type HistoryTf,
} from './userTimeframes';

export { DEFAULT_RESULTS_PATH } from './userTimeframes';

const RESULTS_KEY = 'vova.lastResultsPath';
const APP_KEY = 'vova.lastAppPath';
const HISTORY_KEY = 'vova.historyFilters';
/** Session-only: exact list URL + scroll when opening a chart card. */
const CHART_RETURN_KEY = 'vova.chartReturn';

const HISTORY_RANGE_OPTIONS = HISTORY_RANGES;

const RESULTS_PATH_RE = new RegExp(
  `^/results/(${UNIVERSES.join('|')})/(${TIMEFRAMES.join('|')})/(${BUCKETS.join('|')})$`,
);
const VALUE_PATH_RE = /^\/results\/value$/;

const MANUAL_PATH_RE = /^\/results\/manual(\/rejected\/[^/]+)?$/;
const HISTORY_PATH_RE = /^\/history$/;
const CHART_PATH_RE = /^\/chart\/[^/]+$/;
const LEGACY_FUNDAMENTALS_PATH_RE = /^\/fundamentals\/([^/]+)$/;

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
  return RESULTS_PATH_RE.test(path) || VALUE_PATH_RE.test(path) || MANUAL_PATH_RE.test(path);
}

/** List screens that can open a chart and should be restored by Back. */
export function isChartReturnSourcePath(path: string): boolean {
  return isValidResultsPath(path) || HISTORY_PATH_RE.test(path);
}

function isChartPath(path: string): boolean {
  return CHART_PATH_RE.test(path) || LEGACY_FUNDAMENTALS_PATH_RE.test(path);
}

function isCanonicalAppPath(path: string): boolean {
  return (
    isValidResultsPath(path) ||
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
    const { path, search } = splitPathSearch(saved);
    const rewritten = rewriteLegacyResultsPath(path);
    if (isValidResultsPath(rewritten)) return `${rewritten}${search}`;
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
    const rewritten = rewriteLegacyResultsPath(path);
    const normalized = normalizeAppLocation(rewritten, search);
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
  return `/results/${universe}/Weekly/new`;
}

export type HistoryFilters = {
  universe: Universe;
  tf: HistoryTf;
  groupBy: HistoryGroupBy;
  range: HistoryRange;
};

export function loadHistoryFilters(): HistoryFilters {
  try {
    const raw = migrateFromSession(HISTORY_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<HistoryFilters> & { groupBy?: string; tf?: string };
      const normalized = normalizeHistoryFilters(parsed);
      const range =
        typeof parsed.range === 'string' && isHistoryRange(parsed.range) ? parsed.range : 'all';
      return { ...normalized, range };
    }
  } catch {
    // ignore
  }
  return { universe: 'Stocks', tf: 'All', groupBy: 'Day', range: 'all' };
}

export function saveHistoryFilters(filters: HistoryFilters): void {
  writeStorage(localStorage, HISTORY_KEY, JSON.stringify(filters));
}

export type ChartReturnSnapshot = {
  /** Full path + search of the list page that opened the chart. */
  url: string;
  scrollY: number;
};

/**
 * Remember where the user left when opening a chart. Session-only so TA / Fundamentals
 * toggles keep the same return target without leaking across browser sessions.
 */
export function rememberChartReturn(url: string, scrollY: number): void {
  const { path } = splitPathSearch(url);
  if (!isChartReturnSourcePath(path)) return;
  const snapshot: ChartReturnSnapshot = {
    url,
    scrollY: Number.isFinite(scrollY) && scrollY > 0 ? Math.round(scrollY) : 0,
  };
  writeStorage(sessionStorage, CHART_RETURN_KEY, JSON.stringify(snapshot));
}

export function chartReturnPath(): string | null {
  const raw = readStorage(sessionStorage, CHART_RETURN_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<ChartReturnSnapshot>;
    if (typeof parsed.url !== 'string' || !parsed.url) return null;
    const { path } = splitPathSearch(parsed.url);
    if (!isChartReturnSourcePath(path)) return null;
    return parsed.url;
  } catch {
    return null;
  }
}

/**
 * Consume scroll once when remounting the list after Back. Leaves the URL snapshot so
 * subsequent Back presses (without leaving the list) still have a fallback if needed.
 */
export function takeChartReturnScroll(forUrl: string): number | null {
  const raw = readStorage(sessionStorage, CHART_RETURN_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<ChartReturnSnapshot>;
    if (typeof parsed.url !== 'string' || parsed.url !== forUrl) return null;
    const y =
      typeof parsed.scrollY === 'number' && Number.isFinite(parsed.scrollY) ? parsed.scrollY : 0;
    // Clear scroll so a later visit to the same URL (without opening a chart) does not jump.
    writeStorage(
      sessionStorage,
      CHART_RETURN_KEY,
      JSON.stringify({ url: parsed.url, scrollY: 0 } satisfies ChartReturnSnapshot),
    );
    return y > 0 ? y : null;
  } catch {
    return null;
  }
}

/** True when pathname is a chart (or legacy fundamentals) route. */
export function isChartLocation(pathname: string): boolean {
  return isChartPath(pathname);
}
