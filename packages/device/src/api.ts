/**
 * The `/api` the web UI talks to, answered on the phone. Routes, query parameters and response
 * shapes follow the NestJS controllers (`apps/api`), so the same React screens run unchanged.
 * Fundamentals (FMP) routes are not served.
 */
import { barPeriodKey, periodKey as clockPeriodKey } from '../../../apps/api/src/scans/period.ts';
import { evaluateClose, evaluateSymbol, type BuySignal, type EvaluateParams } from '../../engine/src/evaluate.ts';
import type { IndicatorParams } from '../../engine/src/indicatorParams.ts';
import { signalAge, type SignalAge } from '../../engine/src/signalAge.ts';
import {
  canadianYahooCandidates,
  canadianYahooCandidatesIfBare,
  parseManualTickers,
  resolveManualAgainstListings,
  shortSymbol,
  stripCanadianYahooSuffix,
  type ParsedEntry,
} from '../../engine/src/tickers.ts';
import { inferTvSymbol } from '../../engine/src/tradingview.ts';
import type { OhlcSeries } from '../../engine/src/types.ts';
import { buildChartPayload, ChartError } from './chart';
import { historyReport, historyTrades } from './history';
import { newId } from './id';
import { interestRank, round2 } from './money';
import { emptyRebuildStatus, rebuildHistory, type RebuildStatus } from './rebuild';
import { bucketCounts, listResults } from './results';
import { scanUniverse } from './scan';
import type { BarsTf, ScreenerStore } from './store';
import { resizeActive, setInterest } from './tracker';
import {
  BUCKETS,
  HISTORY_GROUP_BYS,
  TIMEFRAMES,
  UNIVERSES,
  type AppSettings,
  type Bucket,
  type HistoryGroupBy,
  type HistoryPeriodSort,
  type HistoryRange,
  type HistoryTf,
  type HistoryTradeSort,
  type Interest,
  type RejectDetail,
  type ResultSort,
  type ScanMeta,
  type SortDir,
  type TrackedSignal,
  type Universe,
  type UserTf,
} from './types';
import { fetchYahooDailyCloses, fetchYahooOhlc } from './yahoo';

export type ApiResponse = { status: number; body: string };

export type DeviceApiDeps = {
  store: ScreenerStore;
  universe: Record<Universe, ParsedEntry[]>;
  fetchBars?: typeof fetchYahooOhlc;
  fetchCloses?: typeof fetchYahooDailyCloses;
  /** Lists changed without the UI asking (a background scan finished a universe). */
  onDataChanged?: () => void;
  now?: () => Date;
};

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

const FUNDAMENTALS_MESSAGE = 'Fundamentals are not part of the iPhone app.';
const RESULT_SORTS: readonly ResultSort[] = ['rr', 'pnl', 'interest', 'symbol'];
const TRADE_SORTS: readonly HistoryTradeSort[] = ['date', 'pnl', 'r', 'rr', 'interest', 'symbol'];
const PERIOD_SORTS: readonly HistoryPeriodSort[] = ['period', 'pnl', 'winRate', 'trades', 'rr'];
const HISTORY_RANGES: readonly HistoryRange[] = ['all', 'ytd', '1m', '3m', '6m', '1y', 'max'];
/** Unlisted (Manual) tickers: same freshness the server's chart path uses before it re-downloads. */
const CHART_BARS_MAX_AGE_MS = 30 * 60 * 1000;
const DAILY_REFRESH_MS = 24 * 60 * 60 * 1000;
const BENCHMARK_REFRESH_MS = 24 * 60 * 60 * 1000;
const MANUAL_RUNS_KEY = 'device:manualRuns';
const TICKER_INTEREST_KEY = 'device:tickerInterest';

type ScanCounters = {
  total: number;
  downloaded: number;
  evaluated: number;
  signals: number;
  closes: number;
  rejected: number;
  skipped: number;
  fromCache: number;
};

type ProgressEvent = {
  runId: string;
  phase: 'queued' | 'resolving' | 'scanning' | 'saving' | 'completed' | 'cancelled' | 'failed';
  percent: number;
  message: string;
  counters?: Record<string, number>;
};

type ManualRun = {
  _id: string;
  params: Record<string, unknown> & { tf: UserTf; manualTickers: string };
  status: 'queued' | 'running' | 'completed' | 'cancelled' | 'failed';
  asOf: string | null;
  barsOldestAt: string | null;
  counters: ScanCounters;
  reasonCounts: Record<string, number>;
  timings: { totalMs: number };
  newSymbols: string[];
  error?: string;
  createdAt: string;
  signals: BuySignal[];
  rejections: Array<{ _id: string; symbol: string; yahooTicker: string; reason: string; detail: RejectDetail | null }>;
  progress: ProgressEvent;
};

const EMPTY_COUNTERS: ScanCounters = {
  total: 0,
  downloaded: 0,
  evaluated: 0,
  signals: 0,
  closes: 0,
  rejected: 0,
  skipped: 0,
  fromCache: 0,
};

/** `toResultRow` from the server, without the FMP entry stamps. */
export function toResultRow(doc: TrackedSignal) {
  const closed = doc.status === 'closed';
  const atExit = closed || doc.provisionalClose;
  const shares = doc.shares ?? 0;
  return {
    id: doc.id,
    symbol: shortSymbol(doc.symbol || doc.yahooTicker),
    tvSymbol: doc.tvSymbol || doc.symbol,
    yahooTicker: doc.yahooTicker,
    companyName: doc.companyName || doc.yahooTicker,
    universe: doc.universe,
    tf: doc.tf,
    status: doc.status,
    provisional: Boolean(doc.provisional),
    provisionalClose: Boolean(doc.provisionalClose),
    entry: doc.entry,
    tp: doc.tp,
    sl: doc.sl,
    rr: doc.rrAtEntry,
    currentRr: doc.lastRr,
    shares,
    positionValue: round2((doc.entry ?? 0) * shares),
    riskUsd: doc.riskUsd ?? 0,
    isStrong: Boolean(doc.isStrong),
    openedPeriodKey: doc.openedPeriodKey,
    openedAsOf: doc.openedAsOf ?? null,
    barsSinceValid: doc.barsSinceValid,
    validSinceAsOf: doc.validSinceAsOf ?? null,
    lastPrice: doc.lastPrice,
    lastSeenAsOf: doc.lastSeenAsOf ?? null,
    pnlUsd: atExit ? doc.pnlUsd : doc.unrealizedUsd,
    pnlR: atExit ? doc.pnlR : doc.unrealizedR,
    pnlPct: atExit ? doc.pnlPct : doc.unrealizedPct,
    realized: closed,
    closedPeriodKey: doc.closedPeriodKey ?? null,
    exitDate: doc.exitDate ?? null,
    exitPrice: doc.exitPrice,
    exitReason: doc.exitReason ?? null,
    holdPeriods: doc.holdPeriods,
    interest: doc.interest ?? null,
  };
}

export function createDeviceApi(deps: DeviceApiDeps) {
  const store = deps.store;
  const fetchBars = deps.fetchBars ?? fetchYahooOhlc;
  const fetchCloses = deps.fetchCloses ?? fetchYahooDailyCloses;
  const now = deps.now ?? (() => new Date());
  const listed = new Map<string, { entry: ParsedEntry; universe: Universe }>();
  for (const universe of UNIVERSES) {
    for (const entry of deps.universe[universe]) {
      if (!listed.has(entry.yahoo)) listed.set(entry.yahoo, { entry, universe });
    }
  }

  let scanning: { universe: Universe; tf: UserTf } | null = null;
  let scanBusy = false;
  let scanAbort: AbortController | null = null;
  let rebuildState: RebuildStatus = emptyRebuildStatus();
  let rebuildBusy = false;
  let manualRuns: Map<string, ManualRun> | null = null;
  const manualAborts = new Map<string, AbortController>();
  const ageMemo = new Map<string, SignalAge>();
  const dailyRefreshing = new Set<string>();
  let benchmarkRefreshing = false;

  const json = (status: number, data: unknown): ApiResponse => ({
    status,
    body: JSON.stringify(data ?? null),
  });

  async function settings(): Promise<AppSettings> {
    return store.getSettings();
  }

  async function scanMeta(universe: Universe, tf: UserTf): Promise<ScanMeta> {
    const stored = await store.getScanMeta(universe, tf);
    const running = scanning?.universe === universe && scanning.tf === tf;
    return {
      periodKey: stored?.periodKey ?? clockPeriodKey(tf, now()),
      asOf: stored?.asOf ?? null,
      newestAsOf: stored?.newestAsOf ?? stored?.asOf ?? null,
      finishedAt: stored?.finishedAt ?? null,
      running,
      status: running ? 'running' : (stored?.status ?? null),
    };
  }

  function cachedAge(ticker: string, tf: UserTf, bars: OhlcSeries): SignalAge {
    const key = `${ticker}|${tf}|${bars.length}|${bars[bars.length - 1]?.date ?? ''}`;
    let hit = ageMemo.get(key);
    if (!hit) {
      hit = signalAge(bars);
      ageMemo.set(key, hit);
    }
    return hit;
  }

  /** `ResultsService.revalidateLiveAges`: drop dead setups from NEW/VALID against stored bars. */
  async function revalidateLiveAges(universe: Universe, tf: UserTf) {
    const signals = await store.listSignals();
    const changed: TrackedSignal[] = [];
    for (const doc of signals) {
      if (doc.universe !== universe || doc.tf !== tf || doc.status !== 'active') continue;
      if (doc.provisionalClose || doc.imported) continue;
      const cached = await store.getBars(doc.yahooTicker, tf);
      if (!cached?.bars.length) continue;
      const age = cachedAge(doc.yahooTicker, tf, cached.bars);
      const nextValid = age.barsSinceValid != null;
      const prevValid = doc.signalValid !== false;
      if (
        prevValid === nextValid &&
        doc.barsSinceValid === age.barsSinceValid &&
        (doc.validSinceAsOf ?? null) === (age.validSinceAsOf ?? null)
      ) {
        continue;
      }
      changed.push({
        ...doc,
        barsSinceValid: age.barsSinceValid,
        validSinceAsOf: age.validSinceAsOf,
        signalValid: nextValid,
      });
    }
    if (changed.length) await store.upsertSignals(changed);
  }

  async function findSignal(id: string): Promise<TrackedSignal> {
    const doc = (await store.listSignals()).find((row) => row.id === id);
    if (!doc) throw new HttpError(404, 'signal not found');
    return doc;
  }

  // ---------------------------------------------------------------- Results

  async function results(q: URLSearchParams) {
    const universe = parseUniverse(q.get('universe'));
    const tf = parseTf(q.get('tf'));
    const bucket = BUCKETS.includes(q.get('bucket') as Bucket) ? (q.get('bucket') as Bucket) : 'new';
    const sort = RESULT_SORTS.includes(q.get('sort') as ResultSort) ? (q.get('sort') as ResultSort) : 'rr';
    const dir = parseDir(q.get('dir'));
    const scan = await scanMeta(universe, tf);
    if (bucket !== 'closed') await revalidateLiveAges(universe, tf);
    const { minRr } = await settings();
    const page = listResults({
      signals: await store.listSignals(),
      universe,
      tf,
      bucket,
      sort,
      dir,
      minRr,
      scan,
      limit: int(q.get('limit')),
      offset: int(q.get('offset')),
    });
    return { universe, tf, bucket, sort, dir, total: page.total, rows: page.rows.map(toResultRow), scan };
  }

  async function summary() {
    const { minRr } = await settings();
    const out: Record<string, Record<string, unknown>> = {};
    for (const universe of UNIVERSES) {
      out[universe] = {};
      for (const tf of TIMEFRAMES) {
        const scan = await scanMeta(universe, tf);
        await revalidateLiveAges(universe, tf);
        const counts = bucketCounts(await store.listSignals(), universe, tf, minRr, scan);
        out[universe][tf] = { counts, scan };
      }
    }
    return out;
  }

  async function lookup(q: URLSearchParams) {
    const ticker = q.get('yahooTicker') ?? '';
    const tf = parseTf(q.get('tf'));
    const doc = (await store.listSignals()).find(
      (row) => row.yahooTicker === ticker && row.tf === tf && row.status === 'active',
    );
    return doc ? toResultRow(doc) : null;
  }

  async function patchInterest(id: string, body: unknown) {
    const value = (body as { interest?: unknown } | null)?.interest;
    const interest: Interest | null = value === 'interested' || value === 'not_interested' ? value : null;
    const doc = await findSignal(id);
    const [next] = setInterest([doc], id, interest);
    await store.upsertSignals([next]);
    return toResultRow(next);
  }

  // ---------------------------------------------------------------- History

  async function benchmark() {
    const cached = await store.getBenchmark();
    const stale = !cached || Date.now() - Date.parse(cached.fetchedAt) > BENCHMARK_REFRESH_MS;
    if (!cached) await refreshBenchmark();
    else if (stale) void refreshBenchmark();
    const row = await store.getBenchmark();
    return row ? { symbol: row.symbol, bars: row.bars } : null;
  }

  async function refreshBenchmark() {
    if (benchmarkRefreshing) return;
    benchmarkRefreshing = true;
    try {
      for (const symbol of ['SPY', '^GSPC']) {
        const bars = await fetchCloses(symbol).catch(() => null);
        if (bars?.length) {
          await store.putBenchmark({ symbol, bars, fetchedAt: new Date().toISOString() });
          return;
        }
      }
    } finally {
      benchmarkRefreshing = false;
    }
  }

  async function history(q: URLSearchParams) {
    const { maxRiskUsd, minRr } = await settings();
    return historyReport({
      signals: await store.listSignals(),
      universe: parseUniverse(q.get('universe')),
      tf: parseHistoryTf(q.get('tf')),
      groupBy: parseGroupBy(q.get('groupBy')),
      range: parseRange(q.get('range')),
      sort: PERIOD_SORTS.includes(q.get('sort') as HistoryPeriodSort) ? (q.get('sort') as HistoryPeriodSort) : 'period',
      dir: parseDir(q.get('dir')),
      minRr,
      maxRiskUsd,
      benchmark: await benchmark(),
      now: now(),
    });
  }

  async function trades(q: URLSearchParams) {
    const { maxRiskUsd, minRr } = await settings();
    const page = historyTrades({
      signals: await store.listSignals(),
      universe: parseUniverse(q.get('universe')),
      tf: parseHistoryTf(q.get('tf')),
      groupBy: parseGroupBy(q.get('groupBy')),
      range: parseRange(q.get('range')),
      periodKey: q.get('periodKey') ?? undefined,
      sort: TRADE_SORTS.includes(q.get('sort') as HistoryTradeSort) ? (q.get('sort') as HistoryTradeSort) : 'date',
      dir: parseDir(q.get('dir')),
      minRr,
      maxRiskUsd,
      limit: int(q.get('limit')),
      offset: int(q.get('offset')),
      now: now(),
    });
    return { total: page.total, rows: page.rows.map(toResultRow) };
  }

  function startRebuild() {
    if (rebuildBusy) return { started: false, reason: 'A history rebuild is already running' };
    rebuildBusy = true;
    rebuildState = { ...emptyRebuildStatus(), status: 'running', startedAt: new Date().toISOString() };
    void (async () => {
      try {
        const { maxRiskUsd } = await settings();
        const next = await rebuildHistory({
          signals: await store.listSignals(),
          universe: deps.universe,
          barsFor: async (ticker, tf) => (await store.getBars(ticker, tf))?.bars ?? null,
          maxRiskUsd,
          state: rebuildState,
          yieldEvery: () => new Promise((r) => setTimeout(r, 0)),
        });
        await store.saveSignals(next);
        rebuildState = {
          ...rebuildState,
          status: 'done',
          finishedAt: new Date().toISOString(),
          progress: { universe: null, tf: null, symbolsDone: 0, symbolsTotal: 0 },
        };
        deps.onDataChanged?.();
      } catch (err) {
        rebuildState = {
          ...rebuildState,
          status: 'failed',
          finishedAt: new Date().toISOString(),
          error: (err as Error).message,
        };
      } finally {
        rebuildBusy = false;
      }
    })();
    return { started: true };
  }

  // ---------------------------------------------------------------- Settings

  async function putSettings(body: unknown) {
    const patch = (body ?? {}) as Partial<AppSettings>;
    const prev = await settings();
    const next: AppSettings = { ...prev };
    const risk = Number(patch.maxRiskUsd);
    if (Number.isFinite(risk) && risk > 0) next.maxRiskUsd = Math.round(risk * 100) / 100;
    if (patch.minRr !== undefined) {
      const minRr = Number(patch.minRr);
      if (Number.isFinite(minRr) && minRr >= 0) next.minRr = Math.round(minRr * 100) / 100;
    }
    await store.putSettings(next);
    if (next.maxRiskUsd !== prev.maxRiskUsd) {
      await store.saveSignals(resizeActive(await store.listSignals(), next.maxRiskUsd));
    }
    return webSettings(next);
  }

  function webSettings(s: AppSettings) {
    return { maxRiskUsd: s.maxRiskUsd, minRr: s.minRr, fundamentalsFilter: 'all' };
  }

  // ---------------------------------------------------------------- Background scans

  function runNow(body: unknown) {
    const tf = (body as { tf?: string } | null)?.tf;
    if (tf && tf !== 'all' && !TIMEFRAMES.includes(tf as UserTf)) {
      throw new HttpError(400, `unknown timeframe ${tf}`);
    }
    if (scanBusy) return { started: false, timeframes: [], reason: 'A scan is already running' };
    const tfs: UserTf[] = tf && tf !== 'all' ? [tf as UserTf] : [...TIMEFRAMES];
    scanBusy = true;
    const controller = new AbortController();
    scanAbort = controller;
    void (async () => {
      try {
        for (const universe of UNIVERSES) {
          for (const frame of tfs) {
            if (controller.signal.aborted) return;
            scanning = { universe, tf: frame };
            await scanUniverse({
              store,
              entries: deps.universe[universe],
              universe,
              tf: frame,
              settings: await settings(),
              signal: controller.signal,
              concurrency: 6,
              forceRefresh: true,
              fetchBars,
              now: now(),
            });
            deps.onDataChanged?.();
          }
        }
      } finally {
        scanning = null;
        scanBusy = false;
        scanAbort = null;
        deps.onDataChanged?.();
      }
    })();
    return { started: true, timeframes: tfs };
  }

  // ---------------------------------------------------------------- Manual scans

  async function runs(): Promise<Map<string, ManualRun>> {
    if (!manualRuns) {
      const saved = (await store.getPreset<ManualRun[]>(MANUAL_RUNS_KEY)) ?? [];
      manualRuns = new Map(saved.map((run) => [run._id, run]));
    }
    return manualRuns;
  }

  async function persistRuns() {
    const all = [...(await runs()).values()]
      .filter((run) => run.status !== 'running' && run.status !== 'queued')
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .slice(0, 10);
    await store.putPreset(MANUAL_RUNS_KEY, all);
  }

  async function findRun(id: string): Promise<ManualRun> {
    const run = (await runs()).get(id);
    if (!run) throw new HttpError(404, 'run not found');
    return run;
  }

  function publicRun(run: ManualRun) {
    const { signals: _s, rejections: _r, progress: _p, ...rest } = run;
    return { ...rest };
  }

  /** Mirrors `UniverseService.resolveManualEntries` over the bundled lists. */
  function resolveManual(manualTickers: string): ParsedEntry | null {
    const entry = parseManualTickers(manualTickers).entries[0];
    if (!entry) return null;
    const firstPart =
      String(manualTickers || '')
        .split(/[,\n]/)
        .map((s) => s.trim())
        .filter(Boolean)[0] ?? '';
    const short = stripCanadianYahooSuffix(entry.yahoo);
    const candidates = new Set([short, ...canadianYahooCandidates(short)]);
    const tvRe = new RegExp(`(?:^|:)${short.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`, 'i');
    const listings: ParsedEntry[] = [];
    for (const { entry: listedEntry } of listed.values()) {
      if (candidates.has(listedEntry.yahoo) || tvRe.test(listedEntry.tv)) listings.push(listedEntry);
    }
    return resolveManualAgainstListings(firstPart, listings) ?? entry;
  }

  async function startManual(body: unknown) {
    const input = (body ?? {}) as Record<string, unknown>;
    const tf: UserTf = input.tf === 'Monthly' ? 'Monthly' : 'Weekly';
    const { maxRiskUsd } = await settings();
    const params = {
      source: 'MANUAL SCAN',
      manualTickers: String(input.manualTickers ?? ''),
      tf,
      direction: input.direction === 'sell' ? 'sell' : 'buy',
      minRr: Number.isFinite(Number(input.minRr)) ? Number(input.minRr) : 0,
      riskPerTrade: Number(input.riskPerTrade) > 0 ? Number(input.riskPerTrade) : maxRiskUsd,
      noRrReq: input.noRrReq !== false,
      useLastHlSl: input.useLastHlSl !== false,
      newOnly: input.newOnly === true,
    };
    const runId = newId();
    const run: ManualRun = {
      _id: runId,
      params,
      status: 'queued',
      asOf: null,
      barsOldestAt: null,
      counters: { ...EMPTY_COUNTERS },
      reasonCounts: {},
      timings: { totalMs: 0 },
      newSymbols: [],
      createdAt: new Date().toISOString(),
      signals: [],
      rejections: [],
      progress: { runId, phase: 'queued', percent: 0, message: 'Queued' },
    };
    (await runs()).set(runId, run);
    void executeManual(run);
    return { runId, params };
  }

  async function executeManual(run: ManualRun) {
    const started = Date.now();
    const controller = new AbortController();
    manualAborts.set(run._id, controller);
    const publish = (phase: ProgressEvent['phase'], percent: number, message: string) => {
      run.progress = { runId: run._id, phase, percent, message, counters: { ...run.counters } };
    };
    try {
      publish('resolving', 0, 'Resolving universe...');
      const entry = resolveManual(run.params.manualTickers);
      run.status = 'running';
      if (!entry) {
        run.status = 'completed';
        publish('completed', 100, 'Nothing to scan');
        return;
      }
      run.counters.total = 1;
      const known = listed.get(entry.yahoo);
      const tf = run.params.tf;
      let yahoo = entry.yahoo;
      let tv = entry.tv;
      let name = entry.name ?? known?.entry.name ?? null;
      let bars: OhlcSeries | null = null;
      let fromCache = false;
      if (known) {
        // Listed: stored bars only, like the server's barsCacheOnly path; first open downloads once.
        const cached = await store.getBars(entry.yahoo, tf);
        if (cached?.bars.length) {
          bars = cached.bars;
          fromCache = true;
        }
      }
      if (!bars) {
        publish('resolving', 0, `Downloading ${entry.yahoo} from Yahoo…`);
        const pulled = await fetchBars(entry.yahoo, tf, { signal: controller.signal });
        bars = pulled.bars;
        if (pulled.bars) {
          name = name ?? pulled.companyName;
          await store.putBars({ yahooTicker: entry.yahoo, tf, bars: pulled.bars, fetchedAt: new Date().toISOString(), companyName: name });
        } else {
          for (const alt of canadianYahooCandidatesIfBare(entry.yahoo) ?? []) {
            if (controller.signal.aborted) break;
            const altPull = await fetchBars(alt, tf, { signal: controller.signal });
            if (!altPull.bars?.length) continue;
            bars = altPull.bars;
            yahoo = alt;
            tv = inferTvSymbol(alt);
            name = name ?? altPull.companyName;
            await store.putBars({ yahooTicker: alt, tf, bars: altPull.bars, fetchedAt: new Date().toISOString(), companyName: name });
            break;
          }
        }
      }
      run.counters.downloaded = 1;
      if (fromCache) run.counters.fromCache = 1;
      const evalParams: EvaluateParams = {
        direction: run.params.direction as 'buy' | 'sell',
        minRr: run.params.minRr as number,
        riskPerTrade: run.params.riskPerTrade as number,
        noRrReq: run.params.noRrReq as boolean,
        useLastHlSl: run.params.useLastHlSl as boolean,
        newOnly: run.params.newOnly as boolean,
        tf,
        minAvgVolume: 0,
      };
      const evaluation = evaluateSymbol({ bars, yahooTicker: yahoo, tvSymbol: tv, companyName: name ?? undefined, params: evalParams });
      run.counters.evaluated = 1;
      if (evalParams.direction === 'buy') {
        const close = evaluateClose({ bars, yahooTicker: yahoo, tvSymbol: tv, companyName: name ?? undefined, params: evalParams });
        if (close) run.counters.closes = 1;
      }
      if (evaluation.status === 'signal' && evaluation.signal.kind === 'buy') {
        run.counters.signals = 1;
        run.signals = [evaluation.signal];
        run.asOf = evaluation.signal.asOf;
      } else if (evaluation.status !== 'signal') {
        const bucket = evaluation.status === 'rejected' ? 'rejected' : 'skipped';
        run.counters[bucket] += 1;
        run.reasonCounts[evaluation.reason] = (run.reasonCounts[evaluation.reason] ?? 0) + 1;
        if (evaluation.status === 'rejected') {
          run.rejections = [
            {
              _id: newId(),
              symbol: shortSymbol(tv || yahoo),
              yahooTicker: yahoo,
              reason: evaluation.reason,
              detail: evaluation.detail ?? null,
            },
          ];
        }
        run.asOf = bars?.length ? bars[bars.length - 1].date : null;
      }
      publish('scanning', 100, `Scanned 1/1 · ${run.counters.signals} signals`);
      const cancelled = controller.signal.aborted;
      publish(cancelled ? 'cancelled' : 'saving', 99, 'Finalising...');
      run.status = cancelled ? 'cancelled' : 'completed';
      run.timings = { totalMs: Date.now() - started };
      publish(
        cancelled ? 'cancelled' : 'completed',
        100,
        cancelled ? 'Cancelled after 1/1' : `Done · ${run.counters.signals} signals · ${run.counters.closes} closes`,
      );
    } catch (err) {
      run.status = 'failed';
      run.error = (err as Error).message;
      publish('failed', 100, run.error);
    } finally {
      manualAborts.delete(run._id);
      await persistRuns();
    }
  }

  // ---------------------------------------------------------------- Maintenance

  async function resetHistory() {
    const signals = await store.listSignals();
    const metaCount = (
      await Promise.all(UNIVERSES.flatMap((u) => TIMEFRAMES.map((tf) => store.getScanMeta(u, tf))))
    ).filter(Boolean).length;
    const manual = (await runs()).size;
    await store.saveSignals([]);
    await store.clearScanMeta();
    (await runs()).clear();
    await store.putPreset(MANUAL_RUNS_KEY, []);
    return { ok: true, deletedRuns: metaCount + manual, deletedSignals: signals.length };
  }

  async function delistedGroups() {
    const signals = await store.listSignals();
    const groups: Array<{ universe: Universe; symbols: string[] }> = [];
    for (const universe of UNIVERSES) {
      const entries = deps.universe[universe];
      if (!entries.length) continue;
      const live = new Set(entries.map((e) => e.yahoo));
      const seen = [...new Set(signals.filter((s) => s.universe === universe).map((s) => s.yahooTicker))];
      const symbols = seen.filter((t) => !live.has(t)).sort();
      if (symbols.length) groups.push({ universe, symbols });
    }
    return groups;
  }

  async function summarizeDelisted(groups: Array<{ universe: Universe; symbols: string[] }>) {
    const signals = await store.listSignals();
    let symbols = 0;
    let records = 0;
    let closed = 0;
    const sample: string[] = [];
    for (const group of groups) {
      const set = new Set(group.symbols);
      const rows = signals.filter((s) => s.universe === group.universe && set.has(s.yahooTicker));
      symbols += group.symbols.length;
      records += rows.length;
      closed += rows.filter((s) => s.status === 'closed').length;
      sample.push(...group.symbols);
    }
    return { symbols, records, closed, active: records - closed, sample: sample.slice(0, 20) };
  }

  async function purgeDelisted() {
    const groups = await delistedGroups();
    const summaryOut = await summarizeDelisted(groups);
    const drop = new Set(groups.flatMap((g) => g.symbols.map((s) => `${g.universe}|${s}`)));
    const signals = await store.listSignals();
    const kept = signals.filter((s) => !drop.has(`${s.universe}|${s.yahooTicker}`));
    if (kept.length !== signals.length) await store.saveSignals(kept);
    return { ok: true, deletedSignals: signals.length - kept.length, ...summaryOut };
  }

  // ---------------------------------------------------------------- Charts

  async function barsForChart(ticker: string, tf: BarsTf, isListed: boolean): Promise<OhlcSeries | null> {
    const cached = await store.getBars(ticker, tf);
    if (isListed && cached?.bars.length) return cached.bars;
    if (cached?.bars.length && Date.now() - Date.parse(cached.fetchedAt) < CHART_BARS_MAX_AGE_MS) return cached.bars;
    const pulled = await fetchBars(ticker, tf);
    if (pulled.bars?.length) {
      await store.putBars({
        yahooTicker: ticker,
        tf,
        bars: pulled.bars,
        fetchedAt: new Date().toISOString(),
        companyName: cached?.companyName ?? pulled.companyName,
      });
      return pulled.bars;
    }
    return cached?.bars ?? null;
  }

  /** Daily bars feed the watermark's day change and D line. Stored once, refreshed in the background. */
  async function dailyBars(ticker: string): Promise<OhlcSeries | null> {
    const cached = await store.getBars(ticker, 'Daily');
    if (!cached) {
      const pulled = await fetchBars(ticker, 'Daily').catch(() => ({ bars: null, companyName: null }));
      if (!pulled.bars) return null;
      await store.putBars({ yahooTicker: ticker, tf: 'Daily', bars: pulled.bars, fetchedAt: new Date().toISOString(), companyName: pulled.companyName });
      return pulled.bars;
    }
    if (Date.now() - Date.parse(cached.fetchedAt) > DAILY_REFRESH_MS && !dailyRefreshing.has(ticker)) {
      dailyRefreshing.add(ticker);
      void fetchBars(ticker, 'Daily')
        .then(async (pulled) => {
          if (pulled.bars) {
            await store.putBars({ yahooTicker: ticker, tf: 'Daily', bars: pulled.bars, fetchedAt: new Date().toISOString(), companyName: pulled.companyName });
          }
        })
        .catch(() => undefined)
        .finally(() => dailyRefreshing.delete(ticker));
    }
    return cached.bars;
  }

  async function chart(ticker: string, q: URLSearchParams) {
    const tf: UserTf = q.get('tf') === 'Monthly' ? 'Monthly' : 'Weekly';
    const known = listed.get(ticker);
    const series = await barsForChart(ticker, tf, Boolean(known));
    if (!series?.length) throw new HttpError(404, `no bars for ${ticker}`);
    const num = (raw: string | null) => {
      if (raw == null || raw === '') return undefined;
      const v = Number(raw);
      return Number.isFinite(v) ? v : undefined;
    };
    const chartParams: Partial<IndicatorParams> = {};
    const assign = <K extends keyof IndicatorParams>(key: K, raw: string | null) => {
      const v = num(raw);
      if (v !== undefined) chartParams[key] = v as IndicatorParams[K];
    };
    assign('len_fast', q.get('lenFast'));
    assign('len_slow', q.get('lenSlow'));
    assign('length_major', q.get('lengthMajor'));
    assign('lookback', q.get('lookback'));
    assign('multiplier', q.get('multiplier'));
    assign('bb_length', q.get('bbLength'));
    assign('bb_mult', q.get('bbMult'));
    const asOfRaw = q.get('asOf') ?? '';
    const asOf = /^\d{4}-\d{2}-\d{2}$/.test(asOfRaw) ? asOfRaw : undefined;
    const other: UserTf = tf === 'Weekly' ? 'Monthly' : 'Weekly';
    const [daily, otherBars] = await Promise.all([
      dailyBars(ticker),
      store.getBars(ticker, other).then((row) => row?.bars ?? null),
    ]);
    const cachedRow = await store.getBars(ticker, tf);
    const tracked = (await store.listSignals()).find((row) => row.yahooTicker === ticker);
    try {
      const built = buildChartPayload(
        {
          yahooTicker: ticker,
          tvSymbol: known?.entry.tv ?? tracked?.tvSymbol ?? ticker,
          companyName: known?.entry.name ?? tracked?.companyName ?? cachedRow?.companyName ?? ticker,
          tf,
          bars: series,
          dailyBars: daily,
          weeklyBars: tf === 'Weekly' ? series : otherBars,
          monthlyBars: tf === 'Monthly' ? series : otherBars,
        },
        {
          minRr: num(q.get('minRr')),
          useLastHlSl: q.get('useLastHlSl') ? q.get('useLastHlSl') === 'true' : undefined,
          riskPerTrade: num(q.get('riskPerTrade')),
          noRrReq: q.get('noRrReq') ? q.get('noRrReq') === 'true' : undefined,
          asOf,
          chartParams,
          fullSeries: q.get('fullSeries') === '1' || q.get('fullSeries') === 'true',
        },
      );
      if (!asOf) await syncTrackedAge(ticker, tf, built.lastBarDate, built.age);
      return built.payload;
    } catch (err) {
      if (err instanceof ChartError) throw new HttpError(404, err.message);
      throw err;
    }
  }

  /** `InstrumentsService.syncTrackedAge`: the live chart realigns the tracked row's age. */
  async function syncTrackedAge(ticker: string, tf: UserTf, barAsOf: string, age: SignalAge) {
    const doc = (await store.listSignals()).find(
      (row) =>
        row.yahooTicker === ticker &&
        row.tf === tf &&
        row.status === 'active' &&
        !row.imported &&
        !row.provisionalClose &&
        (!row.lastSeenAsOf || row.lastSeenAsOf <= barAsOf),
    );
    if (!doc) return;
    const signalValid = age.barsSinceValid != null;
    if (
      doc.barsSinceValid === age.barsSinceValid &&
      (doc.validSinceAsOf ?? null) === (age.validSinceAsOf ?? null) &&
      doc.signalValid === signalValid
    ) {
      return;
    }
    await store.upsertSignals([
      { ...doc, barsSinceValid: age.barsSinceValid, validSinceAsOf: age.validSinceAsOf, signalValid },
    ]);
  }

  async function tickerInterest(ticker: string, body?: unknown) {
    const map = (await store.getPreset<Record<string, Interest>>(TICKER_INTEREST_KEY)) ?? {};
    const key = ticker.toUpperCase();
    if (body !== undefined) {
      const value = (body as { interest?: unknown } | null)?.interest;
      const interest: Interest | null = value === 'interested' || value === 'not_interested' ? value : null;
      const next = { ...map };
      if (interest) next[key] = interest;
      else delete next[key];
      await store.putPreset(TICKER_INTEREST_KEY, next);
      return { yahooTicker: key, interest, interestRank: interestRank(interest) };
    }
    const interest = map[key] ?? null;
    return { yahooTicker: key, interest, interestRank: interestRank(interest) };
  }

  // ---------------------------------------------------------------- Router

  async function route(method: string, path: string, body: unknown): Promise<unknown> {
    const url = new URL(path, 'https://device.local');
    const q = url.searchParams;
    const parts = url.pathname.replace(/^\/+|\/+$/g, '').split('/').map(decodeURIComponent);
    const [head, a, b] = parts;
    const M = method.toUpperCase();

    if (head === 'health') return { ok: true };

    if (head === 'results') {
      if (M === 'GET' && parts.length === 1) return results(q);
      if (M === 'GET' && a === 'summary') return summary();
      if (M === 'GET' && a === 'lookup') return lookup(q);
      if (M === 'GET' && a === 'signal' && b) return toResultRow(await findSignal(b));
      if (M === 'PATCH' && a && b === 'interest') return patchInterest(a, body);
    }

    if (head === 'history') {
      if (M === 'GET' && parts.length === 1) return history(q);
      if (M === 'GET' && a === 'trades') return trades(q);
      if (a === 'rebuild') return M === 'POST' ? startRebuild() : { ...rebuildState };
      if (a?.startsWith('enrich-')) throw new HttpError(404, FUNDAMENTALS_MESSAGE);
    }

    if (head === 'settings') {
      if (M === 'GET') return webSettings(await settings());
      if (M === 'PUT') return putSettings(body);
    }

    if (head === 'scans') {
      if (M === 'POST' && a === 'run-now') return runNow(body);
      if (M === 'POST' && parts.length === 1) return startManual(body);
      if (M === 'DELETE' && a === 'history') return resetHistory();
      if (a === 'delisted') {
        if (M === 'GET') return summarizeDelisted(await delistedGroups());
        if (M === 'DELETE') return purgeDelisted();
      }
      if (M === 'GET' && a === 'defaults') {
        return { source: 'Stocks', tf: 'Weekly', direction: 'buy', minRr: 0, noRrReq: true, useLastHlSl: true, newOnly: false };
      }
      if (M === 'GET' && parts.length === 1) {
        return [...(await runs()).values()].map(publicRun).sort((x, y) => y.createdAt.localeCompare(x.createdAt));
      }
      if (a) {
        if (M === 'POST' && b === 'cancel') {
          manualAborts.get(a)?.abort();
          if (scanAbort && !manualAborts.has(a)) scanAbort.abort();
          return { ok: true };
        }
        const run = await findRun(a);
        if (M === 'GET' && !b) return publicRun(run);
        if (M === 'GET' && b === 'progress') return run.progress;
        if (M === 'GET' && b === 'signals') {
          const onlyStrong = q.get('onlyStrong') === 'true';
          const rows = run.signals
            .filter((s) => !onlyStrong || s.isStrong)
            .sort((x, y) => Number(y.isStrong) - Number(x.isStrong) || (y.rr ?? -Infinity) - (x.rr ?? -Infinity));
          return { run: publicRun(run), count: rows.length, rows, newSymbols: run.newSymbols };
        }
        if (M === 'GET' && b === 'rejections') {
          const limit = Math.min(int(q.get('limit')) ?? 300, 2000);
          return { rows: run.rejections.slice(0, limit), reasonCounts: run.reasonCounts, total: run.counters.rejected };
        }
      }
    }

    if (head === 'instruments') {
      if (a && a.startsWith('fundamentals')) throw new HttpError(404, FUNDAMENTALS_MESSAGE);
      if (a && b === 'chart' && M === 'GET') return chart(a, q);
      if (a && b === 'interest') return tickerInterest(a, M === 'PATCH' ? (body ?? {}) : undefined);
      if (a && (b === 'fundamentals' || b === 'dcf')) throw new HttpError(404, FUNDAMENTALS_MESSAGE);
    }

    if (head === 'presets' && a) {
      if (M === 'GET') return (await store.getPreset(a)) ?? {};
      if (M === 'PUT') {
        await store.putPreset(a, body ?? {});
        return { ok: true, key: a, data: body ?? {} };
      }
    }

    if (head === 'universe' && a === 'summary') {
      const stocks = deps.universe.Stocks.length;
      const etf = deps.universe.ETF.length;
      return { stocks, etf, total: listed.size };
    }

    throw new HttpError(404, `Cannot ${M} /api${url.pathname}`);
  }

  return {
    async handle(method: string, path: string, rawBody?: string | null): Promise<ApiResponse> {
      try {
        const body = rawBody ? JSON.parse(rawBody) : undefined;
        return json(200, await route(method, path, body));
      } catch (err) {
        if (err instanceof HttpError) {
          return json(err.status, { message: err.message, statusCode: err.status });
        }
        return json(500, { message: (err as Error).message ?? 'Internal error', statusCode: 500 });
      }
    },
  };
}

function parseUniverse(value: string | null): Universe {
  return UNIVERSES.includes(value as Universe) ? (value as Universe) : 'Stocks';
}

function parseTf(value: string | null): UserTf {
  return value === 'Monthly' ? 'Monthly' : 'Weekly';
}

function parseHistoryTf(value: string | null): HistoryTf {
  return value === 'Weekly' || value === 'Monthly' || value === 'All' ? value : 'All';
}

/** `parseHistoryGroupBy`: `Daily` is the old chip name; anything unknown is `Day`. */
function parseGroupBy(value: string | null): HistoryGroupBy {
  if (value === 'Weekly' || value === 'Monthly') return value;
  return 'Day';
}

function parseRange(value: string | null): HistoryRange {
  return HISTORY_RANGES.includes(value as HistoryRange) ? (value as HistoryRange) : 'all';
}

function parseDir(value: string | null): SortDir {
  return value === 'asc' ? 'asc' : 'desc';
}

function int(value: string | null): number | undefined {
  if (value == null || value === '') return undefined;
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

