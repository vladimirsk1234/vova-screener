/** On-device scan. Bars come from Yahoo and stay in the local store. */
import { evaluateClose, evaluateSymbol, type EvaluateParams } from '../../engine/src/evaluate.ts';
import { parseListText, type ParsedEntry } from '../../engine/src/tickers.ts';
import type { OhlcSeries } from '../../engine/src/types.ts';
import { barPeriodKey, isPeriodClosed, periodKey } from '../../../apps/api/src/scans/period.ts';
import type { ScreenerStore } from './store';
import { applyScanResult, type ScanOutcome } from './tracker';
import type { AppSettings, Rejection, ScanMeta, Universe, UserTf } from './types';
import { fetchYahooOhlc } from './yahoo';

const UNEVALUATED = new Set(['NO_DATA', 'INSUFFICIENT_DATA']);
const DEFAULT_MAX_AGE_MS = 6 * 60 * 60 * 1000;

export type ScanProgress = {
  phase: 'running' | 'completed' | 'cancelled' | 'failed';
  evaluated: number;
  total: number;
  signals: number;
  closes: number;
  rejected: number;
  message: string;
};

export function parseUniverse(stocksText: string, etfText: string): Record<Universe, ParsedEntry[]> {
  return {
    Stocks: parseListText(stocksText).entries,
    ETF: parseListText(etfText).entries,
  };
}

export function evaluateBars(input: {
  bars: OhlcSeries | null;
  yahooTicker: string;
  tvSymbol: string;
  companyName: string;
  tf: UserTf;
  riskPerTrade: number;
}): ScanOutcome & { reason: string | null; detail: Rejection['detail'] } {
  const params: EvaluateParams = {
    direction: 'buy',
    minRr: 0,
    riskPerTrade: input.riskPerTrade,
    noRrReq: true,
    useLastHlSl: true,
    newOnly: false,
    tf: input.tf,
  };
  const evaluation = evaluateSymbol({
    bars: input.bars,
    yahooTicker: input.yahooTicker,
    tvSymbol: input.tvSymbol,
    companyName: input.companyName,
    params,
  });
  const close = evaluateClose({
    bars: input.bars,
    yahooTicker: input.yahooTicker,
    tvSymbol: input.tvSymbol,
    companyName: input.companyName,
    params,
  });
  const buy = evaluation.status === 'signal' && evaluation.signal.kind === 'buy' ? evaluation.signal : null;
  const reason = evaluation.status === 'signal' ? null : evaluation.reason;
  return {
    yahooTicker: input.yahooTicker,
    symbol: buy?.symbol ?? close?.symbol ?? input.yahooTicker,
    tvSymbol: input.tvSymbol,
    companyName: input.companyName,
    buy,
    sell: close,
    unevaluated: reason != null && UNEVALUATED.has(reason),
    reason,
    detail: evaluation.status === 'rejected' ? (evaluation.detail ?? null) : null,
  };
}

export async function scanUniverse(opts: {
  store: ScreenerStore;
  entries: ParsedEntry[];
  universe: Universe;
  tf: UserTf;
  settings: AppSettings;
  signal?: AbortSignal;
  onProgress?: (progress: ScanProgress) => void;
  concurrency?: number;
  maxAgeMs?: number;
  fetchBars?: typeof fetchYahooOhlc;
  now?: Date;
}): Promise<ScanProgress> {
  const started = opts.now ?? new Date();
  const confirmed = isPeriodClosed(opts.tf, started);
  const clockKey = periodKey(opts.tf, started);
  const fetchBars = opts.fetchBars ?? fetchYahooOhlc;
  const maxAge = opts.maxAgeMs ?? DEFAULT_MAX_AGE_MS;
  const queue = [...opts.entries];
  const total = queue.length;
  const outcomes: Array<ScanOutcome & { reason: string | null }> = [];
  const periods = new Map<string, { count: number; newest: string }>();
  let evaluated = 0;
  let signals = 0;
  let closes = 0;
  let rejected = 0;
  let asOf: string | null = null;

  const publish = (phase: ScanProgress['phase'], message: string) => {
    opts.onProgress?.({ phase, evaluated, total, signals, closes, rejected, message });
  };
  publish('running', total ? `Scanning 0/${total}` : 'Nothing to scan');

  const worker = async () => {
    while (queue.length) {
      if (opts.signal?.aborted) return;
      const entry = queue.shift();
      if (!entry) return;
      const cached = await opts.store.getBars(entry.yahoo, opts.tf);
      const fresh = cached && Date.now() - Date.parse(cached.fetchedAt) < maxAge ? cached : null;
      let bars = fresh?.bars ?? null;
      let companyName = fresh?.companyName || entry.name || entry.yahoo;
      if (!fresh) {
        const pulled = await fetchBars(entry.yahoo, opts.tf, { signal: opts.signal });
        bars = pulled.bars;
        if (pulled.companyName) companyName = pulled.companyName;
        if (bars) {
          await opts.store.putBars({
            yahooTicker: entry.yahoo,
            tf: opts.tf,
            bars,
            fetchedAt: new Date().toISOString(),
            companyName,
          });
        }
      }
      const outcome = evaluateBars({
        bars,
        yahooTicker: entry.yahoo,
        tvSymbol: entry.tv,
        companyName,
        tf: opts.tf,
        riskPerTrade: opts.settings.maxRiskUsd,
      });
      outcomes.push(outcome);
      if (outcome.buy) signals += 1;
      if (outcome.sell) closes += 1;
      if (!outcome.buy && outcome.reason) rejected += 1;
      if (bars?.length) {
        const barDate = bars[bars.length - 1].date;
        if (!asOf || barDate < asOf) asOf = barDate;
        const key = barPeriodKey(opts.tf, barDate);
        const slot = periods.get(key);
        if (!slot) periods.set(key, { count: 1, newest: barDate });
        else {
          slot.count += 1;
          if (barDate > slot.newest) slot.newest = barDate;
        }
      }
      evaluated += 1;
      if (evaluated % 5 === 0 || evaluated === total) {
        publish('running', `Scanned ${evaluated}/${total} · ${signals} signals`);
      }
    }
  };

  try {
    const workers = Math.max(1, Math.min(opts.concurrency ?? 4, queue.length || 1));
    await Promise.all(Array.from({ length: workers }, () => worker()));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const failed: ScanProgress = { phase: 'failed', evaluated, total, signals, closes, rejected, message };
    opts.onProgress?.(failed);
    return failed;
  }

  const cancelled = Boolean(opts.signal?.aborted);
  const existing = await opts.store.listSignals();
  const barMap = new Map<string, OhlcSeries>();
  const tickers = new Set<string>(outcomes.map((o) => o.yahooTicker));
  for (const row of existing) {
    if (row.universe === opts.universe && row.tf === opts.tf && row.status === 'active') tickers.add(row.yahooTicker);
  }
  for (const ticker of tickers) {
    const cached = await opts.store.getBars(ticker, opts.tf);
    if (cached?.bars) barMap.set(ticker, cached.bars);
  }
  const applied = applyScanResult({
    signals: existing,
    universe: opts.universe,
    tf: opts.tf,
    periodKey: clockKey,
    confirmed,
    maxRiskUsd: opts.settings.maxRiskUsd,
    outcomes,
    complete: !cancelled,
    barsFor: (ticker) => barMap.get(ticker) ?? null,
  });
  await opts.store.saveSignals(applied.signals);
  if (!cancelled) {
    const meta: ScanMeta = {
      periodKey: clockKey,
      asOf,
      newestAsOf: consensusAsOf(periods),
      finishedAt: new Date().toISOString(),
      running: false,
      status: 'completed',
    };
    await opts.store.putScanMeta(opts.universe, opts.tf, meta);
  }
  const done: ScanProgress = {
    phase: cancelled ? 'cancelled' : 'completed',
    evaluated,
    total,
    signals,
    closes,
    rejected,
    message: cancelled ? `Cancelled after ${evaluated}/${total}` : `Done · ${signals} signals · ${closes} closes`,
  };
  opts.onProgress?.(done);
  return done;
}

function consensusAsOf(periods: Map<string, { count: number; newest: string }>): string | null {
  let best: { key: string; count: number; newest: string } | null = null;
  for (const [key, slot] of periods) {
    if (!best || slot.count > best.count || (slot.count === best.count && key > best.key)) {
      best = { key, ...slot };
    }
  }
  return best?.newest ?? null;
}
