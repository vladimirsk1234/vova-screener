/** Mirrors `HistoryRebuildService`: replay the close-scan ledger over cached bars, insert missing closes. */
import { barPeriodKey } from '../../../apps/api/src/scans/period.ts';
import { runCloseLedger } from '../../engine/src/sequenceVova.ts';
import { shortSymbol, type ParsedEntry } from '../../engine/src/tickers.ts';
import type { CloseTrade, OhlcSeries } from '../../engine/src/types.ts';
import { newId } from './id';
import { computePnl, finiteOrNull, holdPeriods, round2, sharesFromRisk } from './money';
import { TIMEFRAMES, UNIVERSES, type TrackedSignal, type Universe, type UserTf } from './types';

export type RebuildStatus = {
  status: 'idle' | 'running' | 'done' | 'failed';
  startedAt: string | null;
  finishedAt: string | null;
  error: string | null;
  progress: { universe: Universe | null; tf: UserTf | null; symbolsDone: number; symbolsTotal: number };
  counts: { inserted: number; skipped: number; noBars: number; symbols: number };
};

export function emptyRebuildStatus(): RebuildStatus {
  return {
    status: 'idle',
    startedAt: null,
    finishedAt: null,
    error: null,
    progress: { universe: null, tf: null, symbolsDone: 0, symbolsTotal: 0 },
    counts: { inserted: 0, skipped: 0, noBars: 0, symbols: 0 },
  };
}

export async function rebuildHistory(opts: {
  signals: TrackedSignal[];
  universe: Record<Universe, ParsedEntry[]>;
  barsFor: (ticker: string, tf: UserTf) => Promise<OhlcSeries | null>;
  maxRiskUsd: number;
  state: RebuildStatus;
  yieldEvery?: () => Promise<void>;
}): Promise<TrackedSignal[]> {
  const { state } = opts;
  const next = [...opts.signals];
  for (const universe of UNIVERSES) {
    for (const tf of TIMEFRAMES) {
      const entries = opts.universe[universe];
      state.progress = { universe, tf, symbolsDone: 0, symbolsTotal: entries.length };
      const recorded = new Set<string>();
      for (const row of next) {
        if (row.universe !== universe || row.tf !== tf) continue;
        if (row.openedAsOf) recorded.add(`${row.yahooTicker}@entry@${row.openedAsOf}`);
        if (row.exitDate) recorded.add(`${row.yahooTicker}@exit@${row.exitDate}`);
      }
      for (const entry of entries) {
        state.counts.symbols += 1;
        state.progress.symbolsDone += 1;
        const bars = await opts.barsFor(entry.yahoo, tf);
        if (!bars?.length) {
          state.counts.noBars += 1;
          continue;
        }
        const ledger = runCloseLedger(bars, {
          min_rr: 0,
          no_rr_req: true,
          use_last_hl_sl: true,
          risk_dollars: opts.maxRiskUsd,
        });
        if (!ledger) continue;
        for (const trade of ledger.trades) {
          if (trade.exit_index == null || !trade.exit_date) continue;
          const entered = `${entry.yahoo}@entry@${trade.entry_date}`;
          const exited = `${entry.yahoo}@exit@${trade.exit_date}`;
          if (recorded.has(entered) || recorded.has(exited)) {
            state.counts.skipped += 1;
            continue;
          }
          recorded.add(entered);
          recorded.add(exited);
          next.push(closedFromTrade(entry, trade, universe, tf, opts.maxRiskUsd));
          state.counts.inserted += 1;
        }
        if (opts.yieldEvery && state.progress.symbolsDone % 50 === 0) await opts.yieldEvery();
      }
    }
  }
  return next;
}

function closedFromTrade(
  entry: ParsedEntry,
  trade: CloseTrade,
  universe: Universe,
  tf: UserTf,
  maxRiskUsd: number,
): TrackedSignal {
  const tv = entry.tv || entry.yahoo;
  const price = round2(trade.entry_price);
  const sl = finiteOrNull(trade.entry_sl);
  const exitDate = trade.exit_date as string;
  const exitPrice = round2(trade.exit_price);
  const shares = sharesFromRisk(price, sl, maxRiskUsd);
  const pnl = computePnl(price, sl, shares, exitPrice);
  return {
    id: newId(),
    yahooTicker: entry.yahoo,
    symbol: shortSymbol(tv),
    tvSymbol: tv,
    companyName: entry.name ?? entry.yahoo,
    universe,
    tf,
    status: 'closed',
    provisional: false,
    provisionalClose: false,
    signalValid: false,
    imported: false,
    openedPeriodKey: barPeriodKey(tf, trade.entry_date),
    openedAsOf: trade.entry_date,
    openedAt: new Date(`${trade.entry_date}T12:00:00Z`).toISOString(),
    entry: price,
    tp: finiteOrNull(trade.entry_tp),
    sl,
    rrAtEntry: finiteOrNull(trade.entry_rr),
    shares,
    riskUsd: maxRiskUsd,
    lastSeenPeriodKey: barPeriodKey(tf, exitDate),
    lastSeenAsOf: exitDate,
    lastPrice: exitPrice,
    lastRr: null,
    barsSinceValid: null,
    validSinceAsOf: null,
    isStrong: false,
    unrealizedUsd: null,
    unrealizedR: null,
    unrealizedPct: null,
    closedPeriodKey: barPeriodKey(tf, exitDate),
    closedAt: new Date().toISOString(),
    exitDate,
    exitPrice,
    exitReason: 'sell_to_close',
    pnlUsd: pnl.usd,
    pnlR: pnl.r,
    pnlPct: pnl.pct,
    holdPeriods: holdPeriods(tf, trade.entry_date, exitDate),
    interest: null,
    interestRank: 1,
  };
}
