/** Replay the close-scan ledger over cached bars and insert missing closed trades. */
import { runCloseLedger, shortSymbol, type CloseTrade } from '../../engine/src/index.ts';
import { barPeriodKey } from '../../../apps/api/src/scans/period.ts';
import { computePnl, finiteOrNull, holdPeriods, round2, sharesFromRisk } from './money';
import type { ScreenerStore } from './store';
import type { TrackedSignal, Universe, UserTf } from './types';

export type RebuildCounts = { inserted: number; skipped: number; noBars: number; symbols: number };

export async function rebuildHistory(
  store: ScreenerStore,
  opts: { maxRiskUsd: number; onProgress?: (counts: RebuildCounts) => void } = { maxRiskUsd: 100 },
): Promise<RebuildCounts> {
  const counts: RebuildCounts = { inserted: 0, skipped: 0, noBars: 0, symbols: 0 };
  const signals = await store.listSignals();
  const keys = await store.listBarKeys();
  const recorded = new Set<string>();
  for (const row of signals) {
    if (row.openedAsOf) recorded.add(entryKey(row.yahooTicker, row.openedAsOf));
    if (row.exitDate) recorded.add(exitKey(row.yahooTicker, row.exitDate));
  }
  const next = [...signals];
  for (const key of keys) {
    counts.symbols += 1;
    const cached = await store.getBars(key.yahooTicker, key.tf);
    if (!cached?.bars.length) {
      counts.noBars += 1;
      continue;
    }
    const ledger = runCloseLedger(cached.bars, {
      min_rr: 0,
      no_rr_req: true,
      use_last_hl_sl: true,
      risk_dollars: opts.maxRiskUsd,
    });
    if (!ledger) continue;
    const universe = guessUniverse(key.yahooTicker, signals);
    for (const trade of ledger.trades) {
      if (trade.exit_index == null || !trade.exit_date) continue;
      const entered = entryKey(key.yahooTicker, trade.entry_date);
      const exited = exitKey(key.yahooTicker, trade.exit_date);
      if (recorded.has(entered) || recorded.has(exited)) {
        counts.skipped += 1;
        continue;
      }
      recorded.add(entered);
      recorded.add(exited);
      next.push(closedFromTrade(cached, trade, universe, key.tf, opts.maxRiskUsd));
      counts.inserted += 1;
    }
    opts.onProgress?.(counts);
  }
  if (counts.inserted) await store.saveSignals(next);
  return counts;
}

function guessUniverse(ticker: string, signals: TrackedSignal[]): Universe {
  return signals.find((s) => s.yahooTicker === ticker)?.universe ?? 'Stocks';
}

function closedFromTrade(
  cached: { yahooTicker: string; companyName: string | null },
  trade: CloseTrade,
  universe: Universe,
  tf: UserTf,
  maxRiskUsd: number,
): TrackedSignal {
  const entry = round2(trade.entry_price);
  const sl = finiteOrNull(trade.entry_sl);
  const exitDate = trade.exit_date as string;
  const exitPrice = round2(trade.exit_price);
  const shares = sharesFromRisk(entry, sl, maxRiskUsd);
  const pnl = computePnl(entry, sl, shares, exitPrice);
  const symbol = shortSymbol(cached.yahooTicker);
  return {
    id: globalThis.crypto.randomUUID(),
    yahooTicker: cached.yahooTicker,
    symbol,
    tvSymbol: symbol,
    companyName: cached.companyName || symbol,
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
    entry,
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

function entryKey(ticker: string, date: string): string {
  return `${ticker}@entry@${date}`;
}

function exitKey(ticker: string, date: string): string {
  return `${ticker}@exit@${date}`;
}
