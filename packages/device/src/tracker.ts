/**
 * Turns a finished scan into tracked signals.
 * Same rules as the server tracker, without FMP premium stamps.
 */
import type { BuySignal, SellSignal } from '../../engine/src/evaluate.ts';
import { runCloseLedger } from '../../engine/src/sequenceVova.ts';
import { runStructureOverlay } from '../../engine/src/sequenceVovaFull.ts';
import type { CloseTrade, OhlcSeries } from '../../engine/src/types.ts';
import { barPeriodKey } from '../../../apps/api/src/scans/period.ts';
import { newId } from './id';
import type { ExitReason } from './types';
import { computePnl, finiteOrNull, holdPeriods, interestRank, round2, sharesFromRisk } from './money';
import type { TrackedSignal, Universe, UserTf } from './types';

export type ScanOutcome = {
  yahooTicker: string;
  symbol: string;
  tvSymbol: string;
  companyName: string;
  buy: BuySignal | null;
  sell: SellSignal | null;
  /** NO_DATA / INSUFFICIENT_DATA — the scan could not price the symbol. */
  unevaluated: boolean;
};

export type ApplyScanInput = {
  signals: TrackedSignal[];
  universe: Universe;
  tf: UserTf;
  periodKey: string;
  confirmed: boolean;
  maxRiskUsd: number;
  outcomes: ScanOutcome[];
  /** False when the pass stopped early. Names it never reached stay as they were. */
  complete: boolean;
  barsFor: (yahooTicker: string) => OhlcSeries | null;
  now?: string;
};

export type TrackerReport = {
  opened: number;
  refreshed: number;
  closed: number;
  pendingClose: number;
  hidden: number;
  adopted: number;
  dropped: number;
};

type Replay = { trades: CloseTrade[]; open: CloseTrade | null; breaks: Exit[] };
type Exit = { date: string; price: number; reason: ExitReason };

const LEDGER = { min_rr: 0, no_rr_req: true, use_last_hl_sl: true };

export function applyScanResult(input: ApplyScanInput): { signals: TrackedSignal[]; report: TrackerReport } {
  const now = input.now ?? new Date().toISOString();
  const report: TrackerReport = {
    opened: 0,
    refreshed: 0,
    closed: 0,
    pendingClose: 0,
    hidden: 0,
    adopted: 0,
    dropped: 0,
  };
  const mine = input.signals.filter((s) => s.universe === input.universe && s.tf === input.tf);
  const rest = input.signals.filter((s) => !(s.universe === input.universe && s.tf === input.tf));
  const byTicker = new Map(input.outcomes.map((o) => [o.yahooTicker, o]));
  const active = mine.filter((s) => s.status === 'active');
  const closed = mine.filter((s) => s.status !== 'active').map((s) => ({ ...s }));
  const nextActive: TrackedSignal[] = [];
  const known = new Set<string>();

  const replayCache = new Map<string, Replay | null>();
  const replayOf = (ticker: string): Replay | null => {
    if (replayCache.has(ticker)) return replayCache.get(ticker) ?? null;
    const bars = input.barsFor(ticker);
    const replay = bars?.length ? buildReplay(bars, input.maxRiskUsd) : null;
    replayCache.set(ticker, replay);
    return replay;
  };

  for (const doc of active) {
    known.add(doc.yahooTicker);
    const outcome = byTicker.get(doc.yahooTicker);
    if (!outcome && !input.complete) {
      nextActive.push(doc);
      continue;
    }
    const snapshot = outcome?.buy ? buySnapshot(outcome.buy) : null;
    const unevaluated = outcome?.unevaluated === true;
    const evaluated = input.complete ? !unevaluated : Boolean(outcome) && !unevaluated;

    if (doc.provisional) {
      if (snapshot && outcome) {
        nextActive.push(refresh(doc, snapshot, input, null, now));
        report.refreshed += 1;
      } else if (input.confirmed && !unevaluated) {
        report.dropped += 1;
      } else {
        nextActive.push(doc);
      }
      continue;
    }

    const since = doc.openedAsOf || doc.openedAt.slice(0, 10);
    const replay = replayOf(doc.yahooTicker);
    const trade = tradeOf(replay, since);
    const exit = exitOf(replay, since);
    if (exit) {
      const pending = !input.confirmed && barPeriodKey(input.tf, exit.date) === input.periodKey;
      const closedDoc = closeDoc(doc, trade, exit, input, pending, now);
      if (pending) {
        nextActive.push(closedDoc);
        report.pendingClose += 1;
        continue;
      }
      closed.push(closedDoc);
      report.closed += 1;
      if (snapshot && snapshot.asOf > exit.date) {
        nextActive.push(newDocument(snapshot, replayOf(doc.yahooTicker)?.open ?? null, input, now));
        report.opened += 1;
      }
      continue;
    }

    if (!snapshot) {
      const updated = missing(doc, trade, input, evaluated);
      nextActive.push(updated);
      if (evaluated && doc.signalValid !== false) report.hidden += 1;
      continue;
    }
    nextActive.push(refresh(doc, snapshot, input, trade, now));
    report.refreshed += 1;
  }

  for (const outcome of input.outcomes) {
    if (!outcome.buy || known.has(outcome.yahooTicker)) continue;
    const snapshot = buySnapshot(outcome.buy);
    nextActive.push(newDocument(snapshot, replayOf(outcome.yahooTicker)?.open ?? null, input, now));
    report.opened += 1;
    known.add(outcome.yahooTicker);
  }

  const recorded = new Set<string>();
  for (const row of [...closed, ...nextActive]) {
    if (row.exitDate) recorded.add(`${row.yahooTicker}@${row.exitDate}`);
  }
  for (const outcome of input.outcomes) {
    const sell = outcome.sell;
    if (!sell) continue;
    if (known.has(outcome.yahooTicker)) continue;
    if (recorded.has(`${sell.yahooTicker}@${sell.exitAsOf}`)) continue;
    const pending = !input.confirmed && barPeriodKey(input.tf, sell.exitAsOf) === input.periodKey;
    const adopted = adoptedDocument(sell, input, pending, now);
    if (pending) nextActive.push(adopted);
    else closed.push(adopted);
    recorded.add(`${sell.yahooTicker}@${sell.exitAsOf}`);
    report.adopted += 1;
    if (pending) report.pendingClose += 1;
    else report.closed += 1;
  }

  return { signals: [...rest, ...nextActive, ...closed], report };
}

/** Re-size every open position when Max risk changes. Closed rows keep the size they closed at. */
export function resizeActive(signals: TrackedSignal[], maxRiskUsd: number): TrackedSignal[] {
  return signals.map((doc) => {
    if (doc.status !== 'active') return doc;
    const shares = sharesFromRisk(doc.entry, doc.sl, maxRiskUsd);
    const pnl = computePnl(doc.entry, doc.sl, shares, doc.lastPrice ?? doc.entry);
    return {
      ...doc,
      shares,
      riskUsd: maxRiskUsd,
      unrealizedUsd: pnl.usd,
      unrealizedR: pnl.r,
      unrealizedPct: pnl.pct,
    };
  });
}

export function setInterest(signals: TrackedSignal[], id: string, interest: TrackedSignal['interest']): TrackedSignal[] {
  return signals.map((doc) =>
    doc.id === id ? { ...doc, interest, interestRank: interestRank(interest) } : doc,
  );
}

type Snapshot = {
  yahooTicker: string;
  symbol: string;
  tvSymbol: string;
  companyName: string;
  entry: number;
  tp: number | null;
  sl: number | null;
  rr: number | null;
  isStrong: boolean;
  barsSinceValid: number | null;
  validSinceAsOf: string | null;
  asOf: string;
};

function buySnapshot(buy: BuySignal): Snapshot {
  return {
    yahooTicker: buy.yahooTicker,
    symbol: buy.symbol,
    tvSymbol: buy.tvSymbol,
    companyName: buy.companyName,
    entry: buy.entry,
    tp: Number.isFinite(buy.tp) ? buy.tp : null,
    sl: Number.isFinite(buy.sl) ? buy.sl : null,
    rr: buy.rr,
    isStrong: buy.isStrong,
    barsSinceValid: buy.barsSinceValid,
    validSinceAsOf: buy.validSinceAsOf,
    asOf: buy.asOf,
  };
}

function buildReplay(bars: OhlcSeries, maxRiskUsd: number): Replay | null {
  const ledger = runCloseLedger(bars, { ...LEDGER, risk_dollars: maxRiskUsd });
  if (!ledger) return null;
  const overlay = runStructureOverlay(bars);
  const breaks: Exit[] = [];
  if (overlay) {
    for (let i = 0; i < bars.length; i++) {
      if (overlay.bullish_break[i]) {
        breaks.push({ date: bars[i].date, price: bars[i].close, reason: 'sell_to_close' });
      }
    }
  }
  return { trades: ledger.trades, open: ledger.open, breaks };
}

function tradeOf(replay: Replay | null, since: string): CloseTrade | null {
  if (!replay) return null;
  for (const trade of replay.trades) {
    if (trade.entry_date > since) return null;
    if (trade.exit_date == null || trade.exit_date > since) return trade;
  }
  return null;
}

function exitOf(replay: Replay | null, since: string): Exit | null {
  return replay?.breaks.find((brk) => brk.date > since) ?? null;
}

function alignment(
  doc: TrackedSignal,
  trade: CloseTrade | null,
  tf: UserTf,
  maxRiskUsd: number,
): Partial<TrackedSignal> | null {
  if (!trade || doc.imported) return null;
  const entry = round2(trade.entry_price);
  if (doc.openedAsOf === trade.entry_date && doc.entry === entry) return null;
  const sl = finiteOrNull(trade.entry_sl);
  return {
    entry,
    sl,
    tp: finiteOrNull(trade.entry_tp),
    rrAtEntry: finiteOrNull(trade.entry_rr),
    openedAsOf: trade.entry_date,
    openedPeriodKey: barPeriodKey(tf, trade.entry_date),
    openedAt: atNoon(trade.entry_date),
    shares: sharesFromRisk(entry, sl, maxRiskUsd),
    riskUsd: maxRiskUsd,
  };
}

function clearedExit(): Partial<TrackedSignal> {
  return {
    closedPeriodKey: null,
    closedAt: null,
    exitDate: null,
    exitPrice: null,
    exitReason: null,
    pnlUsd: null,
    pnlR: null,
    pnlPct: null,
    holdPeriods: null,
  };
}

function refresh(
  doc: TrackedSignal,
  snapshot: Snapshot,
  input: ApplyScanInput,
  trade: CloseTrade | null,
  now: string,
): TrackedSignal {
  const aligned = alignment(doc, trade, input.tf, input.maxRiskUsd);
  const entry = aligned?.entry ?? doc.entry;
  const sl = aligned?.sl !== undefined ? aligned.sl : doc.sl;
  const shares = sharesFromRisk(entry, sl, input.maxRiskUsd);
  const pnl = computePnl(entry, sl, shares, snapshot.entry);
  return {
    ...doc,
    ...(aligned ?? {}),
    ...(doc.provisionalClose ? clearedExit() : {}),
    companyName: snapshot.companyName,
    tvSymbol: snapshot.tvSymbol,
    symbol: snapshot.symbol,
    lastSeenPeriodKey: input.periodKey,
    lastSeenAsOf: snapshot.asOf,
    lastPrice: round2(snapshot.entry),
    lastRr: snapshot.rr,
    barsSinceValid: snapshot.barsSinceValid,
    validSinceAsOf: snapshot.validSinceAsOf,
    isStrong: snapshot.isStrong,
    shares,
    riskUsd: input.maxRiskUsd,
    unrealizedUsd: pnl.usd,
    unrealizedR: pnl.r,
    unrealizedPct: pnl.pct,
    provisionalClose: false,
    signalValid: true,
    provisional: input.confirmed ? false : doc.provisional,
    status: 'active',
    openedAt: aligned?.openedAt ?? doc.openedAt,
  };
}

function closeDoc(
  doc: TrackedSignal,
  trade: CloseTrade | null,
  exit: Exit,
  input: ApplyScanInput,
  pending: boolean,
  now: string,
): TrackedSignal {
  const aligned = alignment(doc, trade, input.tf, input.maxRiskUsd);
  const entry = aligned?.entry ?? doc.entry;
  const sl = aligned?.sl !== undefined ? aligned.sl : doc.sl;
  const openedAsOf = aligned?.openedAsOf ?? doc.openedAsOf;
  const shares = aligned?.shares ?? doc.shares;
  const pnl = computePnl(entry, sl, shares, exit.price);
  return {
    ...doc,
    ...(aligned ?? {}),
    status: pending ? 'active' : 'closed',
    provisional: false,
    provisionalClose: pending,
    closedPeriodKey: barPeriodKey(input.tf, exit.date),
    closedAt: now,
    exitDate: exit.date,
    exitPrice: round2(exit.price),
    exitReason: exit.reason,
    pnlUsd: pnl.usd,
    pnlR: pnl.r,
    pnlPct: pnl.pct,
    holdPeriods: holdPeriods(input.tf, openedAsOf, exit.date),
    unrealizedUsd: pending ? doc.unrealizedUsd : null,
    unrealizedR: pending ? doc.unrealizedR : null,
    unrealizedPct: pending ? doc.unrealizedPct : null,
  };
}

function missing(
  doc: TrackedSignal,
  trade: CloseTrade | null,
  input: ApplyScanInput,
  evaluated: boolean,
): TrackedSignal {
  const aligned = alignment(doc, trade, input.tf, input.maxRiskUsd);
  return {
    ...doc,
    ...(aligned ?? {}),
    ...(doc.provisionalClose ? { ...clearedExit(), provisionalClose: false } : {}),
    signalValid: evaluated && doc.signalValid !== false ? false : doc.signalValid,
  };
}

function newDocument(
  snapshot: Snapshot,
  open: CloseTrade | null,
  input: ApplyScanInput,
  now: string,
): TrackedSignal {
  const entry = open ? round2(open.entry_price) : round2(snapshot.entry);
  const sl = open ? finiteOrNull(open.entry_sl) : snapshot.sl;
  const tp = open ? finiteOrNull(open.entry_tp) : snapshot.tp;
  const rr = open ? finiteOrNull(open.entry_rr) : snapshot.rr;
  const openedAsOf = open ? open.entry_date : snapshot.asOf;
  const openedPeriodKey = barPeriodKey(input.tf, openedAsOf);
  const shares = sharesFromRisk(entry, sl, input.maxRiskUsd);
  const pnl = computePnl(entry, sl, shares, snapshot.entry);
  return {
    id: newId(),
    yahooTicker: snapshot.yahooTicker,
    symbol: snapshot.symbol,
    tvSymbol: snapshot.tvSymbol,
    companyName: snapshot.companyName,
    universe: input.universe,
    tf: input.tf,
    status: 'active',
    provisional: !input.confirmed && openedPeriodKey === input.periodKey,
    provisionalClose: false,
    signalValid: true,
    imported: false,
    openedPeriodKey,
    openedAsOf,
    openedAt: now,
    entry,
    tp,
    sl,
    rrAtEntry: rr,
    shares,
    riskUsd: input.maxRiskUsd,
    lastSeenPeriodKey: input.periodKey,
    lastSeenAsOf: snapshot.asOf,
    lastPrice: round2(snapshot.entry),
    lastRr: snapshot.rr,
    barsSinceValid: snapshot.barsSinceValid,
    validSinceAsOf: snapshot.validSinceAsOf,
    isStrong: snapshot.isStrong,
    unrealizedUsd: pnl.usd,
    unrealizedR: pnl.r,
    unrealizedPct: pnl.pct,
    closedPeriodKey: null,
    closedAt: null,
    exitDate: null,
    exitPrice: null,
    exitReason: null,
    pnlUsd: null,
    pnlR: null,
    pnlPct: null,
    holdPeriods: null,
    interest: null,
    interestRank: 1,
  };
}

function adoptedDocument(
  sell: SellSignal,
  input: ApplyScanInput,
  pending: boolean,
  now: string,
): TrackedSignal {
  const entry = round2(sell.entry);
  const sl = Number.isFinite(sell.entrySl) ? sell.entrySl : null;
  const shares = sharesFromRisk(entry, sl, input.maxRiskUsd);
  const pnl = computePnl(entry, sl, shares, sell.exit);
  return {
    id: newId(),
    yahooTicker: sell.yahooTicker,
    symbol: sell.symbol,
    tvSymbol: sell.tvSymbol,
    companyName: sell.companyName,
    universe: input.universe,
    tf: input.tf,
    status: pending ? 'active' : 'closed',
    provisional: false,
    provisionalClose: pending,
    signalValid: false,
    imported: false,
    openedPeriodKey: barPeriodKey(input.tf, sell.entryAsOf),
    openedAsOf: sell.entryAsOf,
    openedAt: atNoon(sell.entryAsOf),
    entry,
    tp: Number.isFinite(sell.entryTp) ? sell.entryTp : null,
    sl,
    rrAtEntry: sell.rrAtEntry,
    shares,
    riskUsd: input.maxRiskUsd,
    lastSeenPeriodKey: barPeriodKey(input.tf, sell.exitAsOf),
    lastSeenAsOf: sell.exitAsOf,
    lastPrice: round2(sell.exit),
    lastRr: null,
    barsSinceValid: null,
    validSinceAsOf: null,
    isStrong: false,
    unrealizedUsd: null,
    unrealizedR: null,
    unrealizedPct: null,
    closedPeriodKey: barPeriodKey(input.tf, sell.exitAsOf),
    closedAt: now,
    exitDate: sell.exitAsOf,
    exitPrice: round2(sell.exit),
    exitReason: 'sell_to_close',
    pnlUsd: pnl.usd,
    pnlR: pnl.r,
    pnlPct: pnl.pct,
    holdPeriods: holdPeriods(input.tf, sell.entryAsOf, sell.exitAsOf),
    interest: null,
    interestRank: 1,
  };
}

function atNoon(date: string): string {
  return new Date(`${date}T12:00:00Z`).toISOString();
}
