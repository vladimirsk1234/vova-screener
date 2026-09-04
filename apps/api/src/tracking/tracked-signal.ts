/** Shared shapes and money maths for tracked signals (Results + History). */
import { shortSymbol, type Timeframe } from '@vova/engine';

export { TIMEFRAMES, type HistoryGroupBy, type HistoryTf, type UserTimeframe } from './tf';

export type TrackedUniverse = 'Stocks' | 'ETF';
export type Bucket = 'new' | 'valid' | 'closed';
export type Interest = 'interested' | 'not_interested';
/**
 * A trade this app takes only ever ends on 'sell_to_close'. The rest describe exits recorded by
 * the imported journal or by builds that closed positions on rules this one no longer applies.
 */
export type ExitReason = 'TP' | 'SL' | 'sell_to_close' | 'signal_lost' | 'manual';

export const UNIVERSES: readonly TrackedUniverse[] = ['Stocks', 'ETF'];
export const BUCKETS: readonly Bucket[] = ['new', 'valid', 'closed'];

export const INTEREST_RANK: Record<Interest | 'none', number> = {
  interested: 2,
  none: 1,
  not_interested: 0,
};

export function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

/** Restrict a Mongo match to the Settings valuation set, or leave it unchanged when the filter is off. */
export function withYahooTickers(
  match: Record<string, unknown>,
  tickers: string[] | null,
): Record<string, unknown> {
  if (!tickers) return match;
  return { ...match, yahooTicker: { $in: tickers } };
}

export function finiteOrNull(n: unknown): number | null {
  return typeof n === 'number' && Number.isFinite(n) ? n : null;
}

/** Position size that keeps the loss at SL within `riskUsd`. */
export function sharesFromRisk(
  entry: number,
  sl: number | null | undefined,
  riskUsd: number,
): number {
  if (sl == null || !Number.isFinite(sl) || !Number.isFinite(entry)) return 0;
  const risk = entry - sl;
  if (risk <= 0 || riskUsd <= 0) return 0;
  return Math.max(0, Math.round(riskUsd / risk));
}

export type Pnl = { usd: number | null; r: number | null; pct: number | null };

export function computePnl(
  entry: number,
  sl: number | null | undefined,
  shares: number,
  price: number | null | undefined,
): Pnl {
  if (price == null || !Number.isFinite(price) || !Number.isFinite(entry)) {
    return { usd: null, r: null, pct: null };
  }
  const usd = round2((price - entry) * (shares || 0));
  const risk = sl != null && Number.isFinite(sl) ? entry - sl : 0;
  const invested = entry * (shares || 0);
  return {
    usd,
    r: risk > 0 ? round2((price - entry) / risk) : null,
    pct: invested > 0 ? round2((usd / invested) * 100) : null,
  };
}

/** Holding length in timeframe units: trading days / weeks / months. */
export function holdPeriods(tf: Timeframe, from?: string, to?: string): number | null {
  if (!from || !to) return null;
  const ms = Date.parse(`${to}T12:00:00Z`) - Date.parse(`${from}T12:00:00Z`);
  if (!Number.isFinite(ms) || ms < 0) return null;
  const days = ms / 86_400_000;
  if (tf === 'Daily') return round2(days);
  if (tf === 'Weekly') return round2(days / 7);
  return round2(days / 30.4375);
}

export function holdUnitLabel(tf: Timeframe | 'All'): string {
  if (tf === 'Daily') return 'days';
  if (tf === 'Weekly') return 'weeks';
  if (tf === 'Monthly') return 'months';
  return 'periods';
}

/** One row as rendered by the Results and History lists. */
export type ResultRow = {
  id: string;
  /** Display form, no exchange prefix. `tvSymbol` is what a TradingView link needs. */
  symbol: string;
  tvSymbol: string;
  yahooTicker: string;
  companyName: string;
  universe: TrackedUniverse;
  tf: Timeframe;
  status: 'active' | 'closed';
  provisional: boolean;
  /** Sell-to-close break on the bar still in progress: in CLOSED now, in History once it finishes. */
  provisionalClose: boolean;
  entry: number;
  tp: number | null;
  sl: number | null;
  rr: number | null;
  currentRr: number | null;
  shares: number;
  positionValue: number;
  riskUsd: number;
  isStrong: boolean;
  openedPeriodKey: string;
  openedAsOf: string | null;
  /** Bars of `tf` since the signal became valid: 0 in NEW, 1 or more in VALID. */
  barsSinceValid: number | null;
  validSinceAsOf: string | null;
  lastPrice: number | null;
  lastSeenAsOf: string | null;
  /** Unrealized while the trade runs, priced at the exit once a break closes it. */
  pnlUsd: number | null;
  pnlR: number | null;
  pnlPct: number | null;
  realized: boolean;
  closedPeriodKey: string | null;
  exitDate: string | null;
  exitPrice: number | null;
  exitReason: ExitReason | null;
  holdPeriods: number | null;
  interest: Interest | null;
  /** FMP EPS on/before openedAsOf. null = not tagged yet. */
  epsAtEntry: number | null;
  epsPositiveAtEntry: boolean | null;
  /** Card premium (5Y Op. EPS) as of openedAsOf. null = tagged unknown; omit until stamped. */
  premiumPctAtEntry: number | null;
  undervaluedAtEntry: boolean | null;
};

export function toResultRow(doc: any): ResultRow {
  const closed = doc.status === 'closed';
  const provisionalClose = Boolean(doc.provisionalClose);
  // A trade closing on the bar in progress already has its exit priced, so the card shows that
  // number rather than the mark-to-market one it would otherwise still be carrying.
  const atExit = closed || provisionalClose;
  const shares = doc.shares ?? 0;
  return {
    id: String(doc._id),
    // Records written by older builds — and by the trade journal — kept the exchange prefix here,
    // so the one display format is enforced on the way out as well as on the way in.
    symbol: shortSymbol(doc.symbol ?? doc.yahooTicker),
    tvSymbol: doc.tvSymbol ?? doc.symbol,
    yahooTicker: doc.yahooTicker,
    companyName: doc.companyName ?? doc.yahooTicker,
    universe: doc.universe,
    tf: doc.tf,
    status: doc.status,
    provisional: Boolean(doc.provisional),
    provisionalClose,
    entry: doc.entry,
    tp: finiteOrNull(doc.tp),
    sl: finiteOrNull(doc.sl),
    rr: finiteOrNull(doc.rrAtEntry),
    currentRr: finiteOrNull(doc.lastRr),
    shares,
    positionValue: round2((doc.entry ?? 0) * shares),
    riskUsd: doc.riskUsd ?? 0,
    isStrong: Boolean(doc.isStrong),
    openedPeriodKey: doc.openedPeriodKey,
    openedAsOf: doc.openedAsOf ?? null,
    barsSinceValid: finiteOrNull(doc.barsSinceValid),
    validSinceAsOf: doc.validSinceAsOf ?? null,
    lastPrice: finiteOrNull(doc.lastPrice),
    lastSeenAsOf: doc.lastSeenAsOf ?? null,
    pnlUsd: finiteOrNull(atExit ? doc.pnlUsd : doc.unrealizedUsd),
    pnlR: finiteOrNull(atExit ? doc.pnlR : doc.unrealizedR),
    pnlPct: finiteOrNull(atExit ? doc.pnlPct : doc.unrealizedPct),
    realized: closed,
    closedPeriodKey: doc.closedPeriodKey ?? null,
    exitDate: doc.exitDate ?? null,
    exitPrice: finiteOrNull(doc.exitPrice),
    exitReason: doc.exitReason ?? null,
    holdPeriods: finiteOrNull(doc.holdPeriods),
    interest: doc.interest ?? null,
    epsAtEntry: finiteOrNull(doc.epsAtEntry),
    epsPositiveAtEntry:
      doc.epsPositiveAtEntry == null ? null : Boolean(doc.epsPositiveAtEntry),
    premiumPctAtEntry: finiteOrNull(doc.premiumPctAtEntry),
    undervaluedAtEntry:
      doc.undervaluedAtEntry == null ? null : Boolean(doc.undervaluedAtEntry),
  };
}
