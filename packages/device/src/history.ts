/** History statistics. Same numbers as the server report, with the FMP filter left off. */
import { barPeriodKey } from '../../../apps/api/src/scans/period.ts';
import {
  benchmarkFromSeries,
  withBenchmark,
  type DailyClose,
} from '../../../apps/api/src/tracking/benchmark.ts';
import { computePeakCapital, roiOnAvgPct, roiOnPeakPct, type CapitalTrade } from '../../../apps/api/src/tracking/peak-capital.ts';
import { computePnl, holdUnitLabel, round2, sharesFromRisk } from './money';
import type {
  HistoryGroupBy,
  HistoryPeriodSort,
  HistoryRange,
  HistoryTf,
  HistoryTradeSort,
  SortDir,
  TrackedSignal,
  Universe,
  UserTf,
} from './types';

export type HistoryPeriod = {
  periodKey: string;
  trades: number;
  wins: number;
  losses: number;
  winRatePct: number;
  pnlUsd: number;
  invested: number;
  avgR: number | null;
  avgRrEntry: number | null;
  avgHold: number | null;
  avgTradeSizeUsd: number | null;
  avgWinPct: number | null;
  avgLossPct: number | null;
  avgPnlPct: number | null;
};

export type EquityPoint = { periodKey: string; equity: number };

export type HistoryTimeframe = {
  tf: UserTf;
  closed: number;
  wins: number;
  winRatePct: number;
  pnlUsd: number;
  invested: number;
  avgR: number | null;
  avgTradeSizeUsd: number | null;
  avgWinPct: number | null;
  avgLossPct: number | null;
  avgPnlPct: number | null;
  equity: EquityPoint[];
};

export type HistoryReport = {
  universe: Universe;
  tf: HistoryTf;
  groupBy: HistoryGroupBy;
  range: HistoryRange;
  holdUnit: string;
  periods: HistoryPeriod[];
  equity: EquityPoint[];
  timeframes: HistoryTimeframe[];
  exitReasons: Array<{ reason: string; count: number }>;
  totals: HistoryPeriod & {
    closed: number;
    active: number;
    maxRiskUsd: number;
    totalRiskUsd: number;
    profitToRisk: number | null;
    peakCapitalUsd: number;
    peakCapitalAsOf: string | null;
    peakConcurrentPositions: number;
    openCapitalUsd: number;
    roiOnPeakPct: number | null;
    avgCapitalUsd: number;
    roiOnAvgPct: number | null;
    benchmarkReturnPct: number | null;
    benchmarkSymbol: string | null;
    alphaVsBenchmarkPct: number | null;
    alphaOnAvgPct: number | null;
  };
};

export function historyReport(opts: {
  signals: TrackedSignal[];
  universe: Universe;
  tf: HistoryTf;
  groupBy: HistoryGroupBy;
  range?: HistoryRange;
  sort?: HistoryPeriodSort;
  dir?: SortDir;
  minRr: number;
  maxRiskUsd: number;
  benchmark: { symbol: string; bars: DailyClose[] } | null;
  now?: Date;
}): HistoryReport {
  const range = normalizeRange(opts.range);
  const closed = opts.signals
    .filter((row) => matchesClosed(row, opts.universe, opts.tf, opts.minRr, range, opts.now))
    .map((row) => resizeClosed(row, opts.maxRiskUsd));
  const ascending = groupPeriods(closed, opts.groupBy);
  const equity = equityCurve(ascending);
  const totals = summarize(closed);
  const risk = riskStats(totals.trades, totals.pnlUsd, opts.maxRiskUsd);
  const rangeFrom = lookbackFrom(range, opts.now ?? new Date());
  const rangeEnd = (opts.now ?? new Date()).toISOString().slice(0, 10);
  const capital = capitalTrades(opts.signals, opts, range);
  const peak = computePeakCapital(capital, { rangeFrom, rangeEnd });
  const roiPeak = roiOnPeakPct(totals.pnlUsd, peak.peakCapitalUsd);
  const roiAvg = roiOnAvgPct(totals.pnlUsd, peak.avgCapitalUsd);
  const bench = benchmarkFromSeries(opts.benchmark, peak.windowFrom, peak.windowTo);
  const active = opts.signals.filter((row) => matchesActive(row, opts.universe, opts.tf, opts.minRr)).length;
  return {
    universe: opts.universe,
    tf: opts.tf,
    groupBy: opts.groupBy,
    range,
    holdUnit: holdUnitLabel(opts.tf === 'All' ? 'All' : opts.tf),
    periods: sortPeriods(ascending, opts.sort ?? 'period', opts.dir ?? 'desc'),
    equity,
    timeframes: (['Weekly', 'Monthly'] as const).map((tf) => timeframeRow(opts, tf, range)),
    exitReasons: exitReasons(closed),
    totals: {
      ...totals,
      closed: totals.trades,
      active,
      ...risk,
      peakCapitalUsd: peak.peakCapitalUsd,
      peakCapitalAsOf: peak.peakCapitalAsOf,
      peakConcurrentPositions: peak.peakConcurrentPositions,
      openCapitalUsd: peak.openCapitalUsd,
      roiOnPeakPct: roiPeak,
      avgCapitalUsd: peak.avgCapitalUsd,
      roiOnAvgPct: roiAvg,
      ...withBenchmark(roiPeak, roiAvg, bench),
    },
  };
}

export function historyTrades(opts: {
  signals: TrackedSignal[];
  universe: Universe;
  tf: HistoryTf;
  groupBy: HistoryGroupBy;
  range?: HistoryRange;
  periodKey?: string;
  sort?: HistoryTradeSort;
  dir?: SortDir;
  minRr: number;
  maxRiskUsd: number;
  limit?: number;
  offset?: number;
  now?: Date;
}): { total: number; rows: TrackedSignal[] } {
  const range = normalizeRange(opts.range);
  let rows = opts.signals
    .filter((row) => matchesClosed(row, opts.universe, opts.tf, opts.minRr, range, opts.now))
    .map((row) => resizeClosed(row, opts.maxRiskUsd));
  if (opts.periodKey) {
    const bucket = periodDateRange(opts.periodKey, opts.groupBy);
    if (bucket) {
      rows = rows.filter((row) => {
        const exit = row.exitDate ?? '';
        return exit >= bucket.$gte && exit <= bucket.$lte;
      });
    }
  }
  rows.sort((a, b) => compareTrades(a, b, opts.sort ?? 'date', opts.dir ?? 'desc'));
  const offset = Math.max(opts.offset ?? 0, 0);
  const limit = Math.min(Math.max(opts.limit ?? 200, 1), 500);
  return { total: rows.length, rows: rows.slice(offset, offset + limit) };
}

function timeframeRow(
  opts: {
    signals: TrackedSignal[];
    universe: Universe;
    minRr: number;
    maxRiskUsd: number;
    now?: Date;
  },
  tf: UserTf,
  range: HistoryRange,
): HistoryTimeframe {
  const closed = opts.signals
    .filter((row) => matchesClosed(row, opts.universe, tf, opts.minRr, range, opts.now))
    .map((row) => resizeClosed(row, opts.maxRiskUsd));
  const periods = groupPeriods(closed, tf);
  const totals = summarize(closed);
  return {
    tf,
    closed: totals.trades,
    wins: totals.wins,
    winRatePct: totals.winRatePct,
    pnlUsd: totals.pnlUsd,
    invested: totals.invested,
    avgR: totals.avgR,
    avgTradeSizeUsd: totals.avgTradeSizeUsd,
    avgWinPct: totals.avgWinPct,
    avgLossPct: totals.avgLossPct,
    avgPnlPct: totals.avgPnlPct,
    equity: equityCurve(periods),
  };
}

function matchesClosed(
  row: TrackedSignal,
  universe: Universe,
  tf: HistoryTf,
  minRr: number,
  range: HistoryRange,
  now?: Date,
): boolean {
  if (row.universe !== universe || row.status !== 'closed') return false;
  if (tf !== 'All' && row.tf !== tf) return false;
  if (minRr > 0 && (row.rrAtEntry == null || row.rrAtEntry < minRr)) return false;
  const from = lookbackFrom(range, now ?? new Date());
  if (from && (row.exitDate == null || row.exitDate < from)) return false;
  return true;
}

function matchesActive(row: TrackedSignal, universe: Universe, tf: HistoryTf, minRr: number): boolean {
  if (row.universe !== universe || row.status !== 'active' || row.provisional) return false;
  if (tf !== 'All' && row.tf !== tf) return false;
  if (minRr > 0 && (row.lastRr == null || row.lastRr < minRr)) return false;
  return true;
}

function resizeClosed(row: TrackedSignal, maxRiskUsd: number): TrackedSignal {
  const shares = sharesFromRisk(row.entry, row.sl, maxRiskUsd);
  const pnl = computePnl(row.entry, row.sl, shares, row.exitPrice);
  return { ...row, shares, riskUsd: maxRiskUsd, pnlUsd: pnl.usd, pnlPct: pnl.pct };
}

function resizeShares(row: TrackedSignal, maxRiskUsd: number): TrackedSignal {
  const shares = sharesFromRisk(row.entry, row.sl, maxRiskUsd);
  return { ...row, shares, riskUsd: maxRiskUsd };
}

function capitalTrades(
  signals: TrackedSignal[],
  opts: { universe: Universe; tf: HistoryTf; minRr: number; maxRiskUsd: number; now?: Date },
  range: HistoryRange,
): CapitalTrade[] {
  const rows: CapitalTrade[] = [];
  const seen = new Set<string>();
  for (const raw of signals) {
    const closed = matchesClosed(raw, opts.universe, opts.tf, opts.minRr, range, opts.now);
    const active = matchesActive(raw, opts.universe, opts.tf, opts.minRr);
    if (!closed && !active) continue;
    if (seen.has(raw.id)) continue;
    seen.add(raw.id);
    const row = resizeShares(raw, opts.maxRiskUsd);
    rows.push({
      id: row.id,
      openedAsOf: row.openedAsOf,
      exitDate: row.status === 'closed' ? row.exitDate : null,
      positionValue: round2((row.entry ?? 0) * (row.shares ?? 0)),
    });
  }
  return rows;
}

function groupKey(exitDate: string, groupBy: HistoryGroupBy | UserTf): string {
  if (groupBy === 'Day') return exitDate.slice(0, 10);
  if (groupBy === 'Monthly') return exitDate.slice(0, 7);
  return barPeriodKey('Weekly', exitDate);
}

function groupPeriods(rows: TrackedSignal[], groupBy: HistoryGroupBy | UserTf): HistoryPeriod[] {
  const buckets = new Map<string, TrackedSignal[]>();
  for (const row of rows) {
    const key = row.exitDate ? groupKey(row.exitDate, groupBy) : 'unknown';
    const list = buckets.get(key) ?? [];
    list.push(row);
    buckets.set(key, list);
  }
  return [...buckets.keys()]
    .sort()
    .map((key) => summarize(buckets.get(key) ?? [], key));
}

function summarize(rows: TrackedSignal[], periodKey = ''): HistoryPeriod {
  const trades = rows.length;
  const wins = rows.filter((r) => (r.pnlUsd ?? 0) > 0).length;
  const losses = rows.filter((r) => (r.pnlUsd ?? 0) < 0).length;
  const pnlUsd = round2(rows.reduce((sum, r) => sum + (r.pnlUsd ?? 0), 0));
  const invested = round2(rows.reduce((sum, r) => sum + r.entry * r.shares, 0));
  return {
    periodKey,
    trades,
    wins,
    losses,
    winRatePct: trades ? round2((wins / trades) * 100) : 0,
    pnlUsd,
    invested,
    avgR: avg(rows.map((r) => r.pnlR)),
    avgRrEntry: avg(rows.map((r) => r.rrAtEntry)),
    avgHold: avg(rows.map((r) => r.holdPeriods)),
    avgTradeSizeUsd: avg(rows.map((r) => r.entry * r.shares)),
    avgWinPct: avg(rows.filter((r) => (r.pnlUsd ?? 0) > 0).map((r) => r.pnlPct)),
    avgLossPct: avg(rows.filter((r) => (r.pnlUsd ?? 0) < 0).map((r) => r.pnlPct)),
    avgPnlPct: avg(rows.map((r) => r.pnlPct)),
  };
}

function avg(values: Array<number | null>): number | null {
  const nums = values.filter((n): n is number => n != null && Number.isFinite(n));
  if (!nums.length) return null;
  return round2(nums.reduce((sum, n) => sum + n, 0) / nums.length);
}

function equityCurve(periods: HistoryPeriod[]): EquityPoint[] {
  let cumulative = 0;
  return periods.map((p) => {
    cumulative += p.pnlUsd;
    return { periodKey: p.periodKey, equity: round2(cumulative) };
  });
}

function exitReasons(rows: TrackedSignal[]): Array<{ reason: string; count: number }> {
  const counts = new Map<string, number>();
  for (const row of rows) {
    const reason = row.exitReason ?? 'unknown';
    counts.set(reason, (counts.get(reason) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count || a.reason.localeCompare(b.reason));
}

function riskStats(closed: number, pnlUsd: number, maxRiskUsd: number) {
  const totalRiskUsd = round2(closed * maxRiskUsd);
  return {
    maxRiskUsd: round2(maxRiskUsd),
    totalRiskUsd,
    profitToRisk: totalRiskUsd > 0 ? round2(pnlUsd / totalRiskUsd) : null,
  };
}

function sortPeriods(periods: HistoryPeriod[], sort: HistoryPeriodSort, dir: SortDir): HistoryPeriod[] {
  const order = dir === 'asc' ? 1 : -1;
  const value = (p: HistoryPeriod) =>
    sort === 'pnl' ? p.pnlUsd : sort === 'winRate' ? p.winRatePct : sort === 'trades' ? p.trades : sort === 'rr' ? (p.avgRrEntry ?? 0) : 0;
  return [...periods].sort((a, b) => {
    if (sort === 'period') return a.periodKey.localeCompare(b.periodKey) * order;
    return (value(a) - value(b)) * order || a.periodKey.localeCompare(b.periodKey) * -1;
  });
}

function compareTrades(a: TrackedSignal, b: TrackedSignal, sort: HistoryTradeSort, dir: SortDir): number {
  const sign = dir === 'asc' ? 1 : -1;
  if (sort === 'symbol') return a.symbol.localeCompare(b.symbol) * sign || cmpDate(a, b);
  if (sort === 'interest') return (a.interestRank - b.interestRank) * sign || cmpDate(a, b);
  const num = (n: number | null) => (n == null || !Number.isFinite(n) ? null : n);
  const av =
    sort === 'pnl' ? num(a.pnlUsd) : sort === 'r' ? num(a.pnlR) : sort === 'rr' ? num(a.rrAtEntry) : null;
  const bv =
    sort === 'pnl' ? num(b.pnlUsd) : sort === 'r' ? num(b.pnlR) : sort === 'rr' ? num(b.rrAtEntry) : null;
  if (sort === 'date') return cmpDate(a, b) * sign;
  if (av == null && bv == null) return cmpDate(a, b);
  if (av == null) return 1;
  if (bv == null) return -1;
  return (av - bv) * sign || cmpDate(a, b);
}

function cmpDate(a: TrackedSignal, b: TrackedSignal): number {
  return (b.exitDate ?? '').localeCompare(a.exitDate ?? '');
}

export function normalizeRange(range?: HistoryRange): HistoryRange {
  if (!range || range === 'max') return 'all';
  return range;
}

export function lookbackFrom(range: HistoryRange, now = new Date()): string | null {
  const normalised = normalizeRange(range);
  if (normalised === 'all') return null;
  const y = now.getUTCFullYear();
  const m = now.getUTCMonth();
  const d = now.getUTCDate();
  if (normalised === 'ytd') return `${y}-01-01`;
  const months = normalised === '1m' ? 1 : normalised === '3m' ? 3 : normalised === '6m' ? 6 : 12;
  return new Date(Date.UTC(y, m - months, d)).toISOString().slice(0, 10);
}

function periodDateRange(periodKey: string, groupBy: HistoryGroupBy): { $gte: string; $lte: string } | null {
  if (groupBy === 'Day') return { $gte: periodKey, $lte: periodKey };
  if (groupBy === 'Monthly') return { $gte: `${periodKey}-01`, $lte: `${periodKey}-31` };
  const match = /^(\d{4})-W(\d{2})$/.exec(periodKey);
  if (!match) return null;
  const jan4 = Date.UTC(Number(match[1]), 0, 4);
  const jan4Weekday = new Date(jan4).getUTCDay() || 7;
  const week1Monday = jan4 - (jan4Weekday - 1) * 86_400_000;
  const monday = week1Monday + (Number(match[2]) - 1) * 7 * 86_400_000;
  const iso = (ms: number) => new Date(ms).toISOString().slice(0, 10);
  return { $gte: iso(monday), $lte: iso(monday + 6 * 86_400_000) };
}
