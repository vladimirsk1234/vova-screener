/** History: statistics over closed tracked signals, aggregated in Mongo. */
import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { TRACKED_SIGNAL } from '../db/schemas';
import {
  holdUnitLabel,
  round2,
  toResultRow,
  type ResultRow,
} from './tracked-signal';

export type HistoryTf = Timeframe | 'All';
export type PeriodSort = 'period' | 'pnl' | 'winRate' | 'trades';
export type TradeSort = 'date' | 'pnl' | 'r' | 'rr' | 'interest' | 'symbol';
export type SortDir = 'asc' | 'desc';

export type HistoryPeriod = {
  periodKey: string;
  trades: number;
  wins: number;
  winRatePct: number;
  pnlUsd: number;
  invested: number;
  avgR: number | null;
  avgRrEntry: number | null;
  avgHold: number | null;
};

export type HistoryReport = {
  tf: HistoryTf;
  groupBy: Timeframe;
  holdUnit: string;
  periods: HistoryPeriod[];
  equity: Array<{ periodKey: string; equity: number }>;
  exitReasons: Array<{ reason: string; count: number }>;
  totals: {
    closed: number;
    active: number;
    wins: number;
    winRatePct: number;
    pnlUsd: number;
    invested: number;
    avgR: number | null;
    avgRrEntry: number | null;
    avgHold: number | null;
  };
};

const GROUP_ACCUMULATORS = {
  trades: { $sum: 1 },
  wins: { $sum: { $cond: [{ $gt: ['$pnlUsd', 0] }, 1, 0] } },
  pnlUsd: { $sum: { $ifNull: ['$pnlUsd', 0] } },
  invested: {
    $sum: { $multiply: [{ $ifNull: ['$entry', 0] }, { $ifNull: ['$shares', 0] }] },
  },
  avgR: { $avg: '$pnlR' },
  avgRrEntry: { $avg: '$rrAtEntry' },
  avgHold: { $avg: '$holdPeriods' },
};

@Injectable()
export class HistoryService {
  constructor(@InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>) {}

  async report(opts: {
    tf: HistoryTf;
    groupBy: Timeframe;
    sort?: PeriodSort;
    dir?: SortDir;
  }): Promise<HistoryReport> {
    const { tf, groupBy } = opts;
    const sort = opts.sort ?? 'period';
    const dir = opts.dir ?? 'desc';
    const match = closedMatch(tf);

    const [facet, active] = await Promise.all([
      this.tracked
        .aggregate([
          { $match: match },
          { $addFields: { bucket: bucketExpression(groupBy) } },
          {
            $facet: {
              periods: [{ $group: { _id: '$bucket', ...GROUP_ACCUMULATORS } }, { $sort: { _id: 1 } }],
              totals: [{ $group: { _id: null, ...GROUP_ACCUMULATORS } }],
              exitReasons: [
                { $group: { _id: '$exitReason', count: { $sum: 1 } } },
                { $sort: { count: -1 } },
              ],
            },
          },
        ])
        .exec(),
      this.tracked.countDocuments(activeMatch(tf)).exec(),
    ]);

    const raw = (facet?.[0]?.periods ?? []) as any[];
    const ascending = raw.map(finalizePeriod);

    let cumulative = 0;
    const equity = ascending.map((p) => {
      cumulative += p.pnlUsd;
      return { periodKey: p.periodKey, equity: round2(cumulative) };
    });

    const totalsRaw = (facet?.[0]?.totals ?? [])[0];
    const totals = totalsRaw
      ? finalizePeriod(totalsRaw)
      : finalizePeriod({ _id: '', trades: 0, wins: 0, pnlUsd: 0, invested: 0 });

    return {
      tf,
      groupBy,
      holdUnit: holdUnitLabel(tf),
      periods: sortPeriods(ascending, sort, dir),
      equity,
      exitReasons: ((facet?.[0]?.exitReasons ?? []) as any[]).map((r) => ({
        reason: r._id ?? 'unknown',
        count: r.count,
      })),
      totals: {
        closed: totals.trades,
        active,
        wins: totals.wins,
        winRatePct: totals.winRatePct,
        pnlUsd: totals.pnlUsd,
        invested: totals.invested,
        avgR: totals.avgR,
        avgRrEntry: totals.avgRrEntry,
        avgHold: totals.avgHold,
      },
    };
  }

  async trades(opts: {
    tf: HistoryTf;
    periodKey?: string;
    groupBy?: Timeframe;
    sort?: TradeSort;
    dir?: SortDir;
    limit?: number;
    offset?: number;
  }): Promise<{ total: number; rows: ResultRow[] }> {
    const limit = Math.min(Math.max(opts.limit ?? 100, 1), 500);
    const offset = Math.max(opts.offset ?? 0, 0);
    const filter: Record<string, unknown> = closedMatch(opts.tf);
    if (opts.periodKey) {
      const range = periodDateRange(opts.periodKey, opts.groupBy ?? 'Daily');
      if (range) filter.exitDate = range;
    }

    const [rows, total] = await Promise.all([
      this.tracked
        .find(filter)
        .sort(tradeSortSpec(opts.sort ?? 'date', opts.dir ?? 'desc'))
        .skip(offset)
        .limit(limit)
        .lean<any[]>()
        .exec(),
      this.tracked.countDocuments(filter).exec(),
    ]);
    return { total, rows: rows.map(toResultRow) };
  }
}

function closedMatch(tf: HistoryTf): Record<string, unknown> {
  const match: Record<string, unknown> = { status: 'closed' };
  if (tf !== 'All') match.tf = tf;
  return match;
}

function activeMatch(tf: HistoryTf): Record<string, unknown> {
  const match: Record<string, unknown> = { status: 'active' };
  if (tf !== 'All') match.tf = tf;
  return match;
}

/** Bucket key derived from the exit date, so History can be grouped by day, week or month. */
function bucketExpression(groupBy: Timeframe): Record<string, unknown> {
  if (groupBy === 'Daily') return { $substrCP: [{ $ifNull: ['$exitDate', 'unknown'] }, 0, 10] };
  if (groupBy === 'Monthly') return { $substrCP: [{ $ifNull: ['$exitDate', 'unknown'] }, 0, 7] };

  const exitAt = {
    $dateFromString: {
      dateString: { $substrCP: [{ $ifNull: ['$exitDate', ''] }, 0, 10] },
      format: '%Y-%m-%d',
      onError: null,
      onNull: null,
    },
  };
  const week = { $isoWeek: exitAt };
  return {
    $cond: [
      { $eq: [exitAt, null] },
      'unknown',
      {
        $concat: [
          { $toString: { $isoWeekYear: exitAt } },
          '-W',
          {
            $cond: [
              { $lt: [week, 10] },
              { $concat: ['0', { $toString: week }] },
              { $toString: week },
            ],
          },
        ],
      },
    ],
  };
}

/** Exit-date range covered by one history bucket, so drilling into a period stays an index scan. */
function periodDateRange(
  periodKey: string,
  groupBy: Timeframe,
): { $gte: string; $lte: string } | null {
  if (groupBy === 'Daily') return { $gte: periodKey, $lte: periodKey };
  if (groupBy === 'Monthly') return { $gte: `${periodKey}-01`, $lte: `${periodKey}-31` };

  const match = /^(\d{4})-W(\d{2})$/.exec(periodKey);
  if (!match) return null;
  // ISO-8601: January 4th always falls in week 1.
  const jan4 = Date.UTC(Number(match[1]), 0, 4);
  const jan4Weekday = new Date(jan4).getUTCDay() || 7;
  const week1Monday = jan4 - (jan4Weekday - 1) * 86_400_000;
  const monday = week1Monday + (Number(match[2]) - 1) * 7 * 86_400_000;
  const iso = (ms: number) => new Date(ms).toISOString().slice(0, 10);
  return { $gte: iso(monday), $lte: iso(monday + 6 * 86_400_000) };
}

function finalizePeriod(row: any): HistoryPeriod {
  const trades = row.trades ?? 0;
  const wins = row.wins ?? 0;
  return {
    periodKey: row._id ?? 'unknown',
    trades,
    wins,
    winRatePct: trades ? round2((wins / trades) * 100) : 0,
    pnlUsd: round2(row.pnlUsd ?? 0),
    invested: round2(row.invested ?? 0),
    avgR: row.avgR == null ? null : round2(row.avgR),
    avgRrEntry: row.avgRrEntry == null ? null : round2(row.avgRrEntry),
    avgHold: row.avgHold == null ? null : round2(row.avgHold),
  };
}

function sortPeriods(periods: HistoryPeriod[], sort: PeriodSort, dir: SortDir): HistoryPeriod[] {
  const order = dir === 'asc' ? 1 : -1;
  const value = (p: HistoryPeriod) =>
    sort === 'pnl' ? p.pnlUsd : sort === 'winRate' ? p.winRatePct : sort === 'trades' ? p.trades : 0;
  return [...periods].sort((a, b) => {
    if (sort === 'period') return a.periodKey.localeCompare(b.periodKey) * order;
    return (value(a) - value(b)) * order || a.periodKey.localeCompare(b.periodKey) * -1;
  });
}

function tradeSortSpec(sort: TradeSort, dir: SortDir): Record<string, 1 | -1> {
  const order: 1 | -1 = dir === 'asc' ? 1 : -1;
  const field =
    sort === 'pnl'
      ? 'pnlUsd'
      : sort === 'r'
        ? 'pnlR'
        : sort === 'rr'
          ? 'rrAtEntry'
          : sort === 'interest'
            ? 'interestRank'
            : sort === 'symbol'
              ? 'symbol'
              : 'exitDate';
  const spec: Record<string, 1 | -1> = { [field]: order };
  if (field !== 'exitDate') spec.exitDate = -1;
  return spec;
}
