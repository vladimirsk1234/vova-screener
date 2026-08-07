/** History: statistics over closed tracked signals, aggregated in Mongo. */
import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { TRACKED_SIGNAL } from '../db/schemas';
import { SettingsService } from '../settings/settings.module';
import {
  TIMEFRAMES,
  computePnl,
  holdUnitLabel,
  round2,
  sharesFromRisk,
  toResultRow,
  type ResultRow,
  type TrackedUniverse,
} from './tracked-signal';

export type HistoryTf = Timeframe | 'All';
/** Lookback on exit date. `max` is an alias of `all` (everything already in DB). */
export type HistoryRange = 'all' | 'ytd' | '1m' | '3m' | '6m' | '1y' | 'max';
export type PeriodSort = 'period' | 'pnl' | 'winRate' | 'trades' | 'rr';
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

export type EquityPoint = { periodKey: string; equity: number };

/**
 * One timeframe's own record, independent of the filter above it. Daily, Weekly and Monthly are
 * three different strategies sharing one screener, so the growth of each is worth seeing side by
 * side rather than one at a time.
 */
export type HistoryTimeframe = {
  tf: Timeframe;
  closed: number;
  wins: number;
  winRatePct: number;
  pnlUsd: number;
  /** Sum of entry × shares over closed trades in this timeframe. */
  invested: number;
  /** Net P&L / invested × 100; null when invested is 0. */
  returnPct: number | null;
  avgR: number | null;
  /** Cumulative P&L over that timeframe's own periods, oldest first. */
  equity: EquityPoint[];
};

export type HistoryReport = {
  universe: TrackedUniverse;
  tf: HistoryTf;
  groupBy: Timeframe;
  /** Exit-date lookback applied to closed trades (`max` normalised to `all`). */
  range: HistoryRange;
  holdUnit: string;
  periods: HistoryPeriod[];
  equity: EquityPoint[];
  timeframes: HistoryTimeframe[];
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
  constructor(
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    private readonly settings: SettingsService,
  ) {}

  async report(opts: {
    universe: TrackedUniverse;
    tf: HistoryTf;
    groupBy: Timeframe;
    range?: HistoryRange;
    sort?: PeriodSort;
    dir?: SortDir;
  }): Promise<HistoryReport> {
    const { universe, tf, groupBy } = opts;
    const range = normalizeRange(opts.range);
    const sort = opts.sort ?? 'period';
    const dir = opts.dir ?? 'desc';
    const { maxRiskUsd, minRr } = await this.settings.get();
    const match = closedMatch(universe, tf, minRr, range);

    const [facet, active, timeframes] = await Promise.all([
      this.tracked
        .aggregate([
          { $match: match },
          ...resizeStages(maxRiskUsd),
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
      this.tracked.countDocuments(activeMatch(universe, tf, minRr)).exec(),
      this.byTimeframe(universe, minRr, maxRiskUsd, range),
    ]);

    const raw = (facet?.[0]?.periods ?? []) as any[];
    const ascending = raw.map(finalizePeriod);
    const equity = equityCurve(ascending);

    const totalsRaw = (facet?.[0]?.totals ?? [])[0];
    const totals = totalsRaw
      ? finalizePeriod(totalsRaw)
      : finalizePeriod({ _id: '', trades: 0, wins: 0, pnlUsd: 0, invested: 0 });

    return {
      universe,
      tf,
      groupBy,
      range,
      holdUnit: holdUnitLabel(tf),
      periods: sortPeriods(ascending, sort, dir),
      equity,
      timeframes,
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

  /**
   * Each timeframe grouped by its own calendar, so a Weekly equity curve has one point per week
   * rather than per exit day. Three small aggregations rather than one grouped on a `$switch`:
   * the per-timeframe index does the work and the shapes stay readable.
   */
  private async byTimeframe(
    universe: TrackedUniverse,
    minRr: number,
    maxRiskUsd: number,
    range: HistoryRange,
  ): Promise<HistoryTimeframe[]> {
    return Promise.all(
      TIMEFRAMES.map(async (tf) => {
        const rows = await this.tracked
          .aggregate([
            { $match: closedMatch(universe, tf, minRr, range) },
            ...resizeStages(maxRiskUsd),
            { $addFields: { bucket: bucketExpression(tf) } },
            { $group: { _id: '$bucket', ...GROUP_ACCUMULATORS } },
            { $sort: { _id: 1 } },
          ])
          .exec();

        const periods = (rows as any[]).map(finalizePeriod);
        const closed = periods.reduce((sum, p) => sum + p.trades, 0);
        const wins = periods.reduce((sum, p) => sum + p.wins, 0);
        const pnlUsd = periods.reduce((sum, p) => sum + p.pnlUsd, 0);
        const invested = periods.reduce((sum, p) => sum + p.invested, 0);
        // Weighted by trade count: averaging the period averages would let a one-trade week
        // count as much as a twenty-trade one.
        const rWeighted = periods.reduce((sum, p) => sum + (p.avgR ?? 0) * p.trades, 0);
        return {
          tf,
          closed,
          wins,
          winRatePct: closed ? round2((wins / closed) * 100) : 0,
          pnlUsd: round2(pnlUsd),
          invested: round2(invested),
          returnPct: invested > 0 ? round2((pnlUsd / invested) * 100) : null,
          avgR: closed ? round2(rWeighted / closed) : null,
          equity: equityCurve(periods),
        };
      }),
    );
  }

  async trades(opts: {
    universe: TrackedUniverse;
    tf: HistoryTf;
    periodKey?: string;
    groupBy?: Timeframe;
    range?: HistoryRange;
    sort?: TradeSort;
    dir?: SortDir;
    limit?: number;
    offset?: number;
  }): Promise<{ total: number; rows: ResultRow[] }> {
    const limit = Math.min(Math.max(opts.limit ?? 100, 1), 500);
    const offset = Math.max(opts.offset ?? 0, 0);
    const { maxRiskUsd, minRr } = await this.settings.get();
    const lookback = normalizeRange(opts.range);
    const filter: Record<string, unknown> = closedMatch(opts.universe, opts.tf, minRr, lookback);
    if (opts.periodKey) {
      const bucket = periodDateRange(opts.periodKey, opts.groupBy ?? 'Daily');
      if (bucket) {
        // Drill-down intersects the lookback window so a period outside the range stays empty.
        const existing = filter.exitDate as { $gte?: string; $lte?: string } | undefined;
        filter.exitDate = {
          $gte: maxDateStr(existing?.$gte, bucket.$gte),
          $lte: minDateStr(existing?.$lte, bucket.$lte),
        };
      }
    }

    // Resize before sort so a P&L sort follows the same Max risk the cards display.
    const [rows, total] = await Promise.all([
      this.tracked
        .aggregate([
          { $match: filter },
          ...resizeStages(maxRiskUsd),
          { $sort: tradeSortSpec(opts.sort ?? 'date', opts.dir ?? 'desc') },
          { $skip: offset },
          { $limit: limit },
        ])
        .exec(),
      this.tracked.countDocuments(filter).exec(),
    ]);
    return { total, rows: (rows as any[]).map((doc) => toResultRow(resizeDoc(doc, maxRiskUsd))) };
  }
}

/**
 * Overwrite shares / realized P&L from the current Max risk so History always answers "what if
 * every closed trade had been sized under today's setting". Stored CLOSED rows stay frozen.
 */
/** Stages typed loosely so Nest/mongoose PipelineStage unions stay out of the way. */
function resizeStages(maxRiskUsd: number): any[] {
  return [
    {
      $addFields: {
        shares: {
          $let: {
            vars: {
              riskPerShare: {
                $cond: [
                  {
                    $and: [{ $ne: ['$sl', null] }, { $gt: ['$entry', '$sl'] }],
                  },
                  { $subtract: ['$entry', '$sl'] },
                  null,
                ],
              },
            },
            in: {
              $cond: [
                {
                  $and: [
                    { $ne: ['$$riskPerShare', null] },
                    { $gt: ['$$riskPerShare', 0] },
                    { $gt: [maxRiskUsd, 0] },
                  ],
                },
                {
                  $max: [0, { $round: [{ $divide: [maxRiskUsd, '$$riskPerShare'] }, 0] }],
                },
                0,
              ],
            },
          },
        },
        riskUsd: maxRiskUsd,
      },
    },
    {
      $addFields: {
        pnlUsd: {
          $cond: [
            {
              $and: [{ $ne: ['$exitPrice', null] }, { $ne: ['$entry', null] }],
            },
            {
              $round: [
                {
                  $multiply: [{ $subtract: ['$exitPrice', '$entry'] }, '$shares'],
                },
                2,
              ],
            },
            null,
          ],
        },
      },
    },
    {
      $addFields: {
        pnlPct: {
          $let: {
            vars: {
              invested: { $multiply: [{ $ifNull: ['$entry', 0] }, { $ifNull: ['$shares', 0] }] },
            },
            in: {
              $cond: [
                {
                  $and: [{ $gt: ['$$invested', 0] }, { $ne: ['$pnlUsd', null] }],
                },
                {
                  $round: [{ $multiply: [{ $divide: ['$pnlUsd', '$$invested'] }, 100] }, 2],
                },
                null,
              ],
            },
          },
        },
      },
    },
  ];
}

/** Same resize as the aggregation stages, for rows that already went through them (idempotent). */
function resizeDoc(doc: any, maxRiskUsd: number): any {
  const shares = sharesFromRisk(doc.entry, doc.sl, maxRiskUsd);
  const pnl = computePnl(doc.entry, doc.sl, shares, doc.exitPrice);
  return {
    ...doc,
    shares,
    riskUsd: maxRiskUsd,
    pnlUsd: pnl.usd,
    pnlPct: pnl.pct,
    // pnlR is price-based and independent of size — keep the stored multiple.
  };
}

function equityCurve(periods: HistoryPeriod[]): EquityPoint[] {
  let cumulative = 0;
  return periods.map((p) => {
    cumulative += p.pnlUsd;
    return { periodKey: p.periodKey, equity: round2(cumulative) };
  });
}

/**
 * Realized trades only. A sell-to-close break on the bar in progress leaves the record `active`
 * with a `provisionalClose` flag, so it shows in CLOSED for the current period and joins these
 * statistics only once the bar it broke on has finished.
 */
function closedMatch(
  universe: TrackedUniverse,
  tf: HistoryTf,
  minRr: number,
  range: HistoryRange = 'all',
): Record<string, unknown> {
  const match: Record<string, unknown> = { status: 'closed', universe };
  if (tf !== 'All') match.tf = tf;
  if (minRr > 0) match.rrAtEntry = { $gte: minRr };
  const from = lookbackFrom(range);
  if (from) match.exitDate = { $gte: from };
  return match;
}

function normalizeRange(range?: HistoryRange): HistoryRange {
  if (!range || range === 'max') return 'all';
  return range;
}

/** Lower bound on `exitDate` (YYYY-MM-DD) for the History lookback chips. UTC calendar. */
function lookbackFrom(range: HistoryRange, now = new Date()): string | null {
  const normalised = normalizeRange(range);
  if (normalised === 'all') return null;
  const y = now.getUTCFullYear();
  const m = now.getUTCMonth();
  const d = now.getUTCDate();
  if (normalised === 'ytd') return `${y}-01-01`;
  const months =
    normalised === '1m' ? 1 : normalised === '3m' ? 3 : normalised === '6m' ? 6 : 12;
  const from = new Date(Date.UTC(y, m - months, d));
  return from.toISOString().slice(0, 10);
}

function maxDateStr(a?: string, b?: string): string {
  if (!a) return b ?? '';
  if (!b) return a;
  return a > b ? a : b;
}

function minDateStr(a?: string, b?: string): string {
  if (!a) return b ?? '';
  if (!b) return a;
  return a < b ? a : b;
}

/** Confirmed positions only — a provisional record is not a signal yet, so it is not "open". */
function activeMatch(
  universe: TrackedUniverse,
  tf: HistoryTf,
  minRr: number,
): Record<string, unknown> {
  const match: Record<string, unknown> = {
    status: 'active',
    provisional: { $ne: true },
    universe,
  };
  if (tf !== 'All') match.tf = tf;
  // Active / open count uses live RR, matching NEW/VALID on Results.
  if (minRr > 0) match.lastRr = { $gte: minRr };
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
    sort === 'pnl'
      ? p.pnlUsd
      : sort === 'winRate'
        ? p.winRatePct
        : sort === 'trades'
          ? p.trades
          : sort === 'rr'
            ? (p.avgRrEntry ?? 0)
            : 0;
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
