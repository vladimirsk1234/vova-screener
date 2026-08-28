/** History: statistics over closed tracked signals, aggregated in Mongo. */
import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { TRACKED_SIGNAL } from '../db/schemas';
import { FundamentalsService } from '../instruments/fundamentals.service';
import { SettingsService, type FundamentalsFilter } from '../settings/settings.module';
import {
  TIMEFRAMES,
  computePnl,
  holdUnitLabel,
  round2,
  sharesFromRisk,
  toResultRow,
  withYahooTickers,
  type ResultRow,
  type TrackedUniverse,
} from './tracked-signal';
import { sortByUndervaluation } from './uv-sort';

export type HistoryTf = Timeframe | 'All';
/** Lookback on exit date. `max` is an alias of `all` (everything already in DB). */
export type HistoryRange = 'all' | 'ytd' | '1m' | '3m' | '6m' | '1y' | 'max';
export type PeriodSort = 'period' | 'pnl' | 'winRate' | 'trades' | 'rr';
export const TRADE_SORTS = ['date', 'pnl', 'r', 'rr', 'uv', 'interest', 'symbol'] as const;
export type TradeSort = (typeof TRADE_SORTS)[number];
export type SortDir = 'asc' | 'desc';

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
  /** Mean of entry × shares — each trade counts once, unlike invested. */
  avgTradeSizeUsd: number | null;
  /** Mean pnlPct of winning trades; null when there are no winners. */
  avgWinPct: number | null;
  /** Mean pnlPct of losing trades (negative); null when there are no losers. */
  avgLossPct: number | null;
  /** Equal-weight mean pnlPct of every closed trade. */
  avgPnlPct: number | null;
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
  avgR: number | null;
  avgTradeSizeUsd: number | null;
  avgWinPct: number | null;
  avgLossPct: number | null;
  avgPnlPct: number | null;
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
    /** Current Settings Max risk — the size History resizes every closed trade to. */
    maxRiskUsd: number;
    /** closed × maxRiskUsd. */
    totalRiskUsd: number;
    /** Net P&L / total risked; null when nothing was risked. */
    profitToRisk: number | null;
  };
};

const GROUP_ACCUMULATORS = {
  trades: { $sum: 1 },
  wins: { $sum: { $cond: [{ $gt: ['$pnlUsd', 0] }, 1, 0] } },
  losses: { $sum: { $cond: [{ $lt: ['$pnlUsd', 0] }, 1, 0] } },
  pnlUsd: { $sum: { $ifNull: ['$pnlUsd', 0] } },
  invested: {
    $sum: { $multiply: [{ $ifNull: ['$entry', 0] }, { $ifNull: ['$shares', 0] }] },
  },
  avgR: { $avg: '$pnlR' },
  avgRrEntry: { $avg: '$rrAtEntry' },
  avgHold: { $avg: '$holdPeriods' },
  avgTradeSizeUsd: {
    $avg: { $multiply: [{ $ifNull: ['$entry', 0] }, { $ifNull: ['$shares', 0] }] },
  },
  avgWinPct: { $avg: { $cond: [{ $gt: ['$pnlUsd', 0] }, '$pnlPct', null] } },
  avgLossPct: { $avg: { $cond: [{ $lt: ['$pnlUsd', 0] }, '$pnlPct', null] } },
  avgPnlPct: { $avg: '$pnlPct' },
};

@Injectable()
export class HistoryService {
  constructor(
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    private readonly settings: SettingsService,
    private readonly fundamentals: FundamentalsService,
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
    const { maxRiskUsd, minRr, fundamentalsFilter } = await this.settings.get();
    const matching = await this.fundamentalsTickers(universe, tf, fundamentalsFilter);
    const match = withYahooTickers(closedMatch(universe, tf, minRr, range), matching);

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
      this.tracked.countDocuments(withYahooTickers(activeMatch(universe, tf, minRr), matching)).exec(),
      this.byTimeframe(universe, minRr, maxRiskUsd, range, matching),
    ]);

    const raw = (facet?.[0]?.periods ?? []) as any[];
    const ascending = raw.map(finalizePeriod);
    const equity = equityCurve(ascending);

    const totalsRaw = (facet?.[0]?.totals ?? [])[0];
    const totals = totalsRaw ? finalizePeriod(totalsRaw) : emptyPeriod();
    const risk = riskStats(totals.trades, totals.pnlUsd, maxRiskUsd);

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
        losses: totals.losses,
        winRatePct: totals.winRatePct,
        pnlUsd: totals.pnlUsd,
        invested: totals.invested,
        avgR: totals.avgR,
        avgRrEntry: totals.avgRrEntry,
        avgHold: totals.avgHold,
        avgTradeSizeUsd: totals.avgTradeSizeUsd,
        avgWinPct: totals.avgWinPct,
        avgLossPct: totals.avgLossPct,
        avgPnlPct: totals.avgPnlPct,
        ...risk,
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
    matching: string[] | null,
  ): Promise<HistoryTimeframe[]> {
    return Promise.all(
      TIMEFRAMES.map(async (tf) => {
        const facet = await this.tracked
          .aggregate([
            { $match: withYahooTickers(closedMatch(universe, tf, minRr, range), matching) },
            ...resizeStages(maxRiskUsd),
            { $addFields: { bucket: bucketExpression(tf) } },
            {
              $facet: {
                periods: [
                  { $group: { _id: '$bucket', ...GROUP_ACCUMULATORS } },
                  { $sort: { _id: 1 } },
                ],
                totals: [{ $group: { _id: null, ...GROUP_ACCUMULATORS } }],
              },
            },
          ])
          .exec();

        const periods = ((facet?.[0]?.periods ?? []) as any[]).map(finalizePeriod);
        const totals = (facet?.[0]?.totals ?? [])[0]
          ? finalizePeriod((facet[0].totals as any[])[0])
          : emptyPeriod();
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
    hideUnprofitable?: boolean;
  }): Promise<{ total: number; rows: ResultRow[] }> {
    const limit = Math.min(Math.max(opts.limit ?? 100, 1), 500);
    const offset = Math.max(opts.offset ?? 0, 0);
    const { maxRiskUsd, minRr, fundamentalsFilter } = await this.settings.get();
    const lookback = normalizeRange(opts.range);
    const matching = await this.fundamentalsTickers(opts.universe, opts.tf, fundamentalsFilter);
    const filter: Record<string, unknown> = withYahooTickers(
      closedMatch(opts.universe, opts.tf, minRr, lookback),
      matching,
    );
    if (opts.hideUnprofitable) {
      // Keep unknown (not tagged yet) and profitable; hide known EPS≤0 at entry.
      filter.epsPositiveAtEntry = { $ne: false };
    }
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
    const sort = opts.sort ?? 'date';
    const dir = opts.dir ?? 'desc';
    if (sort === 'uv') {
      const docs = await this.tracked
        .aggregate([{ $match: filter }, ...resizeStages(maxRiskUsd)])
        .exec();
      const mapped = (docs as any[]).map((doc) => toResultRow(resizeDoc(doc, maxRiskUsd)));
      const cards = mapped.length
        ? await this.fundamentals.getCardMetricsAll(mapped.map((r) => r.yahooTicker))
        : {};
      const sorted = sortByUndervaluation(mapped, cards, dir);
      return { total: sorted.length, rows: sorted.slice(offset, offset + limit) };
    }

    const [rows, total] = await Promise.all([
      this.tracked
        .aggregate([
          { $match: filter },
          ...resizeStages(maxRiskUsd),
          { $sort: tradeSortSpec(sort, dir) },
          { $skip: offset },
          { $limit: limit },
        ])
        .exec(),
      this.tracked.countDocuments(filter).exec(),
    ]);
    return { total, rows: (rows as any[]).map((doc) => toResultRow(resizeDoc(doc, maxRiskUsd))) };
  }

  /** One FMP classify per History request, shared by closed stats and the open-count. */
  private async fundamentalsTickers(
    universe: TrackedUniverse,
    tf: HistoryTf,
    fundamentalsFilter: FundamentalsFilter,
  ): Promise<string[] | null> {
    if (fundamentalsFilter === 'all') return null;
    const scope: Record<string, unknown> = { universe };
    if (tf !== 'All') scope.tf = tf;
    return this.fundamentals.tickersForFilter(
      fundamentalsFilter,
      await this.tracked.distinct('yahooTicker', scope),
    );
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

function emptyPeriod(): HistoryPeriod {
  return finalizePeriod({ _id: '', trades: 0, wins: 0, losses: 0, pnlUsd: 0, invested: 0 });
}

function nullableRound(n: unknown): number | null {
  if (n == null) return null;
  const v = Number(n);
  return Number.isFinite(v) ? round2(v) : null;
}

function riskStats(closed: number, pnlUsd: number, maxRiskUsd: number) {
  const totalRiskUsd = round2(closed * maxRiskUsd);
  return {
    maxRiskUsd: round2(maxRiskUsd),
    totalRiskUsd,
    profitToRisk: totalRiskUsd > 0 ? round2(pnlUsd / totalRiskUsd) : null,
  };
}

function finalizePeriod(row: any): HistoryPeriod {
  const trades = row.trades ?? 0;
  const wins = row.wins ?? 0;
  return {
    periodKey: row._id ?? 'unknown',
    trades,
    wins,
    losses: row.losses ?? 0,
    winRatePct: trades ? round2((wins / trades) * 100) : 0,
    pnlUsd: round2(row.pnlUsd ?? 0),
    invested: round2(row.invested ?? 0),
    avgR: nullableRound(row.avgR),
    avgRrEntry: nullableRound(row.avgRrEntry),
    avgHold: nullableRound(row.avgHold),
    avgTradeSizeUsd: nullableRound(row.avgTradeSizeUsd),
    avgWinPct: nullableRound(row.avgWinPct),
    avgLossPct: nullableRound(row.avgLossPct),
    avgPnlPct: nullableRound(row.avgPnlPct),
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
