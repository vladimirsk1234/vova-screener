/** Reads for the Results screen. Every field is precomputed by the scans, so these are index scans. */
import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { SCAN_RUN, TRACKED_SIGNAL } from '../db/schemas';
import { barPeriodKey, periodKey as currentPeriodKey } from '../scans/period';
import {
  INTEREST_RANK,
  TIMEFRAMES,
  UNIVERSES,
  toResultRow,
  type Bucket,
  type Interest,
  type ResultRow,
  type TrackedUniverse,
} from './tracked-signal';

export type SortKey = 'rr' | 'pnl' | 'interest' | 'symbol';
export type SortDir = 'asc' | 'desc';

export type ScanMeta = {
  /** Period of the newest scan that produced data — which period CLOSED reports on. */
  periodKey: string;
  asOf: string | null;
  /** Newest bar the scan saw. CLOSED reports on the period this falls in. */
  newestAsOf: string | null;
  finishedAt: string | null;
  running: boolean;
  status: string | null;
};

export type ResultsPage = {
  universe: TrackedUniverse;
  tf: Timeframe;
  bucket: Bucket;
  sort: SortKey;
  dir: SortDir;
  total: number;
  rows: ResultRow[];
  scan: ScanMeta;
};

const RUNNING = ['queued', 'running'];

@Injectable()
export class ResultsService {
  constructor(
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    @InjectModel(SCAN_RUN) private readonly runs: Model<any>,
  ) {}

  /**
   * A rescan of the current period reuses that period's run document and resets its status, so
   * the CLOSED period follows `lastCompletedAt`. Reading `status` instead would empty the list
   * for the few minutes a rescan takes.
   */
  async scanMeta(universe: TrackedUniverse, tf: Timeframe): Promise<ScanMeta> {
    const base = { 'params.source': universe, periodTf: tf };
    const [latest, scanned] = await Promise.all([
      this.runs
        .findOne(base)
        .sort({ periodKey: -1, createdAt: -1 })
        .select('status')
        .lean<any>()
        .exec(),
      this.runs
        .findOne({ ...base, lastCompletedAt: { $exists: true } })
        .sort({ periodKey: -1, lastCompletedAt: -1 })
        .select('periodKey asOf newestAsOf lastCompletedAt')
        .lean<any>()
        .exec(),
    ]);

    return {
      periodKey: scanned?.periodKey ?? currentPeriodKey(tf),
      asOf: scanned?.asOf ?? null,
      newestAsOf: scanned?.newestAsOf ?? scanned?.asOf ?? null,
      finishedAt: scanned?.lastCompletedAt
        ? new Date(scanned.lastCompletedAt).toISOString()
        : null,
      running: RUNNING.includes(latest?.status ?? ''),
      status: latest?.status ?? null,
    };
  }

  async list(opts: {
    universe: TrackedUniverse;
    tf: Timeframe;
    bucket: Bucket;
    sort?: SortKey;
    dir?: SortDir;
    limit?: number;
    offset?: number;
  }): Promise<ResultsPage> {
    const { universe, tf, bucket } = opts;
    const sort = opts.sort ?? 'rr';
    const dir = opts.dir ?? 'desc';
    const limit = Math.min(Math.max(opts.limit ?? 100, 1), 500);
    const offset = Math.max(opts.offset ?? 0, 0);

    const scan = await this.scanMeta(universe, tf);
    const filter = bucketFilter(universe, tf, bucket, scan);

    const [rows, total] = await Promise.all([
      this.tracked
        .find(filter)
        .sort(sortSpec(bucket, sort, dir))
        .skip(offset)
        .limit(limit)
        .lean<any[]>()
        .exec(),
      this.tracked.countDocuments(filter).exec(),
    ]);

    return { universe, tf, bucket, sort, dir, total, rows: rows.map(toResultRow), scan };
  }

  /** Bucket counts for every universe + timeframe, for the tab badges. */
  async summary() {
    const metas = await Promise.all(
      UNIVERSES.flatMap((universe) =>
        TIMEFRAMES.map(async (tf) => ({ universe, tf, scan: await this.scanMeta(universe, tf) })),
      ),
    );

    const counted = await Promise.all(
      metas.map(async ({ universe, tf, scan }) => {
        const [newCount, valid, closed] = await Promise.all([
          this.tracked.countDocuments(bucketFilter(universe, tf, 'new', scan)).exec(),
          this.tracked.countDocuments(bucketFilter(universe, tf, 'valid', scan)).exec(),
          this.tracked.countDocuments(bucketFilter(universe, tf, 'closed', scan)).exec(),
        ]);
        return { universe, tf, scan, counts: { new: newCount, valid, closed } };
      }),
    );

    const out: Record<string, Record<string, unknown>> = {};
    for (const entry of counted) {
      out[entry.universe] ??= {};
      out[entry.universe][entry.tf] = { counts: entry.counts, scan: entry.scan };
    }
    return out;
  }

  /** One tracked signal whatever its state, so the chart can be opened on a trade from History. */
  async byId(id: string): Promise<ResultRow> {
    if (!Types.ObjectId.isValid(id)) throw new NotFoundException('bad signal id');
    const doc = await this.tracked.findById(id).lean<any>().exec();
    if (!doc) throw new NotFoundException('signal not found');
    return toResultRow(doc);
  }

  /** Active tracked signal for a ticker, so the chart screen can show and toggle the mark. */
  async lookup(yahooTicker: string, tf: Timeframe): Promise<ResultRow | null> {
    const doc = await this.tracked
      .findOne({ yahooTicker, tf, status: 'active' })
      .lean<any>()
      .exec();
    return doc ? toResultRow(doc) : null;
  }

  async setInterest(id: string, interest: Interest | null): Promise<ResultRow> {
    if (!Types.ObjectId.isValid(id)) throw new NotFoundException('bad signal id');
    const doc = await this.tracked
      .findByIdAndUpdate(
        id,
        {
          $set: {
            interest,
            interestRank: INTEREST_RANK[interest ?? 'none'],
            interestAt: interest ? new Date() : null,
          },
        },
        { new: true },
      )
      .lean<any>()
      .exec();
    if (!doc) throw new NotFoundException('signal not found');
    return toResultRow(doc);
  }
}

function bucketFilter(
  universe: TrackedUniverse,
  tf: Timeframe,
  bucket: Bucket,
  scan: ScanMeta,
): Record<string, unknown> {
  // CLOSED is "closed on the newest bar's period", and mid-period that includes a sell-to-close
  // break on the bar still running: the trade reads as closed here from the moment the break
  // appears, and only reaches History if the break survives to the final bar.
  //
  // The period comes from the bar, not from the clock, because that is where `closedPeriodKey`
  // comes from. Over a weekend a Monthly scan already runs under the next month while the newest
  // bar it can see is still the last one of this month. It reads the newest bar rather than the
  // oldest: a single halted ticker must not move the whole screen a period back.
  if (bucket === 'closed') {
    return {
      universe,
      tf,
      closedPeriodKey: scan.newestAsOf ? barPeriodKey(tf, scan.newestAsOf) : scan.periodKey,
      $or: [{ status: 'closed' }, { provisionalClose: true }],
    };
  }
  // A trade only ends on a break, so a position whose buy setup stopped being valid is still open
  // — it just leaves the screen. `signalValid: false` is written when a scan evaluates the symbol
  // and does not report it, so this asks the scan directly instead of inferring an answer from
  // which period a record was last priced in: a scan that has not run yet, or one Yahoo throttled,
  // then cannot empty the list.
  //
  // The NEW / VALID split is the bar the signal became valid on, not the period the tracker first
  // recorded it in: a symbol the scan meets for the first time may already have been valid for
  // four bars, and it belongs next to the other four-bar-old trades rather than next to today's
  // breakouts.
  const live = {
    universe,
    tf,
    status: 'active',
    provisionalClose: { $ne: true },
    signalValid: { $ne: false },
  };
  // NEW is a claim about the current bar, so it does ask for a record this period's scan priced.
  if (bucket === 'new') return { ...live, barsSinceValid: 0, lastSeenPeriodKey: scan.periodKey };
  // Exact complement of NEW among the signals still being reported, so the two counts always add
  // up and a record nobody has priced this period lands here rather than nowhere. Records written
  // before `barsSinceValid` existed match `$ne: 0` on the missing field: a signal the tracker is
  // already carrying is by definition not new on this bar.
  return {
    ...live,
    $or: [{ barsSinceValid: { $ne: 0 } }, { lastSeenPeriodKey: { $ne: scan.periodKey } }],
  };
}

/**
 * Mongo sorts missing values first ascending, so a descending RR sort naturally pushes the
 * signals with no computable RR to the end of the list.
 */
function sortSpec(bucket: Bucket, sort: SortKey, dir: SortDir): Record<string, 1 | -1> {
  const order: 1 | -1 = dir === 'asc' ? 1 : -1;
  const spec: Record<string, 1 | -1> = {};
  if (sort === 'rr') spec[bucket === 'closed' ? 'rrAtEntry' : 'lastRr'] = order;
  else if (sort === 'pnl') spec[bucket === 'closed' ? 'pnlUsd' : 'unrealizedUsd'] = order;
  else if (sort === 'interest') spec.interestRank = order;
  else spec.symbol = order;
  if (!('symbol' in spec)) spec.symbol = 1;
  return spec;
}
