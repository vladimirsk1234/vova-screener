/** Reads for the Results screen. Every field is precomputed by the scans, so these are index scans. */
import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { SCAN_RUN, TRACKED_SIGNAL } from '../db/schemas';
import { periodKey as currentPeriodKey } from '../scans/period';
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
  /** Period of the newest scan that produced data — the boundary between NEW and VALID. */
  periodKey: string;
  asOf: string | null;
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
   * the bucket boundary follows `lastCompletedAt`. Reading `status` instead would move every
   * signal one bucket for the few minutes a rescan takes.
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
        .select('periodKey asOf lastCompletedAt')
        .lean<any>()
        .exec(),
    ]);

    return {
      periodKey: scanned?.periodKey ?? currentPeriodKey(tf),
      asOf: scanned?.asOf ?? null,
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
    const filter = bucketFilter(universe, tf, bucket, scan.periodKey);

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
          this.tracked.countDocuments(bucketFilter(universe, tf, 'new', scan.periodKey)).exec(),
          this.tracked.countDocuments(bucketFilter(universe, tf, 'valid', scan.periodKey)).exec(),
          this.tracked.countDocuments(bucketFilter(universe, tf, 'closed', scan.periodKey)).exec(),
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
  periodKey: string,
): Record<string, unknown> {
  if (bucket === 'closed') {
    return { universe, tf, status: 'closed', closedPeriodKey: periodKey };
  }
  if (bucket === 'new') {
    return { universe, tf, status: 'active', openedPeriodKey: periodKey };
  }
  return { universe, tf, status: 'active', openedPeriodKey: { $lt: periodKey } };
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
