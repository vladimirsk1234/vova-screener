/**
 * Reads for the Results screen. Scans precompute most fields; NEW / VALID also re-check
 * structural age against the bar cache so a setup that died between hourly passes leaves the list.
 */
import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import { signalAge, type Timeframe } from '@vova/engine';
import { SCAN_RUN, TRACKED_SIGNAL } from '../db/schemas';
import { FundamentalsService } from '../instruments/fundamentals.service';
import { BarsService } from '../market/bars.service';
import { barPeriodKey, periodKey as currentPeriodKey } from '../scans/period';
import { SettingsService, type FundamentalsFilter } from '../settings/settings.module';
import { withEntryPremiumFilter, withLiveOrEntryPremiumFilter } from './entry-premium';
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
import { sortByUndervaluation } from './uv-sort';

export const SORT_KEYS = ['rr', 'uv', 'pnl', 'interest', 'symbol'] as const;
export type SortKey = (typeof SORT_KEYS)[number];
export type SortDir = 'asc' | 'desc';

export type ScanMeta = {
  /** Period of the newest scan that produced data — which period CLOSED reports on. */
  periodKey: string;
  asOf: string | null;
  /** Newest bar of the period most of the universe is in. CLOSED reports on that period. */
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
    private readonly settings: SettingsService,
    private readonly bars: BarsService,
    private readonly fundamentals: FundamentalsService,
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
    if (bucket !== 'closed') await this.revalidateLiveAges(universe, tf);
    const { minRr, fundamentalsFilter } = await this.settings.get();
    const filter = bucketFilter(universe, tf, bucket, scan, minRr);
    const filtered = await this.applyFundamentalsFilterAsync(filter, fundamentalsFilter, bucket);

    if (sort === 'uv') {
      const docs = await this.tracked.find(filtered).lean<any[]>().exec();
      const mapped = docs.map(toResultRow);
      const cards = mapped.length
        ? await this.fundamentals.getCardMetricsAll(mapped.map((r) => r.yahooTicker))
        : {};
      const sorted = sortByUndervaluation(mapped, cards, dir);
      return {
        universe,
        tf,
        bucket,
        sort,
        dir,
        total: sorted.length,
        rows: sorted.slice(offset, offset + limit),
        scan,
      };
    }

    const [rows, total] = await Promise.all([
      this.tracked
        .find(filtered)
        .sort(sortSpec(bucket, sort, dir))
        .skip(offset)
        .limit(limit)
        .lean<any[]>()
        .exec(),
      this.tracked.countDocuments(filtered).exec(),
    ]);

    return { universe, tf, bucket, sort, dir, total, rows: rows.map(toResultRow), scan };
  }

  /** Bucket counts for every universe + timeframe, for the tab badges. */
  async summary() {
    const { minRr, fundamentalsFilter } = await this.settings.get();
    const liveTickers =
      fundamentalsFilter === 'all'
        ? null
        : await this.fundamentals.tickersForFilter(
            fundamentalsFilter,
            await this.tracked.distinct('yahooTicker'),
            { warm: false },
          );
    const metas = await Promise.all(
      UNIVERSES.flatMap((universe) =>
        TIMEFRAMES.map(async (tf) => ({ universe, tf, scan: await this.scanMeta(universe, tf) })),
      ),
    );

    const counted = await Promise.all(
      metas.map(async ({ universe, tf, scan }) => {
        // Same live revalidation as list(), so badge counts do not keep dead NEW/VALID cards.
        await this.revalidateLiveAges(universe, tf);
        const [newCount, valid, closed] = await Promise.all([
          this.tracked
            .countDocuments(
              this.applyFundamentalsFilter(
                bucketFilter(universe, tf, 'new', scan, minRr),
                fundamentalsFilter,
                'new',
                liveTickers,
              ),
            )
            .exec(),
          this.tracked
            .countDocuments(
              this.applyFundamentalsFilter(
                bucketFilter(universe, tf, 'valid', scan, minRr),
                fundamentalsFilter,
                'valid',
                liveTickers,
              ),
            )
            .exec(),
          this.tracked
            .countDocuments(
              this.applyFundamentalsFilter(
                bucketFilter(universe, tf, 'closed', scan, minRr),
                fundamentalsFilter,
                'closed',
              ),
            )
            .exec(),
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

  /**
   * Align active rows with the structural age the chart badge uses (`signalAge`, RR off).
   *
   * Hourly scans write `signalValid` / `barsSinceValid`, but a forming bar can kill Seq/Struct
   * between passes — overnight especially, when the cron is off. Reading the bar cache here drops
   * dead setups off NEW/VALID immediately, and brings a recovered setup back (`signalValid: true`
   * with age 0 → NEW again in the same period). Missing cache is treated like an unevaluated scan
   * reject: leave the row alone. Imported journal trades and provisional closes are skipped.
   */
  async revalidateLiveAges(universe: TrackedUniverse, tf: Timeframe): Promise<void> {
    const docs = await this.tracked
      .find({
        universe,
        tf,
        status: 'active',
        provisionalClose: { $ne: true },
        imported: { $ne: true },
      })
      .select('_id yahooTicker barsSinceValid validSinceAsOf signalValid')
      .lean<any[]>()
      .exec();
    if (!docs.length) return;

    const ages = await Promise.all(
      docs.map(async (doc) => {
        const bars = await this.bars.getCached(doc.yahooTicker, tf);
        if (!bars?.length) return null;
        return { doc, age: signalAge(bars) };
      }),
    );

    const ops = [];
    for (const row of ages) {
      if (!row) continue;
      const { doc, age } = row;
      const nextValid = age.barsSinceValid != null;
      const prevValid = doc.signalValid !== false;
      if (
        prevValid === nextValid &&
        doc.barsSinceValid === age.barsSinceValid &&
        (doc.validSinceAsOf ?? null) === (age.validSinceAsOf ?? null)
      ) {
        continue;
      }
      ops.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              barsSinceValid: age.barsSinceValid,
              validSinceAsOf: age.validSinceAsOf,
              signalValid: nextValid,
            },
          },
        },
      });
    }
    if (ops.length) await this.tracked.bulkWrite(ops, { ordered: false });
  }

  /**
   * CLOSED / History-style lists: per-trade `premiumPctAtEntry` only.
   * NEW/VALID: stamp when present; unstamped rows (`$exists: false`) still use
   * today's live `tickersForFilter` until the open stamp or backfill lands.
   */
  private applyFundamentalsFilter(
    match: Record<string, unknown>,
    fundamentalsFilter: FundamentalsFilter,
    bucket: Bucket,
    liveTickers: string[] | null = null,
  ): Record<string, unknown> {
    if (fundamentalsFilter === 'all') return match;
    if (bucket === 'closed') return withEntryPremiumFilter(match, fundamentalsFilter);
    return withLiveOrEntryPremiumFilter(match, fundamentalsFilter, liveTickers);
  }

  private async applyFundamentalsFilterAsync(
    match: Record<string, unknown>,
    fundamentalsFilter: FundamentalsFilter,
    bucket: Bucket,
  ): Promise<Record<string, unknown>> {
    if (fundamentalsFilter === 'all' || bucket === 'closed') {
      return this.applyFundamentalsFilter(match, fundamentalsFilter, bucket);
    }
    const liveTickers = await this.fundamentals.tickersForFilter(
      fundamentalsFilter,
      await this.tracked.distinct('yahooTicker', match),
    );
    return this.applyFundamentalsFilter(match, fundamentalsFilter, bucket, liveTickers);
  }
}

function bucketFilter(
  universe: TrackedUniverse,
  tf: Timeframe,
  bucket: Bucket,
  scan: ScanMeta,
  minRr = 0,
): Record<string, unknown> {
  // CLOSED is "closed on the newest bar's period", and mid-period that includes a sell-to-close
  // break on the bar still running: the trade reads as closed here from the moment the break
  // appears, and only reaches History if the break survives to the final bar.
  //
  // The period comes from the bars, not from the clock, because that is where `closedPeriodKey`
  // comes from. Over a weekend a Monthly scan already runs under the next month while the newest
  // bar it can see is still the last one of this month. `newestAsOf` is the period the universe
  // agrees on rather than any one symbol's newest bar, so neither a halted ticker nor a series
  // Yahoo stamps a day off the grid can point this at a period with nothing in it.
  let filter: Record<string, unknown>;
  if (bucket === 'closed') {
    filter = {
      universe,
      tf,
      closedPeriodKey: scan.newestAsOf ? barPeriodKey(tf, scan.newestAsOf) : scan.periodKey,
      $or: [{ status: 'closed' }, { provisionalClose: true }],
    };
  } else {
    // A trade only ends on a break, so a position whose buy setup stopped being valid is still open
    // — it just leaves the screen. `signalValid: false` is written when a scan evaluates the symbol
    // and does not report it, and again by `revalidateLiveAges` against the bar cache when NEW/VALID
    // are read. Missing cache (or an unevaluated scan reject) leaves the flag alone so a Yahoo
    // outage cannot empty the list.
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
    if (bucket === 'new') {
      filter = { ...live, barsSinceValid: 0, lastSeenPeriodKey: scan.periodKey };
    } else {
      // Exact complement of NEW among the signals still being reported, so the two counts always add
      // up and a record nobody has priced this period lands here rather than nowhere. Records written
      // before `barsSinceValid` existed match `$ne: 0` on the missing field: a signal the tracker is
      // already carrying is by definition not new on this bar.
      filter = {
        ...live,
        $or: [{ barsSinceValid: { $ne: 0 } }, { lastSeenPeriodKey: { $ne: scan.periodKey } }],
      };
    }
  }
  // Global Min RR from Settings: NEW/VALID use live RR (lastRr); CLOSED uses entry RR.
  if (minRr > 0) {
    filter[bucket === 'closed' ? 'rrAtEntry' : 'lastRr'] = { $gte: minRr };
  }
  return filter;
}

/**
 * Mongo sorts missing values first ascending, so a descending RR sort naturally pushes the
 * signals with no computable RR to the end of the list. UV prefers `premiumPctAtEntry`
 * on the trade, then live card premia for rows not yet stamped.
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
