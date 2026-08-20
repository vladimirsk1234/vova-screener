/**
 * Imports the pre-tracking trade journal into `trackedSignals`, so the history built by hand
 * before the app tracked signals itself survives the move.
 *
 * Runs on boot and is safe to repeat: every imported journal entry is stamped, so a second pass
 * only picks up what is left. The `trades` collection is never deleted — it stays as the backup.
 */
import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { InjectConnection, InjectModel } from '@nestjs/mongoose';
import { Types, type Connection, type Model, type mongo } from 'mongoose';
import { shortSymbol, type Timeframe } from '@vova/engine';
import { INSTRUMENT, TRACKED_SIGNAL } from '../db/schemas';
import { periodKey } from '../scans/period';
import { SettingsService } from '../settings/settings.module';
import {
  computePnl,
  finiteOrNull,
  holdPeriods,
  INTEREST_RANK,
  sharesFromRisk,
  TIMEFRAMES,
  type ExitReason,
  type Interest,
  type TrackedUniverse,
} from '../tracking/tracked-signal';

const LEGACY_COLLECTION = 'trades';
const BATCH = 500;

/** Journal states that were thrown away at the time and should not come back as signals. */
const DISCARDED = ['dismissed'];

const PERIOD_SHAPE: Record<Timeframe, RegExp> = {
  Daily: /^\d{4}-\d{2}-\d{2}$/,
  Weekly: /^\d{4}-W\d{2}$/,
  Monthly: /^\d{4}-\d{2}$/,
};

const EXIT_REASONS: ExitReason[] = ['TP', 'SL', 'sell_to_close', 'signal_lost', 'manual'];

type LegacyTrade = {
  _id: Types.ObjectId;
  symbol?: string;
  yahooTicker?: string;
  companyName?: string;
  tf?: string;
  openedAt?: Date;
  createdAt?: Date;
  asOf?: string;
  entry?: number;
  tp?: number;
  sl?: number;
  rrAtEntry?: number;
  shares?: number;
  riskUsd?: number;
  status?: string;
  periodKey?: string;
  exitPrice?: number;
  exitDate?: string;
  exitReason?: string;
  pnlUsd?: number;
  pnlR?: number;
  runId?: Types.ObjectId;
};

/** What the instrument list knows about a journal ticker: its universe and its one ticker format. */
type InstrumentRow = {
  universe: TrackedUniverse;
  tvSymbol: string | null;
  companyName: string | null;
};

export type LegacyImportReport = {
  imported: number;
  /** Already tracked from a live scan, so the journal copy would only duplicate it. */
  superseded: number;
  skipped: number;
};

function isoDay(date: Date): string {
  return date.toISOString().slice(0, 10);
}

/** Midday UTC keeps the calendar day intact when the key is resolved in New York. */
function atNoon(day: string): Date {
  return new Date(`${day}T12:00:00Z`);
}

@Injectable()
export class LegacyTradesMigration implements OnModuleInit {
  private readonly log = new Logger('LegacyTrades');

  constructor(
    @InjectConnection() private readonly connection: Connection,
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    @InjectModel(INSTRUMENT) private readonly instruments: Model<any>,
    private readonly settings: SettingsService,
  ) {}

  onModuleInit() {
    void this.run()
      .then((report) => {
        if (report) {
          this.log.log(
            `imported ${report.imported} journal entries ` +
              `(${report.superseded} already tracked, ${report.skipped} unusable)`,
          );
        }
      })
      .catch((err) => {
        this.log.error(`journal import failed: ${(err as Error).message}`);
      });
  }

  async run(): Promise<LegacyImportReport | null> {
    const db = this.connection.db;
    if (!db) return null;
    if (!(await db.listCollections({ name: LEGACY_COLLECTION }).hasNext())) return null;

    const legacy = db.collection<LegacyTrade>(LEGACY_COLLECTION);
    const filter = { migratedAt: { $exists: false }, status: { $nin: DISCARDED } };
    if (!(await legacy.countDocuments(filter, { limit: 1 }))) return null;

    const { maxRiskUsd } = await this.settings.get();
    const report: LegacyImportReport = { imported: 0, superseded: 0, skipped: 0 };

    // Stamping as we go means the cursor's own filter shrinks under us, so page by hand.
    for (;;) {
      const batch = await legacy.find(filter).limit(BATCH).toArray();
      if (!batch.length) break;
      await this.importBatch(legacy, batch, maxRiskUsd, report);
    }
    return report;
  }

  private async importBatch(
    legacy: mongo.Collection<LegacyTrade>,
    batch: LegacyTrade[],
    maxRiskUsd: number,
    report: LegacyImportReport,
  ) {
    const instruments = await this.instrumentByTicker(batch);
    for (const trade of batch) {
      const doc = this.toTrackedSignal(trade, instruments, maxRiskUsd);
      if (!doc) {
        report.skipped += 1;
        await this.stamp(legacy, trade, 'unusable');
        continue;
      }

      // A live scan may already track the same position; the journal copy would collide with the
      // one-active-signal-per-symbol index and add nothing.
      //
      // That index only covers active records, so a trade the app already closed by itself used to
      // let the journal copy through and show the same trade twice in History under two exits. The
      // second check is the same trade rather than the same open position: one opening period per
      // symbol and timeframe.
      const position = { yahooTicker: doc.yahooTicker, tf: doc.tf, universe: doc.universe };
      const taken =
        doc.status === 'active' && (await this.tracked.exists({ ...position, status: 'active' }));
      const sameTrade =
        !taken &&
        (await this.tracked.exists({ ...position, openedPeriodKey: doc.openedPeriodKey }));
      if (taken || sameTrade) {
        report.superseded += 1;
        await this.stamp(legacy, trade, 'superseded');
        continue;
      }

      try {
        const created = await this.tracked.create(doc);
        report.imported += 1;
        await this.stamp(legacy, trade, 'imported', created._id);
      } catch (err) {
        if ((err as { code?: number }).code !== 11000) throw err;
        report.superseded += 1;
        await this.stamp(legacy, trade, 'superseded');
      }
    }
  }

  private async stamp(
    legacy: mongo.Collection<LegacyTrade>,
    trade: LegacyTrade,
    outcome: string,
    trackedSignalId?: Types.ObjectId,
  ) {
    await legacy.updateOne(
      { _id: trade._id },
      { $set: { migratedAt: new Date(), migratedAs: outcome, trackedSignalId: trackedSignalId ?? null } },
    );
  }

  /**
   * The journal predates universes and it recorded whatever ticker string the build of the day
   * used, so both come from the instrument list instead — that is what keeps an imported trade
   * reading the same as a scanned one.
   */
  private async instrumentByTicker(trades: LegacyTrade[]): Promise<Map<string, InstrumentRow>> {
    const tickers = [...new Set(trades.map((t) => t.yahooTicker).filter(Boolean))] as string[];
    const rows = await this.instruments
      .find({ yahooTicker: { $in: tickers } })
      .select('yahooTicker universes tvSymbol companyName')
      .lean<
        Array<{
          yahooTicker: string;
          universes?: string[];
          tvSymbol?: string;
          companyName?: string;
        }>
      >()
      .exec();
    return new Map(
      rows.map((row) => [
        row.yahooTicker,
        {
          universe: (row.universes?.includes('etf') ? 'ETF' : 'Stocks') as TrackedUniverse,
          tvSymbol: row.tvSymbol ?? null,
          companyName: row.companyName ?? null,
        },
      ]),
    );
  }

  private toTrackedSignal(
    trade: LegacyTrade,
    instruments: Map<string, InstrumentRow>,
    maxRiskUsd: number,
  ): Record<string, unknown> | null {
    const yahooTicker = trade.yahooTicker;
    const entry = finiteOrNull(trade.entry);
    if (!yahooTicker || !trade.symbol || entry == null) return null;

    const tf = (TIMEFRAMES.includes(trade.tf as Timeframe) ? trade.tf : 'Daily') as Timeframe;
    const openedAt = trade.openedAt ?? trade.createdAt ?? new Date();
    const openedAsOf = trade.asOf ?? isoDay(openedAt);
    const openedPeriodKey =
      trade.periodKey && PERIOD_SHAPE[tf].test(trade.periodKey)
        ? trade.periodKey
        : periodKey(tf, openedAt);

    const sl = finiteOrNull(trade.sl);
    const closed = trade.status === 'closed';
    // Closed rows keep the size they were closed at — their realized P&L is history. Open ones
    // join everything else under the one global Max Risk.
    const shares = closed ? (trade.shares ?? 0) : sharesFromRisk(entry, sl, maxRiskUsd);

    const interest: Interest | null =
      trade.status === 'interested' || trade.status === 'not_interested' ? trade.status : null;

    const instrument = instruments.get(yahooTicker);
    const tvSymbol = instrument?.tvSymbol ?? trade.symbol;
    const symbol = shortSymbol(tvSymbol);

    const doc: Record<string, unknown> = {
      yahooTicker,
      symbol,
      tvSymbol,
      companyName: instrument?.companyName ?? trade.companyName ?? symbol,
      universe: instrument?.universe ?? 'Stocks',
      tf,
      status: closed ? 'closed' : 'active',
      provisional: false,
      imported: true,
      openedPeriodKey,
      openedAsOf,
      openedAt,
      entry,
      tp: finiteOrNull(trade.tp),
      sl,
      rrAtEntry: finiteOrNull(trade.rrAtEntry),
      shares,
      riskUsd: closed ? (trade.riskUsd ?? 0) : Math.round(shares * (entry - (sl ?? entry)) * 100) / 100,
      interest,
      interestRank: INTEREST_RANK[interest ?? 'none'],
      interestAt: interest ? openedAt : undefined,
      runId: trade.runId,
    };

    if (!closed) return doc;

    const exitPrice = finiteOrNull(trade.exitPrice);
    const exitDate = trade.exitDate ?? openedAsOf;
    const pnl = computePnl(entry, sl, shares, exitPrice);
    return {
      ...doc,
      closedPeriodKey: periodKey(tf, atNoon(exitDate)),
      closedAt: atNoon(exitDate),
      exitDate,
      exitPrice,
      exitReason: EXIT_REASONS.includes(trade.exitReason as ExitReason)
        ? (trade.exitReason as ExitReason)
        : 'manual',
      // Trust the numbers the journal recorded; only fill in what it never had.
      pnlUsd: finiteOrNull(trade.pnlUsd) ?? pnl.usd,
      pnlR: finiteOrNull(trade.pnlR) ?? pnl.r,
      pnlPct: pnl.pct,
      holdPeriods: holdPeriods(tf, openedAsOf, exitDate),
    };
  }
}
