/**
 * Turns completed background scans into tracked signals.
 *
 * Every completed Stocks/ETF scan marks positions to market and opens provisional records for
 * symbols seen for the first time, so the Results screen is live during the session. Only a scan
 * that runs once the period is closed confirms those records, or closes them with a realized
 * P&L — intra-period noise therefore never reaches History.
 */
import { Injectable, Logger, type OnModuleInit } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import { runStructureOverlay, type OhlcSeries, type Timeframe } from '@vova/engine';
import { REJECTION, SCAN_RUN, SIGNAL, TRACKED_SIGNAL } from '../db/schemas';
import { BarsService } from '../market/bars.service';
import { SettingsService } from '../settings/settings.module';
import {
  computePnl,
  holdPeriods,
  round2,
  sharesFromRisk,
  type ExitReason,
  type TrackedUniverse,
} from './tracked-signal';

const EXIT_CHECK_CONCURRENCY = 8;
const BULK_CHUNK = 500;

type SignalSnapshot = {
  yahooTicker: string;
  symbol: string;
  tvSymbol: string;
  companyName: string;
  entry: number;
  tp: number | null;
  sl: number | null;
  rr: number | null;
  isStrong: boolean;
  asOf: string;
};

type ActiveDoc = {
  _id: Types.ObjectId;
  yahooTicker: string;
  entry: number;
  tp?: number;
  sl?: number;
  shares?: number;
  openedAsOf?: string;
  openedAt?: Date;
  provisional?: boolean;
};

/** Reject reasons that mean "could not evaluate", as opposed to "evaluated, not a buy". */
const UNEVALUATED = ['NO_DATA', 'INSUFFICIENT_DATA'];

type Exit = { date: string; price: number; reason: ExitReason };

export type TrackerReport = {
  universe: TrackedUniverse;
  tf: Timeframe;
  periodKey: string;
  confirmed: boolean;
  opened: number;
  refreshed: number;
  closed: number;
  dropped: number;
};

@Injectable()
export class SignalTrackerService implements OnModuleInit {
  private readonly log = new Logger(SignalTrackerService.name);

  constructor(
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    @InjectModel(SCAN_RUN) private readonly runs: Model<any>,
    @InjectModel(SIGNAL) private readonly signals: Model<any>,
    @InjectModel(REJECTION) private readonly rejections: Model<any>,
    private readonly bars: BarsService,
    private readonly settings: SettingsService,
  ) {}

  onModuleInit() {
    this.settings.onChange(async (next, prev) => {
      if (next.maxRiskUsd === prev.maxRiskUsd) return;
      const count = await this.resizeActive(next.maxRiskUsd);
      this.log.log(
        `max risk ${prev.maxRiskUsd} → ${next.maxRiskUsd}: re-sized ${count} open signals`,
      );
    });
  }

  /**
   * Max risk is one number for every signal, so changing it has to re-size all open positions at
   * once — waiting for the next scan would leave the lists showing sizes from the old risk for up
   * to an hour. Closed signals keep the size they were closed at: their realized P&L is history.
   */
  async resizeActive(maxRiskUsd: number): Promise<number> {
    const docs = await this.tracked
      .find({ status: 'active' })
      .select('entry sl lastPrice')
      .lean<Array<{ _id: Types.ObjectId; entry: number; sl?: number; lastPrice?: number }>>()
      .exec();

    const ops = docs.map((doc) => {
      const shares = sharesFromRisk(doc.entry, doc.sl, maxRiskUsd);
      const pnl = computePnl(doc.entry, doc.sl, shares, doc.lastPrice ?? doc.entry);
      return {
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              shares,
              riskUsd: maxRiskUsd,
              unrealizedUsd: pnl.usd,
              unrealizedR: pnl.r,
              unrealizedPct: pnl.pct,
            },
          },
        },
      };
    });

    await this.flush(ops);
    return ops.length;
  }

  async applyRun(runId: string): Promise<TrackerReport | null> {
    const run = await this.runs.findById(runId).lean<any>().exec();
    if (!run || run.status !== 'completed') return null;
    if (run.params?.direction !== 'buy') return null;

    const universe = run.params?.source as TrackedUniverse;
    if (universe !== 'Stocks' && universe !== 'ETF') return null;

    const tf = (run.periodTf ?? run.params?.tf ?? 'Daily') as Timeframe;
    const periodKey = run.periodKey as string | undefined;
    if (!periodKey) return null;

    // Set when the scan started, so a long hourly pass that happens to finish after the bell
    // cannot confirm or close anything on prices it captured while the market was open.
    const confirmed = run.periodClose === true;
    const { maxRiskUsd } = await this.settings.get();
    const seen = await this.loadSignals(runId);
    const unevaluated = confirmed ? await this.loadUnevaluated(runId) : new Set<string>();
    const active = await this.tracked
      .find({ universe, tf, status: 'active' })
      .select('yahooTicker entry tp sl shares openedAsOf openedAt provisional')
      .lean<ActiveDoc[]>()
      .exec();

    const report: TrackerReport = {
      universe,
      tf,
      periodKey,
      confirmed,
      opened: 0,
      refreshed: 0,
      closed: 0,
      dropped: 0,
    };
    const ops: any[] = [];
    const known = new Set<string>();

    // Only a period-close run needs bars: mid-period runs never close anything.
    const exits = confirmed
      ? await this.evaluateExits(
          active.filter((doc) => !doc.provisional),
          tf,
        )
      : new Map<string, { exit: Exit | null; lastBar: OhlcSeries[number] | null }>();

    for (const doc of active) {
      known.add(doc.yahooTicker);
      const snapshot = seen.get(doc.yahooTicker);

      if (!confirmed) {
        if (!snapshot) continue;
        ops.push(this.refreshOp(doc, snapshot, periodKey, maxRiskUsd, false));
        report.refreshed += 1;
        continue;
      }

      if (doc.provisional) {
        if (snapshot) {
          ops.push(this.refreshOp(doc, snapshot, periodKey, maxRiskUsd, true));
          report.refreshed += 1;
        } else {
          ops.push({ deleteOne: { filter: { _id: doc._id } } });
          report.dropped += 1;
        }
        continue;
      }

      const evaluated = exits.get(String(doc._id));
      if (evaluated?.exit) {
        ops.push(this.closeOp(doc, evaluated.exit, periodKey, tf));
        report.closed += 1;
        continue;
      }
      if (!snapshot) {
        // A symbol Yahoo could not deliver is missing from the scan for a reason that says
        // nothing about the trade, so a data outage must never close a position.
        if (unevaluated.has(doc.yahooTicker)) continue;
        const bar = evaluated?.lastBar;
        if (!bar) continue;
        ops.push(
          this.closeOp(doc, { date: bar.date, price: bar.close, reason: 'signal_lost' }, periodKey, tf),
        );
        report.closed += 1;
        continue;
      }
      ops.push(this.refreshOp(doc, snapshot, periodKey, maxRiskUsd, true));
      report.refreshed += 1;
    }

    for (const [yahooTicker, snapshot] of seen) {
      if (known.has(yahooTicker)) continue;
      ops.push({
        insertOne: {
          document: this.newDocument(snapshot, universe, tf, periodKey, maxRiskUsd, confirmed, runId),
        },
      });
      report.opened += 1;
    }

    await this.flush(ops);
    this.log.log(
      `tracker ${universe}/${tf} ${periodKey} (${confirmed ? 'period close' : 'live'}): ` +
        `+${report.opened} opened, ${report.refreshed} refreshed, ${report.closed} closed, ${report.dropped} dropped`,
    );
    return report;
  }

  private async loadSignals(runId: string): Promise<Map<string, SignalSnapshot>> {
    const rows = await this.signals
      .find({ runId: new Types.ObjectId(runId), kind: 'buy' })
      .select('payload')
      .lean<any[]>()
      .exec();

    const out = new Map<string, SignalSnapshot>();
    for (const row of rows) {
      const p = row.payload;
      if (!p?.yahooTicker || !Number.isFinite(p.entry)) continue;
      out.set(p.yahooTicker, {
        yahooTicker: p.yahooTicker,
        symbol: p.symbol ?? p.yahooTicker,
        tvSymbol: p.tvSymbol ?? p.symbol ?? p.yahooTicker,
        companyName: p.companyName ?? p.yahooTicker,
        entry: p.entry,
        tp: Number.isFinite(p.tp) ? p.tp : null,
        sl: Number.isFinite(p.sl) ? p.sl : null,
        rr: Number.isFinite(p.rr) ? p.rr : null,
        isStrong: Boolean(p.isStrong),
        asOf: p.asOf ?? '',
      });
    }
    return out;
  }

  private async loadUnevaluated(runId: string): Promise<Set<string>> {
    const rows = await this.rejections
      .find({ runId: new Types.ObjectId(runId), reason: { $in: UNEVALUATED } })
      .select('symbol')
      .lean<Array<{ symbol: string }>>()
      .exec();
    return new Set(rows.map((r) => r.symbol));
  }

  /** TP / SL / sell-to-close check against the bars the scan just cached. */
  private async evaluateExits(docs: ActiveDoc[], tf: Timeframe) {
    const out = new Map<string, { exit: Exit | null; lastBar: OhlcSeries[number] | null }>();
    if (!docs.length) return out;

    const queue = [...docs];
    const worker = async () => {
      while (queue.length) {
        const doc = queue.shift();
        if (!doc) return;
        const bars = await this.bars.getCached(doc.yahooTicker, tf);
        if (!bars?.length) continue;
        out.set(String(doc._id), {
          exit: findExit(bars, doc),
          lastBar: bars[bars.length - 1],
        });
      }
    };
    await Promise.all(Array.from({ length: EXIT_CHECK_CONCURRENCY }, () => worker()));
    return out;
  }

  private refreshOp(
    doc: ActiveDoc,
    snapshot: SignalSnapshot,
    periodKey: string,
    maxRiskUsd: number,
    confirm: boolean,
  ) {
    const shares = sharesFromRisk(doc.entry, doc.sl, maxRiskUsd);
    const pnl = computePnl(doc.entry, doc.sl, shares, snapshot.entry);
    return {
      updateOne: {
        filter: { _id: doc._id },
        update: {
          $set: {
            companyName: snapshot.companyName,
            tvSymbol: snapshot.tvSymbol,
            lastSeenPeriodKey: periodKey,
            lastSeenAsOf: snapshot.asOf,
            lastSeenAt: new Date(),
            lastPrice: round2(snapshot.entry),
            lastRr: snapshot.rr,
            isStrong: snapshot.isStrong,
            shares,
            riskUsd: maxRiskUsd,
            unrealizedUsd: pnl.usd,
            unrealizedR: pnl.r,
            unrealizedPct: pnl.pct,
            ...(confirm ? { provisional: false } : {}),
          },
        },
      },
    };
  }

  private closeOp(doc: ActiveDoc, exit: Exit, periodKey: string, tf: Timeframe) {
    const shares = doc.shares ?? 0;
    const pnl = computePnl(doc.entry, doc.sl, shares, exit.price);
    return {
      updateOne: {
        filter: { _id: doc._id },
        update: {
          $set: {
            status: 'closed',
            provisional: false,
            closedPeriodKey: periodKey,
            closedAt: new Date(),
            exitDate: exit.date,
            exitPrice: round2(exit.price),
            exitReason: exit.reason,
            pnlUsd: pnl.usd,
            pnlR: pnl.r,
            pnlPct: pnl.pct,
            holdPeriods: holdPeriods(tf, doc.openedAsOf, exit.date),
          },
          $unset: { unrealizedUsd: '', unrealizedR: '', unrealizedPct: '' },
        },
      },
    };
  }

  private newDocument(
    snapshot: SignalSnapshot,
    universe: TrackedUniverse,
    tf: Timeframe,
    periodKey: string,
    maxRiskUsd: number,
    confirmed: boolean,
    runId: string,
  ) {
    const shares = sharesFromRisk(snapshot.entry, snapshot.sl, maxRiskUsd);
    return {
      yahooTicker: snapshot.yahooTicker,
      symbol: snapshot.symbol,
      tvSymbol: snapshot.tvSymbol,
      companyName: snapshot.companyName,
      universe,
      tf,
      status: 'active',
      provisional: !confirmed,
      openedPeriodKey: periodKey,
      openedAsOf: snapshot.asOf,
      openedAt: new Date(),
      entry: round2(snapshot.entry),
      tp: snapshot.tp,
      sl: snapshot.sl,
      rrAtEntry: snapshot.rr,
      shares,
      riskUsd: maxRiskUsd,
      lastSeenPeriodKey: periodKey,
      lastSeenAsOf: snapshot.asOf,
      lastSeenAt: new Date(),
      lastPrice: round2(snapshot.entry),
      lastRr: snapshot.rr,
      isStrong: snapshot.isStrong,
      unrealizedUsd: 0,
      unrealizedR: 0,
      unrealizedPct: 0,
      interest: null,
      interestRank: 1,
      runId: new Types.ObjectId(runId),
    };
  }

  private async flush(ops: any[]) {
    for (let i = 0; i < ops.length; i += BULK_CHUNK) {
      await this.tracked.bulkWrite(ops.slice(i, i + BULK_CHUNK), { ordered: false });
    }
  }
}

/**
 * First bar after entry that hits SL, TP or a bullish break (sell-to-close), in that order — SL
 * wins when a bar spans both stop and target, because the intrabar path is unknowable.
 */
function findExit(bars: OhlcSeries, doc: ActiveDoc): Exit | null {
  // Without a floor every bar in the series qualifies and the very first one would close the
  // signal at a price from years ago, so fall back to the day the signal was opened.
  const since = doc.openedAsOf || openedOn(doc);
  const overlay = runStructureOverlay(bars);
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    if (bar.date <= since) continue;
    if (doc.sl != null && bar.low <= doc.sl) return { date: bar.date, price: doc.sl, reason: 'SL' };
    if (doc.tp != null && bar.high >= doc.tp) return { date: bar.date, price: doc.tp, reason: 'TP' };
    if (overlay?.bullish_break[i]) {
      return { date: bar.date, price: bar.close, reason: 'sell_to_close' };
    }
  }
  return null;
}

function openedOn(doc: ActiveDoc): string {
  return (doc.openedAt ? new Date(doc.openedAt) : new Date()).toISOString().slice(0, 10);
}
