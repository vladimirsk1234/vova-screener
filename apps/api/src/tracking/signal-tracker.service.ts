/**
 * Turns completed background scans into tracked signals.
 *
 * A position ends one way only: the sell-to-close break, exactly as the Streamlit close scan
 * defines it — the close of a bar falls back through the critical level of a bullish sequence.
 * TP and SL are entry-time numbers that size the trade and state its potential; touching either
 * does not end anything, and a buy setup that stops being valid does not either.
 *
 * Every completed Stocks/ETF scan marks positions to market, opens provisional records for symbols
 * seen for the first time, and flags a break on the bar in progress as a provisional close, so the
 * Results screen is live during the session. Only a scan that runs once the period is closed turns
 * a provisional record into a signal, or a provisional close into realized P&L — intra-period noise
 * therefore never reaches History.
 */
import { Injectable, Logger, type OnModuleInit } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import { runStructureOverlay, type OhlcSeries, type Timeframe } from '@vova/engine';
import { SCAN_RUN, SIGNAL, TRACKED_SIGNAL } from '../db/schemas';
import { BarsService } from '../market/bars.service';
import { periodKey } from '../scans/period';
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
  /** Bars of `tf` since the signal became valid, as of this scan: 0 on the bar it appeared on. */
  barsSinceValid: number | null;
  validSinceAsOf: string | null;
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
  provisionalClose?: boolean;
};

type Exit = { date: string; price: number; reason: ExitReason };

/**
 * The exit a provisional close writes, so undoing one leaves an ordinary open position. The flag
 * itself is set back to false rather than removed: every active record then carries it, and code
 * reading one never has to tell "not closing" apart from "written before this existed".
 */
const EXIT_FIELDS = {
  closedPeriodKey: '',
  closedAt: '',
  exitDate: '',
  exitPrice: '',
  exitReason: '',
  pnlUsd: '',
  pnlR: '',
  pnlPct: '',
  holdPeriods: '',
} as const;

export type TrackerReport = {
  universe: TrackedUniverse;
  tf: Timeframe;
  periodKey: string;
  confirmed: boolean;
  opened: number;
  refreshed: number;
  closed: number;
  /** Breaks on the bar in progress: shown in CLOSED, not yet in History. */
  pendingClose: number;
  dropped: number;
};

@Injectable()
export class SignalTrackerService implements OnModuleInit {
  private readonly log = new Logger(SignalTrackerService.name);

  constructor(
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    @InjectModel(SCAN_RUN) private readonly runs: Model<any>,
    @InjectModel(SIGNAL) private readonly signals: Model<any>,
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
    const active = await this.tracked
      .find({ universe, tf, status: 'active' })
      .select('yahooTicker entry tp sl shares openedAsOf openedAt provisional provisionalClose')
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
      pendingClose: 0,
      dropped: 0,
    };
    const ops: any[] = [];
    const known = new Set<string>();

    // Checked on every pass, not just at the close: a break on the bar in progress is what puts a
    // trade in CLOSED for the current period, and the close scan is what makes it real.
    const exits = await this.evaluateExits(
      active.filter((doc) => !doc.provisional),
      tf,
    );

    for (const doc of active) {
      known.add(doc.yahooTicker);
      const snapshot = seen.get(doc.yahooTicker);

      // A record first seen mid-period is not a position yet: the close scan confirms it or drops
      // it, and nothing can end a trade that has not started.
      if (doc.provisional) {
        if (snapshot) {
          ops.push(this.refreshOp(doc, snapshot, periodKey, maxRiskUsd, confirmed));
          report.refreshed += 1;
        } else if (confirmed) {
          ops.push({ deleteOne: { filter: { _id: doc._id } } });
          report.dropped += 1;
        }
        continue;
      }

      const exit = exits.get(String(doc._id))?.exit ?? null;
      if (exit) {
        ops.push(this.exitOp(doc, exit, tf, confirmed));
        if (confirmed) report.closed += 1;
        else report.pendingClose += 1;
        continue;
      }

      // No break, so the trade is still open whatever the scan says about the buy setup. A symbol
      // that fell out of the scan — the setup broke down, or Yahoo could not price it — simply
      // stops being refreshed, which is what hides it from NEW and VALID until it comes back.
      if (!snapshot) {
        if (doc.provisionalClose) ops.push(this.clearPendingCloseOp(doc));
        continue;
      }
      ops.push(this.refreshOp(doc, snapshot, periodKey, maxRiskUsd, confirmed));
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
        `+${report.opened} opened, ${report.refreshed} refreshed, ${report.closed} closed, ` +
        `${report.pendingClose} closing, ${report.dropped} dropped`,
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
        barsSinceValid: Number.isFinite(p.barsSinceValid) ? p.barsSinceValid : null,
        validSinceAsOf: p.validSinceAsOf ?? null,
        asOf: p.asOf ?? '',
      });
    }
    return out;
  }

  /** Sell-to-close check against the bars the scan just cached. */
  private async evaluateExits(docs: ActiveDoc[], tf: Timeframe) {
    const out = new Map<string, { exit: Exit | null }>();
    if (!docs.length) return out;

    const queue = [...docs];
    const worker = async () => {
      while (queue.length) {
        const doc = queue.shift();
        if (!doc) return;
        const bars = await this.bars.getCached(doc.yahooTicker, tf);
        if (!bars?.length) continue;
        out.set(String(doc._id), { exit: findExit(bars, doc) });
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
            // Re-read from the engine on every scan: this is what ages a signal out of NEW.
            barsSinceValid: snapshot.barsSinceValid,
            validSinceAsOf: snapshot.validSinceAsOf,
            isStrong: snapshot.isStrong,
            shares,
            riskUsd: maxRiskUsd,
            unrealizedUsd: pnl.usd,
            unrealizedR: pnl.r,
            unrealizedPct: pnl.pct,
            // The break that put this trade in CLOSED is gone from the bar in progress, so the
            // trade is open again and the exit numbers it carried go with it.
            provisionalClose: false,
            ...(confirm ? { provisional: false } : {}),
          },
          ...(doc.provisionalClose ? { $unset: EXIT_FIELDS } : {}),
        },
      },
    };
  }

  /**
   * A break on the bar in progress is a close for the current period only: the record stays active
   * and out of History until a period-close scan sees the same break on the finished bar. Both
   * write the same exit fields, so CLOSED reads and sorts one shape either way.
   */
  private exitOp(doc: ActiveDoc, exit: Exit, tf: Timeframe, confirmed: boolean) {
    const shares = doc.shares ?? 0;
    const pnl = computePnl(doc.entry, doc.sl, shares, exit.price);
    return {
      updateOne: {
        filter: { _id: doc._id },
        update: {
          $set: {
            status: confirmed ? 'closed' : 'active',
            provisional: false,
            provisionalClose: !confirmed,
            // The period the exit bar belongs to, which for a scan that ran on time is the one
            // being scanned. When a scan is missed for a week, or a trade is re-opened by a
            // migration and its break turns out to be old, the trade is still filed under the
            // period it actually ended in rather than the one the catch-up ran in.
            closedPeriodKey: periodKeyOf(tf, exit.date),
            closedAt: new Date(),
            exitDate: exit.date,
            exitPrice: round2(exit.price),
            exitReason: exit.reason,
            pnlUsd: pnl.usd,
            pnlR: pnl.r,
            pnlPct: pnl.pct,
            holdPeriods: holdPeriods(tf, doc.openedAsOf, exit.date),
          },
          ...(confirmed
            ? { $unset: { unrealizedUsd: '', unrealizedR: '', unrealizedPct: '' } }
            : {}),
        },
      },
    };
  }

  /** The break vanished before the bar closed, and the scan no longer prices the symbol. */
  private clearPendingCloseOp(doc: ActiveDoc) {
    return {
      updateOne: {
        filter: { _id: doc._id },
        update: { $set: { provisionalClose: false }, $unset: EXIT_FIELDS },
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
      provisionalClose: false,
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
      barsSinceValid: snapshot.barsSinceValid,
      validSinceAsOf: snapshot.validSinceAsOf,
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
 * First bullish break after entry — the sell-to-close rule of the Streamlit close scan, where a
 * long is held until the close of a bar falls back through the critical level of the sequence.
 *
 * TP and SL are not consulted. They are entry-time numbers: SL sizes the position and both state
 * what the setup was worth when it was taken, so price passing through either changes what the
 * trade is worth, not whether it is still on.
 */
/** Calendar slot of a bar, read at noon so a date string cannot land in the previous day. */
function periodKeyOf(tf: Timeframe, date: string): string {
  return periodKey(tf, new Date(`${date}T12:00:00-05:00`));
}

function findExit(bars: OhlcSeries, doc: ActiveDoc): Exit | null {
  // Without a floor every bar in the series qualifies and the very first one would close the
  // signal at a price from years ago, so fall back to the day the signal was opened.
  const since = doc.openedAsOf || openedOn(doc);
  const overlay = runStructureOverlay(bars);
  if (!overlay) return null;
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    if (bar.date <= since) continue;
    if (overlay.bullish_break[i]) {
      return { date: bar.date, price: bar.close, reason: 'sell_to_close' };
    }
  }
  return null;
}

function openedOn(doc: ActiveDoc): string {
  return (doc.openedAt ? new Date(doc.openedAt) : new Date()).toISOString().slice(0, 10);
}
