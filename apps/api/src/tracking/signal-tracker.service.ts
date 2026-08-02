/**
 * Turns completed background scans into tracked signals.
 *
 * A tracked position is the Streamlit close scan's trade, nothing else. That scan replays a
 * symbol's whole history — take the long on the bar a buy signal appears, give it up on the bar
 * the sequence closes back through its critical level — and it does so from the bars alone. So
 * the trade a symbol is in does not depend on when this app started watching it: the replay says
 * where the position was entered, and the tracker writes that down.
 *
 * Two things follow, and they are what make the CLOSED list agree with Streamlit's.
 *
 * A break ends a trade whether or not this app ever recorded its start. A symbol closing today is
 * not a buy today — the break puts the sequence down — so it has usually never been reported by a
 * buy scan at all. The scan finds those breaks over the whole universe (`kind: 'sell'` rows), and
 * a trade with no record of its own is written down complete, entry and exit together.
 *
 * A record's entry follows the replay rather than the day the app first noticed the symbol. A
 * position met four months into its run is priced from the bar it actually started on. Imported
 * journal trades are left alone: those entries are the user's own.
 *
 * TP and SL are entry-time numbers that size the trade and state its potential; touching either
 * does not end anything, and a buy setup that stops being valid does not either. Only the break on
 * the bar still in progress waits — that bar can still recover, so the trade reads as closed for
 * the current period and reaches History once the bar is final.
 */
import { Injectable, Logger, type OnModuleInit } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import { runCloseLedger, runStructureOverlay, type CloseTrade, type Timeframe } from '@vova/engine';
import { REJECTION, SCAN_RUN, SIGNAL, TRACKED_SIGNAL } from '../db/schemas';
import { BarsService } from '../market/bars.service';
import { barPeriodKey } from '../scans/period';
import { SettingsService } from '../settings/settings.module';
import {
  computePnl,
  holdPeriods,
  round2,
  sharesFromRisk,
  type ExitReason,
  type TrackedUniverse,
} from './tracked-signal';

const LEDGER_CONCURRENCY = 8;
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

/** A sell-to-close break the scan found over the whole universe. */
type CloseRow = {
  yahooTicker: string;
  symbol: string;
  tvSymbol: string;
  companyName: string;
  entryAsOf: string;
  entry: number;
  entrySl: number | null;
  entryTp: number | null;
  rrAtEntry: number | null;
  exitAsOf: string;
  exit: number;
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
  signalValid?: boolean;
  /** Came from the user's journal, so its entry is theirs and the replay must not overwrite it. */
  imported?: boolean;
};

type Exit = { date: string; price: number; reason: ExitReason };

/** What one symbol's bars say: the trades the close scan takes, and every break in the series. */
type Replay = { trades: CloseTrade[]; open: CloseTrade | null; breaks: Exit[] };

/** Ledger options taken from the run, so a record is priced the way the scan that found it was. */
type LedgerOpts = {
  min_rr: number;
  no_rr_req: boolean;
  use_last_hl_sl: boolean;
  risk_dollars: number;
};

/** Reject reasons that mean "could not evaluate", as opposed to "evaluated, not a buy". */
const UNEVALUATED = ['NO_DATA', 'INSUFFICIENT_DATA'];

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
  /** Open positions the scan evaluated and no longer reports: still running, off the screen. */
  hidden: number;
  /** Breaks on symbols this app was not tracking, written down entry and exit together. */
  adopted: number;
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
    const closes = await this.loadCloses(runId);
    const unevaluated = await this.loadUnevaluated(runId);
    const active = await this.tracked
      .find({ universe, tf, status: 'active' })
      .select(
        'yahooTicker entry tp sl shares openedAsOf openedAt provisional provisionalClose signalValid imported',
      )
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
      hidden: 0,
      adopted: 0,
      dropped: 0,
    };
    const ops: any[] = [];
    // A symbol that closes and is a buy setup again on a later bar starts its next trade here,
    // but only once the close has landed: one active record per symbol is a unique index.
    const reopens: any[] = [];
    const known = new Set<string>();

    // Which trade each record is in, replayed from the bars the scan just cached. This answers
    // both questions at once: whether the position has been given up, and — for one still
    // running — the bar it was entered on, which is rarely the day this app first saw it.
    const ledgerOpts = this.ledgerOpts(run, maxRiskUsd);
    const positions = await this.replay(
      active.filter((doc) => !doc.provisional).map((doc) => doc.yahooTicker),
      tf,
      ledgerOpts,
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

      // Without a floor every bar in the series qualifies and the first one would close the trade
      // at a price from years ago, so fall back to the day the record was written.
      const since = doc.openedAsOf || openedOn(doc);
      const replay = positions.get(doc.yahooTicker);
      const trade = tradeOf(replay, since);
      const exit = exitOf(replay, since);
      if (exit) {
        // Only the bar still forming can take a break back, and it is the one this scan is
        // scanning. A break on any earlier bar is settled — a close scan missed for a week, or a
        // catch-up after a migration, realizes it straight away rather than parking it.
        const pending = !confirmed && barPeriodKey(tf, exit.date) === periodKey;
        ops.push(this.exitOp(doc, trade, exit, tf, maxRiskUsd, pending));
        if (pending) {
          report.pendingClose += 1;
          continue;
        }
        report.closed += 1;
        // The setup is a buy again on a bar after the one that closed the trade, so the symbol
        // opens its next position now instead of waiting for a scan to notice the record is gone.
        if (snapshot && snapshot.asOf > exit.date) {
          reopens.push({
            insertOne: {
              document: this.newDocument(snapshot, null, universe, tf, periodKey, maxRiskUsd, confirmed, runId),
            },
          });
          report.opened += 1;
        }
        continue;
      }

      // No break, so the trade is still open whatever the scan says about the buy setup. A symbol
      // the scan evaluated and no longer reports drops off NEW and VALID and keeps running out of
      // sight; one Yahoo could not price says nothing either way and stays exactly as it was.
      if (!snapshot) {
        const evaluated = !unevaluated.has(doc.yahooTicker);
        const op = this.missingOp(doc, trade, tf, periodKey, maxRiskUsd, evaluated);
        if (op) ops.push(op);
        if (evaluated && doc.signalValid !== false) report.hidden += 1;
        continue;
      }
      ops.push(this.refreshOp(doc, snapshot, periodKey, maxRiskUsd, confirmed, trade, tf));
      report.refreshed += 1;
    }

    // Symbols the scan reports as buy setups for the first time. Their position usually started
    // well before this bar, so the replay is asked where rather than assuming it is today.
    const fresh = [...seen.keys()].filter((ticker) => !known.has(ticker));
    const openings = await this.replay(fresh, tf, ledgerOpts);
    for (const yahooTicker of fresh) {
      const snapshot = seen.get(yahooTicker);
      if (!snapshot) continue;
      ops.push({
        insertOne: {
          document: this.newDocument(
            snapshot,
            openings.get(yahooTicker)?.open ?? null,
            universe,
            tf,
            periodKey,
            maxRiskUsd,
            confirmed,
            runId,
          ),
        },
      });
      report.opened += 1;
    }

    // Breaks on symbols with no record of their own — the bulk of any close list, because a
    // symbol that closes today is not a buy today and so was never opened here.
    const recorded = await this.loadRecorded(universe, tf, closes);
    for (const [yahooTicker, row] of closes) {
      if (known.has(yahooTicker)) continue;
      // Every pass over the period finds the same break again, and a break on a finished bar is
      // written down realized, which takes the record out of `active` and so out of `known`.
      // Without this an hourly cadence stacks a copy of each closed trade an hour.
      if (recorded.has(`${yahooTicker}@${row.exitAsOf}`)) continue;
      const pending = !confirmed && barPeriodKey(tf, row.exitAsOf) === periodKey;
      ops.push({
        insertOne: {
          document: this.adoptedDocument(row, universe, tf, maxRiskUsd, pending, runId),
        },
      });
      report.adopted += 1;
      if (pending) report.pendingClose += 1;
      else report.closed += 1;
    }

    await this.flush(ops);
    await this.flush(reopens);
    this.log.log(
      `tracker ${universe}/${tf} ${periodKey} (${confirmed ? 'period close' : 'live'}): ` +
        `+${report.opened} opened, ${report.refreshed} refreshed, ${report.closed} closed, ` +
        `${report.pendingClose} closing, ${report.adopted} adopted, ` +
        `${report.hidden} hidden, ${report.dropped} dropped`,
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

  /**
   * The sell-to-close breaks this run found across the whole universe — the Streamlit close scan
   * list, symbol for symbol, each carrying the trade behind it rather than only its exit.
   */
  private async loadCloses(runId: string): Promise<Map<string, CloseRow>> {
    const rows = await this.signals
      .find({ runId: new Types.ObjectId(runId), kind: 'sell' })
      .select('payload')
      .lean<any[]>()
      .exec();

    const out = new Map<string, CloseRow>();
    for (const row of rows) {
      const p = row.payload;
      if (!p?.yahooTicker || !p.entryAsOf || !p.exitAsOf) continue;
      if (!Number.isFinite(p.entry) || !Number.isFinite(p.exit)) continue;
      out.set(p.yahooTicker, {
        yahooTicker: p.yahooTicker,
        symbol: p.symbol ?? p.yahooTicker,
        tvSymbol: p.tvSymbol ?? p.symbol ?? p.yahooTicker,
        companyName: p.companyName ?? p.yahooTicker,
        entryAsOf: p.entryAsOf,
        entry: p.entry,
        entrySl: Number.isFinite(p.entrySl) ? p.entrySl : null,
        entryTp: Number.isFinite(p.entryTp) ? p.entryTp : null,
        rrAtEntry: Number.isFinite(p.rrAtEntry) ? p.rrAtEntry : null,
        exitAsOf: p.exitAsOf,
        exit: p.exit,
      });
    }
    return out;
  }

  /**
   * Closes already written down, as `ticker@exit-date` pairs. Same symbol and same exit bar is
   * the same trade; a later break on the same symbol is a different one and gets its own record.
   */
  private async loadRecorded(
    universe: TrackedUniverse,
    tf: Timeframe,
    closes: Map<string, CloseRow>,
  ): Promise<Set<string>> {
    if (!closes.size) return new Set();
    const rows = await this.tracked
      .find({
        universe,
        tf,
        status: 'closed',
        yahooTicker: { $in: [...closes.keys()] },
        exitDate: { $in: [...new Set([...closes.values()].map((row) => row.exitAsOf))] },
      })
      .select('yahooTicker exitDate')
      .lean<Array<{ yahooTicker: string; exitDate: string }>>()
      .exec();
    return new Set(rows.map((row) => `${row.yahooTicker}@${row.exitDate}`));
  }

  /**
   * Symbols this run could not look at. Yahoo failing on a ticker says nothing about the setup,
   * so those positions must not be treated as ones the scan stopped reporting.
   */
  private async loadUnevaluated(runId: string): Promise<Set<string>> {
    const rows = await this.rejections
      .find({ runId: new Types.ObjectId(runId), reason: { $in: UNEVALUATED } })
      .select('symbol')
      .lean<Array<{ symbol: string }>>()
      .exec();
    return new Set(rows.map((row) => row.symbol));
  }

  /** Ledger settings from the run, so a position is replayed the way the scan evaluated it. */
  private ledgerOpts(run: any, maxRiskUsd: number): LedgerOpts {
    return {
      min_rr: Number.isFinite(run.params?.minRr) ? run.params.minRr : 0,
      no_rr_req: run.params?.noRrReq !== false,
      use_last_hl_sl: run.params?.useLastHlSl !== false,
      risk_dollars: maxRiskUsd,
    };
  }

  /**
   * Close-scan replay against the bars the scan just cached, one series per symbol.
   *
   * The breaks are collected alongside the trades because the two answer different questions. A
   * trade says where a position started and so what it is worth; a break ends one, and it ends a
   * record this app opened even on a bar the replay itself was flat on — the record is a position
   * that was taken, and the sell-to-close rule applies to it whatever the replay was doing.
   */
  private async replay(tickers: string[], tf: Timeframe, opts: LedgerOpts) {
    const out = new Map<string, Replay>();
    if (!tickers.length) return out;

    const queue = [...new Set(tickers)];
    const worker = async () => {
      while (queue.length) {
        const ticker = queue.shift();
        if (!ticker) return;
        const bars = await this.bars.getCached(ticker, tf);
        if (!bars?.length) continue;
        const ledger = runCloseLedger(bars, opts);
        if (!ledger) continue;
        const overlay = runStructureOverlay(bars);
        const breaks: Exit[] = [];
        if (overlay) {
          for (let i = 0; i < bars.length; i++) {
            if (overlay.bullish_break[i]) {
              breaks.push({ date: bars[i].date, price: bars[i].close, reason: 'sell_to_close' });
            }
          }
        }
        out.set(ticker, { trades: ledger.trades, open: ledger.open, breaks });
      }
    };
    await Promise.all(Array.from({ length: LEDGER_CONCURRENCY }, () => worker()));
    return out;
  }

  private refreshOp(
    doc: ActiveDoc,
    snapshot: SignalSnapshot,
    periodKey: string,
    maxRiskUsd: number,
    confirm: boolean,
    trade: CloseTrade | null = null,
    tf?: Timeframe,
  ) {
    const aligned = tf ? alignment(doc, trade, tf, maxRiskUsd) : null;
    const entry = aligned ? (aligned.entry as number) : doc.entry;
    const sl = aligned ? (aligned.sl as number | null) : doc.sl;
    const shares = sharesFromRisk(entry, sl, maxRiskUsd);
    const pnl = computePnl(entry, sl, shares, snapshot.entry);
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
            // Back on screen if it had dropped off: this scan reports the setup again.
            signalValid: true,
            ...(confirm ? { provisional: false } : {}),
            ...aligned,
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
  private exitOp(
    doc: ActiveDoc,
    trade: CloseTrade | null,
    exit: Exit,
    tf: Timeframe,
    maxRiskUsd: number,
    pending: boolean,
  ) {
    // The trade closes on the entry the replay gives it, which is the one Streamlit prices its
    // close list from — not the day this app happened to start following the symbol.
    const aligned = alignment(doc, trade, tf, maxRiskUsd);
    const entry = aligned ? (aligned.entry as number) : doc.entry;
    const sl = aligned ? (aligned.sl as number | null) : doc.sl;
    const openedAsOf = aligned ? (aligned.openedAsOf as string) : doc.openedAsOf;
    const shares = aligned ? (aligned.shares as number) : (doc.shares ?? 0);
    const pnl = computePnl(entry, sl, shares, exit.price);
    return {
      updateOne: {
        filter: { _id: doc._id },
        update: {
          $set: {
            ...aligned,
            status: pending ? 'active' : 'closed',
            provisional: false,
            provisionalClose: pending,
            // The period the exit bar belongs to, which for a scan that ran on time is the one
            // being scanned. When a scan is missed for a week, or a trade is re-opened by a
            // migration and its break turns out to be old, the trade is still filed under the
            // period it actually ended in rather than the one the catch-up ran in.
            closedPeriodKey: barPeriodKey(tf, exit.date),
            closedAt: new Date(),
            exitDate: exit.date,
            exitPrice: round2(exit.price),
            exitReason: exit.reason,
            pnlUsd: pnl.usd,
            pnlR: pnl.r,
            pnlPct: pnl.pct,
            holdPeriods: holdPeriods(tf, openedAsOf, exit.date),
          },
          ...(pending ? {} : { $unset: { unrealizedUsd: '', unrealizedR: '', unrealizedPct: '' } }),
        },
      },
    };
  }

  /**
   * The scan has nothing to say about this symbol: no buy setup and no break. Three things can be
   * true at once — a break that was showing on the bar in progress is gone, the setup behind the
   * trade is gone, and the replay has moved the entry — so they are written in one update.
   */
  private missingOp(
    doc: ActiveDoc,
    trade: CloseTrade | null,
    tf: Timeframe,
    periodKey: string,
    maxRiskUsd: number,
    evaluated: boolean,
  ) {
    const set: Record<string, unknown> = { ...alignment(doc, trade, tf, maxRiskUsd) };
    if (doc.provisionalClose) set.provisionalClose = false;
    if (evaluated && doc.signalValid !== false) set.signalValid = false;
    if (!Object.keys(set).length) return null;
    return {
      updateOne: {
        filter: { _id: doc._id },
        update: { $set: set, ...(doc.provisionalClose ? { $unset: EXIT_FIELDS } : {}) },
      },
    };
  }

  /**
   * A symbol the scan reports as a buy setup and this app has no record of. The replay says where
   * its position started, which for a setup that has been valid for weeks is not today — the buy
   * snapshot is only the fallback for a setup the close scan is not holding a trade on.
   */
  private newDocument(
    snapshot: SignalSnapshot,
    open: CloseTrade | null,
    universe: TrackedUniverse,
    tf: Timeframe,
    periodKey: string,
    maxRiskUsd: number,
    confirmed: boolean,
    runId: string,
  ) {
    const entry = open ? round2(open.entry_price) : round2(snapshot.entry);
    const sl = open ? finite(open.entry_sl) : snapshot.sl;
    const tp = open ? finite(open.entry_tp) : snapshot.tp;
    const rr = open ? finite(open.entry_rr) : snapshot.rr;
    const openedAsOf = open ? open.entry_date : snapshot.asOf;
    const openedPeriodKey = barPeriodKey(tf, openedAsOf);
    const shares = sharesFromRisk(entry, sl, maxRiskUsd);
    // A position taken months ago is already worth something, so the card says so from the first
    // scan that meets it rather than reading flat until the next one marks it to market.
    const pnl = computePnl(entry, sl, shares, snapshot.entry);
    return {
      yahooTicker: snapshot.yahooTicker,
      symbol: snapshot.symbol,
      tvSymbol: snapshot.tvSymbol,
      companyName: snapshot.companyName,
      universe,
      tf,
      status: 'active',
      // A trade that started on a bar already finished is a fact, whatever kind of scan found it.
      provisional: !confirmed && openedPeriodKey === periodKey,
      provisionalClose: false,
      signalValid: true,
      openedPeriodKey,
      openedAsOf,
      openedAt: new Date(),
      entry,
      tp,
      sl,
      rrAtEntry: rr,
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
      unrealizedUsd: pnl.usd,
      unrealizedR: pnl.r,
      unrealizedPct: pnl.pct,
      interest: null,
      interestRank: 1,
      runId: new Types.ObjectId(runId),
    };
  }

  /**
   * A break on a symbol with no record of its own. The whole trade goes in at once — the replay
   * knows where it was entered — so CLOSED shows it with the numbers Streamlit shows, and History
   * gets a trade rather than an exit with nothing in front of it.
   *
   * It is written `signalValid: false` because it is not a buy setup: the break is what ended it.
   * If the break turns out to be on a bar that recovers, the next scan takes the exit back off and
   * what is left is an ordinary open position, which is exactly what the replay then says it is.
   */
  private adoptedDocument(
    row: CloseRow,
    universe: TrackedUniverse,
    tf: Timeframe,
    maxRiskUsd: number,
    pending: boolean,
    runId: string,
  ) {
    const entry = round2(row.entry);
    const shares = sharesFromRisk(entry, row.entrySl, maxRiskUsd);
    const pnl = computePnl(entry, row.entrySl, shares, row.exit);
    return {
      yahooTicker: row.yahooTicker,
      symbol: row.symbol,
      tvSymbol: row.tvSymbol,
      companyName: row.companyName,
      universe,
      tf,
      status: pending ? 'active' : 'closed',
      provisional: false,
      provisionalClose: pending,
      signalValid: false,
      openedPeriodKey: barPeriodKey(tf, row.entryAsOf),
      openedAsOf: row.entryAsOf,
      openedAt: atNoon(row.entryAsOf),
      entry,
      tp: row.entryTp,
      sl: row.entrySl,
      rrAtEntry: row.rrAtEntry,
      shares,
      riskUsd: maxRiskUsd,
      lastSeenPeriodKey: barPeriodKey(tf, row.exitAsOf),
      lastSeenAsOf: row.exitAsOf,
      lastSeenAt: new Date(),
      lastPrice: round2(row.exit),
      lastRr: null,
      barsSinceValid: null,
      validSinceAsOf: null,
      isStrong: false,
      closedPeriodKey: barPeriodKey(tf, row.exitAsOf),
      closedAt: new Date(),
      exitDate: row.exitAsOf,
      exitPrice: round2(row.exit),
      exitReason: 'sell_to_close' as ExitReason,
      pnlUsd: pnl.usd,
      pnlR: pnl.r,
      pnlPct: pnl.pct,
      holdPeriods: holdPeriods(tf, row.entryAsOf, row.exitAsOf),
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
 * The trade the replay was holding on the day this record was opened.
 *
 * Only a trade already running then can say where this position started; one the replay took later
 * is a different trade, and one it finished earlier belongs to whoever was watching back then. A
 * record with no such trade keeps the entry it was written with.
 */
function tradeOf(replay: Replay | undefined, since: string): CloseTrade | null {
  if (!replay) return null;
  for (const trade of replay.trades) {
    if (trade.entry_date > since) return null;
    if (trade.exit_date == null || trade.exit_date > since) return trade;
  }
  return null;
}

/**
 * First sell-to-close break after the position was taken, and nothing else. TP and SL are
 * entry-time numbers: SL sizes the position and both state what the setup was worth when it was
 * taken, so price passing through either changes what the trade is worth, not whether it is on.
 */
function exitOf(replay: Replay | undefined, since: string): Exit | null {
  return replay?.breaks.find((brk) => brk.date > since) ?? null;
}

/**
 * The record's entry, brought back to the trade the replay says it is in. This is what makes a
 * position priced the way the Streamlit close scan prices it, whenever this app first met the
 * symbol; a no-op once the two already agree.
 *
 * Imported journal trades are never touched. Those entries are the user's own record of what they
 * paid, and no replay gets to overwrite them.
 */
function alignment(
  doc: ActiveDoc,
  trade: CloseTrade | null,
  tf: Timeframe,
  maxRiskUsd: number,
): Record<string, unknown> | null {
  if (!trade || doc.imported) return null;
  const entry = round2(trade.entry_price);
  if (doc.openedAsOf === trade.entry_date && doc.entry === entry) return null;
  const sl = finite(trade.entry_sl);
  return {
    entry,
    sl,
    tp: finite(trade.entry_tp),
    rrAtEntry: finite(trade.entry_rr),
    openedAsOf: trade.entry_date,
    openedPeriodKey: barPeriodKey(tf, trade.entry_date),
    openedAt: atNoon(trade.entry_date),
    shares: sharesFromRisk(entry, sl, maxRiskUsd),
    riskUsd: maxRiskUsd,
  };
}

function finite(n: number): number | null {
  return Number.isFinite(n) ? round2(n) : null;
}

function atNoon(date: string): Date {
  return new Date(`${date}T12:00:00Z`);
}

function openedOn(doc: ActiveDoc): string {
  return (doc.openedAt ? new Date(doc.openedAt) : new Date()).toISOString().slice(0, 10);
}
