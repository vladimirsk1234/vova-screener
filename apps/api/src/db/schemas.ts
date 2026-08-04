/** Mongoose schemas — mirrors docs/architecture/data-model.md. */
import { Schema } from 'mongoose';

export const INSTRUMENT = 'Instrument';
export const BAR_SERIES = 'BarSeries';
export const SCAN_RUN = 'ScanRun';
export const SIGNAL = 'Signal';
export const REJECTION = 'ScanRejection';
export const TRACKED_SIGNAL = 'TrackedSignal';
export const PRESET = 'Preset';

export const InstrumentSchema = new Schema(
  {
    yahooTicker: { type: String, required: true, unique: true, index: true },
    tvSymbol: { type: String, required: true },
    exchange: String,
    companyName: String,
    assetType: { type: String, default: 'stock' },
    universes: { type: [String], default: [], index: true },
    active: { type: Boolean, default: true },
  },
  { timestamps: true },
);

export const BarSeriesSchema = new Schema(
  {
    yahooTicker: { type: String, required: true },
    interval: { type: String, required: true },
    firstDate: String,
    lastDate: String,
    barCount: { type: Number, default: 0 },
    dates: Buffer,
    open: Buffer,
    high: Buffer,
    low: Buffer,
    close: Buffer,
    volume: Buffer,
    updatedAt: { type: Date, default: Date.now },
  },
  { versionKey: false },
);
BarSeriesSchema.index({ yahooTicker: 1, interval: 1 }, { unique: true });

export const ScanRunSchema = new Schema(
  {
    params: { type: Schema.Types.Mixed, required: true },
    status: {
      type: String,
      enum: ['queued', 'running', 'completed', 'cancelled', 'failed'],
      default: 'queued',
      index: true,
    },
    /** Calendar slot: YYYY-MM-DD | YYYY-Www | YYYY-MM */
    periodKey: { type: String, index: true },
    periodTf: { type: String, index: true },
    trigger: { type: String, enum: ['manual', 'scheduled'], default: 'manual' },
    /**
     * Whether the period was already closed when this scan started. Decided at start, not at
     * finish: an hourly pass that begins at 15:05 and runs long would otherwise look like a
     * period-close scan and let the tracker act on prices captured before the close.
     */
    periodClose: { type: Boolean, default: false },
    /**
     * Last time this period was scanned end to end. A rescan reuses the run document and resets
     * `status`, so this is the only field that answers "does this period have data yet".
     */
    lastCompletedAt: Date,
    /** Oldest last-bar date across the run: how far behind the market its weakest symbol is. */
    asOf: String,
    /** Newest last-bar date across the run — the bar the period screens report on. */
    newestAsOf: String,
    /** Oldest Yahoo pull behind this run's bars (cache age at scan time). */
    barsOldestAt: Date,
    counters: {
      total: { type: Number, default: 0 },
      downloaded: { type: Number, default: 0 },
      evaluated: { type: Number, default: 0 },
      signals: { type: Number, default: 0 },
      /** Sell-to-close breaks the same pass found, which are never also buy signals. */
      closes: { type: Number, default: 0 },
      rejected: { type: Number, default: 0 },
      skipped: { type: Number, default: 0 },
      fromCache: { type: Number, default: 0 },
    },
    reasonCounts: { type: Schema.Types.Mixed, default: {} },
    timings: {
      downloadMs: { type: Number, default: 0 },
      processMs: { type: Number, default: 0 },
      totalMs: { type: Number, default: 0 },
    },
    newSymbols: { type: [String], default: [] },
    summary: { type: Schema.Types.Mixed, default: null },
    cancelRequested: { type: Boolean, default: false },
    error: String,
    startedAt: Date,
    finishedAt: Date,
  },
  { timestamps: true },
);
ScanRunSchema.index({ createdAt: -1 });
ScanRunSchema.index({ periodKey: 1, periodTf: 1, 'params.source': 1 });

export const SignalSchema = new Schema(
  {
    runId: { type: Schema.Types.ObjectId, required: true, index: true },
    kind: { type: String, enum: ['buy', 'sell'], required: true },
    symbol: { type: String, required: true },
    yahooTicker: { type: String, required: true },
    companyName: String,
    isNew: { type: Boolean, default: false },
    isStrong: { type: Boolean, default: false },
    rr: Number,
    payload: { type: Schema.Types.Mixed, required: true },
  },
  { timestamps: true },
);
SignalSchema.index({ runId: 1, symbol: 1 });

export const RejectionSchema = new Schema(
  {
    runId: { type: Schema.Types.ObjectId, required: true, index: true },
    symbol: { type: String, required: true },
    reason: { type: String, required: true },
    /** Engine numbers behind the reject (barDate, close, criticalLevel, seqState, rr, sl, tp, minRr). */
    detail: { type: Schema.Types.Mixed, default: null },
    createdAt: { type: Date, default: Date.now },
  },
  { versionKey: false },
);
// Rejections are audit data, not history: expire after 30 days.
RejectionSchema.index({ createdAt: 1 }, { expireAfterSeconds: 60 * 60 * 24 * 30 });

/**
 * One tracked buy signal per (yahooTicker, tf, universe) while active. Written only by the
 * background scans, so Results and History are plain indexed reads with no per-request maths.
 *
 * `barsSinceValid` is what splits NEW from VALID: `0` means the signal became valid on the latest
 * bar of its timeframe, anything higher means it has been valid for that many bars already.
 *
 * `provisional` marks a signal first seen mid-period. It no longer decides which list a signal
 * shows in, but the period-close scan still decides whether it is confirmed or dropped, and only
 * a confirmed signal can ever be closed — so a signal that comes and goes inside one period
 * never reaches history.
 *
 * `provisionalClose` is the mirror image on the way out: a sell-to-close break on the bar still in
 * progress. The record stays `active` and carries the exit it would realize, so CLOSED shows the
 * trade for the current period while History keeps waiting for the bar to finish. The close scan
 * either turns it into a real close or takes the exit fields back off.
 *
 * `signalValid` is what keeps a position off the screen once its buy setup stops being reported.
 * It is written only by a scan that actually evaluated the symbol, so a Yahoo outage — or simply
 * no scan having run yet — leaves the record showing exactly where it was.
 */
export const TrackedSignalSchema = new Schema(
  {
    yahooTicker: { type: String, required: true },
    symbol: { type: String, required: true },
    tvSymbol: String,
    companyName: String,
    universe: { type: String, enum: ['Stocks', 'ETF'], required: true },
    tf: { type: String, enum: ['Daily', 'Weekly', 'Monthly'], required: true },
    status: { type: String, enum: ['active', 'closed'], default: 'active' },
    provisional: { type: Boolean, default: true },
    /** Sell-to-close break on the bar in progress: shown in CLOSED, still open in History. */
    provisionalClose: { type: Boolean, default: false },
    /** Does the newest scan that could price this symbol still report the buy setup? */
    signalValid: { type: Boolean, default: true },
    /**
     * Came in from the user's trade journal rather than from a scan. Its entry is what they
     * actually paid, so the close-scan replay closes it but never re-prices it.
     */
    imported: { type: Boolean, default: false },
    /**
     * Written by History rebuild (`runCloseLedger` over the bar cache), not by a live scan
     * adoption. Same shape as an adopted close; the flag is for audit / rebuild reports.
     */
    backfilled: { type: Boolean, default: false },

    /** Frozen at first appearance — the entry the P&L is measured from. */
    openedPeriodKey: { type: String, required: true },
    openedAsOf: String,
    openedAt: { type: Date, default: Date.now },
    entry: { type: Number, required: true },
    tp: Number,
    sl: Number,
    rrAtEntry: Number,
    shares: { type: Number, default: 0 },
    riskUsd: { type: Number, default: 0 },

    /** Refreshed by every completed scan of the same (universe, tf). */
    lastSeenPeriodKey: String,
    lastSeenAsOf: String,
    lastSeenAt: Date,
    lastPrice: Number,
    lastRr: Number,
    /** Bars of `tf` since the signal became valid: 0 on the bar it appeared on. Splits NEW/VALID. */
    barsSinceValid: Number,
    validSinceAsOf: String,
    isStrong: { type: Boolean, default: false },
    unrealizedUsd: Number,
    unrealizedR: Number,
    unrealizedPct: Number,

    /** Written by the sell-to-close break, provisionally mid-period and for good at the close. */
    closedPeriodKey: String,
    closedAt: Date,
    exitDate: String,
    exitPrice: Number,
    /**
     * The tracker only ever writes 'sell_to_close'. The other codes stay in the enum for the
     * imported journal, which recorded exits this app no longer takes.
     */
    exitReason: { type: String, enum: ['TP', 'SL', 'sell_to_close', 'signal_lost', 'manual'] },
    pnlUsd: Number,
    pnlR: Number,
    pnlPct: Number,
    holdPeriods: Number,

    interest: { type: String, enum: ['interested', 'not_interested'], default: null },
    /** Sortable form of `interest`: interested 2, unmarked 1, not interested 0. */
    interestRank: { type: Number, default: 1 },
    interestAt: Date,

    runId: Schema.Types.ObjectId,
  },
  { timestamps: true },
);
TrackedSignalSchema.index(
  { yahooTicker: 1, tf: 1, universe: 1 },
  { unique: true, partialFilterExpression: { status: 'active' } },
);
TrackedSignalSchema.index({ universe: 1, tf: 1, status: 1, signalValid: 1, barsSinceValid: 1 });
TrackedSignalSchema.index({ universe: 1, tf: 1, status: 1, openedPeriodKey: -1 });
TrackedSignalSchema.index({ universe: 1, tf: 1, status: 1, closedPeriodKey: -1 });
TrackedSignalSchema.index({ universe: 1, tf: 1, status: 1, lastRr: -1 });
TrackedSignalSchema.index({ universe: 1, tf: 1, status: 1, interestRank: -1 });
TrackedSignalSchema.index({ status: 1, tf: 1, closedPeriodKey: -1 });
// CLOSED spans both a realized close and a break on the bar in progress, so it reads by period
// first and only then asks which of the two a record is.
TrackedSignalSchema.index({ universe: 1, tf: 1, closedPeriodKey: -1 });

export const PresetSchema = new Schema(
  {
    key: { type: String, required: true, unique: true },
    data: { type: Schema.Types.Mixed, default: {} },
  },
  { timestamps: true },
);
