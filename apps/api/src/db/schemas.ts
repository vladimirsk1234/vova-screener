/** Mongoose schemas — mirrors docs/architecture/data-model.md. */
import { Schema } from 'mongoose';

export const INSTRUMENT = 'Instrument';
export const BAR_SERIES = 'BarSeries';
export const SCAN_RUN = 'ScanRun';
export const SIGNAL = 'Signal';
export const REJECTION = 'ScanRejection';
export const TRADE = 'Trade';
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
    asOf: String,
    counters: {
      total: { type: Number, default: 0 },
      downloaded: { type: Number, default: 0 },
      evaluated: { type: Number, default: 0 },
      signals: { type: Number, default: 0 },
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
    createdAt: { type: Date, default: Date.now },
  },
  { versionKey: false },
);
// Rejections are audit data, not history: expire after 30 days.
RejectionSchema.index({ createdAt: 1 }, { expireAfterSeconds: 60 * 60 * 24 * 30 });

export const TradeSchema = new Schema(
  {
    symbol: { type: String, required: true },
    yahooTicker: { type: String, required: true },
    companyName: String,
    tf: { type: String, default: 'Daily' },
    openedAt: { type: Date, default: Date.now },
    asOf: String,
    entry: { type: Number, required: true },
    tp: Number,
    sl: Number,
    rrAtEntry: Number,
    shares: { type: Number, default: 0 },
    riskUsd: { type: Number, default: 0 },
    status: {
      type: String,
      enum: ['open', 'closed', 'dismissed'],
      default: 'open',
      index: true,
    },
    source: { type: String, enum: ['auto', 'manual'], default: 'manual' },
    periodKey: String,
    exitPrice: Number,
    exitDate: String,
    exitReason: String,
    pnlUsd: Number,
    pnlR: Number,
    runId: Schema.Types.ObjectId,
  },
  { timestamps: true },
);
TradeSchema.index({ status: 1, symbol: 1 });
TradeSchema.index({ symbol: 1, tf: 1, status: 1 });

export const PresetSchema = new Schema(
  {
    key: { type: String, required: true, unique: true },
    data: { type: Schema.Types.Mixed, default: {} },
  },
  { timestamps: true },
);
