/**
 * End-to-end check of the signal lifecycle without touching Yahoo.
 *
 * Feeds synthetic scans through SignalTrackerService — a period close, a session pass, the next
 * period close and a period that rolls over with its close scan missed — then asserts the Results
 * buckets and the History aggregation.
 *
 * The rules it is here to hold: a trade ends on the sell-to-close break and on nothing else, a
 * break on the bar in progress shows in CLOSED without reaching History while a break on a bar
 * that has already finished is realized whichever scan finds it, and a position the scan stops
 * reporting stays open and simply drops off the live lists.
 *
 *   npm run smoke:tracker -w @vova/api
 */
import { NestFactory } from '@nestjs/core';
import { getModelToken } from '@nestjs/mongoose';
import mongoose, { Types, type Model } from 'mongoose';
import { encodeSeries, evaluateClose, type OhlcSeries, type SellSignal } from '@vova/engine';
import { AppModule } from '../app.module';
import { BAR_SERIES, REJECTION, SCAN_RUN, SIGNAL, TRACKED_SIGNAL } from '../db/schemas';
import { HistoryService } from '../tracking/history.service';
import { ResultsService } from '../tracking/results.service';
import { SignalTrackerService } from '../tracking/signal-tracker.service';
import { SettingsService } from '../settings/settings.module';
import { check, finish, useSmokeDatabase } from './smoke-harness';

const TICKERS = [
  'ZZTEST-A',
  'ZZTEST-B',
  'ZZTEST-C',
  'ZZTEST-D',
  'ZZTEST-E',
  'ZZTEST-F',
  'ZZTEST-G',
  'ZZTEST-H',
  'ZZTEST-I',
  'ZZTEST-K',
  'ZZTEST-L',
  'ZZTEST-M',
  'ZZTEST-N',
];

/** Rides the climb, then closes back through the critical level: the sell-to-close break. */
const BREAK_TAIL: Tail = [
  { date: '2026-07-15', high: 180.5, low: 179.5, close: 180 },
  { date: '2026-07-16', high: 169, low: 167, close: 168 },
];

/** The same bar, recovering before it finishes: no break left for the close scan to confirm. */
const RECOVERED_TAIL: Tail = [
  { date: '2026-07-15', high: 180.5, low: 179.5, close: 180 },
  { date: '2026-07-16', high: 183, low: 179, close: 182 },
];

/** Climbs through the week, then closes back through the critical level on the Monday. */
const BREAK_ON_20: Tail = [
  { date: '2026-07-15', high: 182.5, low: 181.5, close: 182 },
  { date: '2026-07-16', high: 184.5, low: 183.5, close: 184 },
  { date: '2026-07-17', high: 186.5, low: 185.5, close: 186 },
  { date: '2026-07-20', high: 186, low: 149, close: 150 },
];

/** Recovers, then breaks for good on 2026-07-17 and keeps going down. */
const LATE_BREAK_TAIL: Tail = [
  ...RECOVERED_TAIL,
  { date: '2026-07-17', high: 180, low: 169, close: 170 },
  { date: '2026-07-20', high: 171, low: 167, close: 168 },
];

/** 2026-07-15 and 2026-07-16 are a Wednesday and a Thursday; 20:15Z is 16:15 in New York. */
const DAY1_CLOSE = new Date('2026-07-15T20:15:00Z');
/** An hourly pass that started at 15:05 and ran past the bell — deliberately not a close scan. */
const DAY2_OVERRUN = new Date('2026-07-16T20:05:00Z');
const DAY2_CLOSE = new Date('2026-07-16T20:15:00Z');

type Tail = Array<{ date: string; high: number; low: number; close: number }>;

/** Flat 100.00 series so the only interesting bars are the ones written by hand. */
function series(tail: Tail): OhlcSeries {
  return buildSeries(tail, () => 100);
}

/**
 * Flat, then a steady climb to 178. That is what puts the engine into a bullish sequence with a
 * critical level, so a tail that closes back through it produces the sell-to-close break.
 */
function climbingSeries(tail: Tail): OhlcSeries {
  return buildSeries(tail, (i) => (i < 80 ? 100 : 100 + (i - 80) * 2));
}

/**
 * Higher high, higher low, then a sequence up: the shape the close scan needs before it will take
 * a long at all. It enters on 2026-04-16 at 137.50 with a 134.50 stop, which is months before any
 * of these scans run — which is the point. That entry is what a tracked record has to end up
 * carrying, whenever this app first happens to meet the symbol.
 */
function structuredSeries(tail: Tail): OhlcSeries {
  return buildSeries(tail, (i) => {
    if (i < 48) return 100;
    if (i < 68) return ramp(100, 130, 20, i - 48);
    if (i < 78) return ramp(130, 110, 10, i - 68);
    if (i < 93) return ramp(110, 150, 15, i - 78);
    if (i < 101) return ramp(150, 135, 8, i - 93);
    if (i < 111) return ramp(135, 160, 10, i - 101);
    return ramp(160, 180, 9, i - 111);
  });
}

function ramp(from: number, to: number, over: number, step: number): number {
  return from + ((to - from) * (step + 1)) / over;
}

function buildSeries(tail: Tail, closeAt: (i: number) => number): OhlcSeries {
  const bars: OhlcSeries = [];
  const start = Date.UTC(2026, 0, 5);
  for (let i = 0; i < 120; i++) {
    const date = new Date(start + i * 86_400_000).toISOString().slice(0, 10);
    const close = closeAt(i);
    bars.push({ date, open: close, high: close + 0.5, low: close - 0.5, close, volume: 1_000_000 });
  }
  for (const bar of tail) {
    bars.push({ date: bar.date, open: bar.close, high: bar.high, low: bar.low, close: bar.close, volume: 1_000_000 });
  }
  return bars;
}

async function saveBars(barSeries: Model<any>, yahooTicker: string, bars: OhlcSeries) {
  const enc = encodeSeries(bars);
  await barSeries.updateOne(
    { yahooTicker, interval: '1d' },
    {
      $set: {
        yahooTicker,
        interval: '1d',
        firstDate: enc.firstDate,
        lastDate: enc.lastDate,
        barCount: enc.barCount,
        dates: Buffer.from(enc.dates),
        open: Buffer.from(enc.open),
        high: Buffer.from(enc.high),
        low: Buffer.from(enc.low),
        close: Buffer.from(enc.close),
        volume: Buffer.from(enc.volume),
        updatedAt: new Date(),
      },
    },
    { upsert: true },
  );
}

type Snapshot = {
  ticker: string;
  entry: number;
  tp: number;
  sl: number;
  rr: number;
  /** Bars of the timeframe since the signal became valid — 0 is "became valid on this bar". */
  barsSinceValid: number;
  validSince?: string;
};

async function fakeScan(
  runs: Model<any>,
  signals: Model<any>,
  rejections: Model<any>,
  opts: {
    periodKey: string;
    asOf: string;
    finishedAt: Date;
    periodClose: boolean;
    rows: Snapshot[];
    /** Symbols Yahoo could not deliver, recorded the way a real scan records them. */
    noData?: string[];
    /** Sell-to-close breaks, as a buy pass records them alongside its signals. */
    closes?: SellSignal[];
  },
) {
  const run = await runs.findOneAndUpdate(
    { periodKey: opts.periodKey, periodTf: 'Daily', 'params.source': 'Stocks', trigger: 'smoke' },
    {
      $set: {
        params: { source: 'Stocks', tf: 'Daily', direction: 'buy', noRrReq: true, newOnly: false },
        periodKey: opts.periodKey,
        periodTf: 'Daily',
        periodClose: opts.periodClose,
        trigger: 'smoke',
        status: 'completed',
        asOf: opts.asOf,
        finishedAt: opts.finishedAt,
        lastCompletedAt: opts.finishedAt,
        counters: { signals: opts.rows.length },
      },
    },
    { upsert: true, new: true },
  );

  const runId = String(run._id);
  await signals.deleteMany({ runId: new Types.ObjectId(runId) });
  await rejections.deleteMany({ runId: new Types.ObjectId(runId) });
  if (opts.noData?.length) {
    await rejections.insertMany(
      opts.noData.map((symbol) => ({
        runId: new Types.ObjectId(runId),
        symbol,
        yahooTicker: symbol,
        reason: 'NO_DATA',
      })),
    );
  }
  if (opts.closes?.length) {
    await signals.insertMany(
      opts.closes.map((close) => ({
        runId: new Types.ObjectId(runId),
        kind: 'sell',
        symbol: close.symbol,
        yahooTicker: close.yahooTicker,
        companyName: close.companyName,
        isNew: true,
        isStrong: false,
        rr: close.rrAtEntry,
        payload: close,
      })),
    );
  }
  if (opts.rows.length) {
    await signals.insertMany(
      opts.rows.map((row) => ({
        runId: new Types.ObjectId(runId),
        kind: 'buy',
        symbol: row.ticker,
        yahooTicker: row.ticker,
        isNew: true,
        isStrong: false,
        rr: row.rr,
        payload: {
          kind: 'buy',
          symbol: row.ticker,
          tvSymbol: row.ticker,
          yahooTicker: row.ticker,
          companyName: `${row.ticker} Inc`,
          entry: row.entry,
          tp: row.tp,
          sl: row.sl,
          rr: row.rr,
          isStrong: false,
          barsSinceValid: row.barsSinceValid,
          validSinceAsOf: row.validSince ?? opts.asOf,
          asOf: opts.asOf,
        },
      })),
    );
  }
  return runId;
}

async function main() {
  await useSmokeDatabase('vova-smoke');
  const app = await NestFactory.createApplicationContext(AppModule, { logger: ['error'] });
  const tracker = app.get(SignalTrackerService);
  const results = app.get(ResultsService);
  const history = app.get(HistoryService);
  const settings = app.get(SettingsService);
  const runs = app.get<Model<any>>(getModelToken(SCAN_RUN));
  const signals = app.get<Model<any>>(getModelToken(SIGNAL));
  const tracked = app.get<Model<any>>(getModelToken(TRACKED_SIGNAL));
  const barSeries = app.get<Model<any>>(getModelToken(BAR_SERIES));
  const rejections = app.get<Model<any>>(getModelToken(REJECTION));

  const oldRuns = await runs.find({ trigger: 'smoke' }).select('_id').lean<any[]>();
  await Promise.all([
    tracked.deleteMany({ yahooTicker: { $in: TICKERS } }),
    barSeries.deleteMany({ yahooTicker: { $in: TICKERS } }),
    signals.deleteMany({ runId: { $in: oldRuns.map((r) => r._id) } }),
    rejections.deleteMany({ runId: { $in: oldRuns.map((r) => r._id) } }),
    runs.deleteMany({ trigger: 'smoke' }),
  ]);

  // A drifts up, B runs through its target on day 2, C gaps down through its stop, D only ever
  // shows mid-session, E shows mid-session across two periods whose closes are never scanned,
  // F goes missing because Yahoo failed, G sells to close on a bullish break, H is met by the
  // scanner when it has already been valid for four bars, I becomes valid on the current bar,
  // K breaks intraday only for the bar to recover before it finishes, and L breaks on a bar that
  // has already closed by the time any scan looks at it.
  const tails: Record<string, Tail> = {
    'ZZTEST-A': [
      { date: '2026-07-15', high: 101, low: 99, close: 100 },
      { date: '2026-07-16', high: 106, low: 100, close: 105 },
    ],
    'ZZTEST-B': [
      { date: '2026-07-15', high: 101, low: 99, close: 100 },
      { date: '2026-07-16', high: 121, low: 100, close: 119 },
    ],
    'ZZTEST-C': [
      { date: '2026-07-15', high: 101, low: 99, close: 100 },
      { date: '2026-07-16', high: 100, low: 88, close: 89 },
    ],
    'ZZTEST-D': [
      { date: '2026-07-15', high: 101, low: 99, close: 100 },
      { date: '2026-07-16', high: 101, low: 99, close: 100 },
    ],
    'ZZTEST-E': [
      { date: '2026-07-17', high: 101, low: 99, close: 100 },
      { date: '2026-07-20', high: 101, low: 99, close: 100 },
    ],
    'ZZTEST-F': [
      { date: '2026-07-15', high: 101, low: 99, close: 100 },
      { date: '2026-07-16', high: 101, low: 99, close: 100 },
    ],
    'ZZTEST-G': BREAK_TAIL,
    'ZZTEST-H': [{ date: '2026-07-20', high: 101, low: 99, close: 100 }],
    'ZZTEST-I': [{ date: '2026-07-20', high: 101, low: 99, close: 100 }],
    'ZZTEST-K': BREAK_TAIL,
    'ZZTEST-L': [{ date: '2026-07-15', high: 180.5, low: 179.5, close: 180 }],
    'ZZTEST-M': [{ date: '2026-07-15', high: 182.5, low: 181.5, close: 182 }],
    'ZZTEST-N': [{ date: '2026-07-15', high: 182.5, low: 181.5, close: 182 }],
  };
  const climbers = ['ZZTEST-G', 'ZZTEST-K', 'ZZTEST-L'];
  // M and N are the only two whose bars the close scan will actually take a long on, so they are
  // the two that can say where a position started.
  const structured = ['ZZTEST-M', 'ZZTEST-N'];
  const bars = new Map<string, OhlcSeries>();
  const save = async (ticker: string, rows: OhlcSeries) => {
    bars.set(ticker, rows);
    await saveBars(barSeries, ticker, rows);
  };
  for (const ticker of TICKERS) {
    const build = structured.includes(ticker)
      ? structuredSeries
      : climbers.includes(ticker)
        ? climbingSeries
        : series;
    await save(ticker, build(tails[ticker]));
  }

  /** The close row a buy pass records for a symbol, built the way the scan runner builds it. */
  const closeRow = (ticker: string): SellSignal => {
    const close = evaluateClose({
      bars: bars.get(ticker),
      yahooTicker: ticker,
      tvSymbol: ticker,
      companyName: `${ticker} Inc`,
      params: {
        direction: 'buy',
        minRr: 0,
        riskPerTrade: 100,
        noRrReq: true,
        useLastHlSl: true,
        newOnly: false,
        tf: 'Daily',
      },
    });
    if (!close) throw new Error(`${ticker} has no close signal — fixture is wrong`);
    return close;
  };

  /** Distinct RRs so the sort assertions have something to order by. */
  const RR: Record<string, number> = {
    'ZZTEST-A': 2,
    'ZZTEST-B': 5,
    'ZZTEST-C': 3,
    'ZZTEST-D': 1,
    'ZZTEST-E': 4,
    'ZZTEST-F': 2.5,
    'ZZTEST-G': 1.5,
    'ZZTEST-H': 3.5,
    'ZZTEST-I': 4.5,
    'ZZTEST-K': 1.25,
    'ZZTEST-L': 0.75,
    'ZZTEST-M': 0.5,
    'ZZTEST-N': 0.25,
  };
  const snap = (
    ticker: string,
    entry: number,
    age: { barsSinceValid: number; validSince?: string } = { barsSinceValid: 0 },
  ): Snapshot => ({
    ticker,
    entry,
    tp: entry * 1.2,
    sl: entry * 0.9,
    rr: RR[ticker],
    ...age,
  });

  // 1. Period close: A, B, C, F, G, K and L open as confirmed signals.
  const run1 = await fakeScan(runs, signals, rejections, {
    periodKey: '2026-07-15',
    asOf: '2026-07-15',
    finishedAt: DAY1_CLOSE,
    periodClose: true,
    rows: [
      snap('ZZTEST-A', 100),
      snap('ZZTEST-B', 100),
      snap('ZZTEST-C', 100),
      snap('ZZTEST-F', 100),
      snap('ZZTEST-G', 180),
      snap('ZZTEST-K', 180),
      snap('ZZTEST-L', 180),
    ],
  });
  const report1 = await tracker.applyRun(run1);
  check('day 1 close confirms', [report1?.confirmed, report1?.opened], [true, 7]);
  check(
    'day 1 provisional count',
    await tracked.countDocuments({ yahooTicker: { $in: TICKERS }, provisional: true }),
    0,
  );
  // All seven became valid on the bar the scan evaluated, which is what NEW means.
  check(
    'day 1 signals are all NEW',
    (await results.list({ universe: 'Stocks', tf: 'Daily', bucket: 'new' })).rows.filter((r) =>
      TICKERS.includes(r.yahooTicker),
    ).length,
    7,
  );

  // 2. Session pass: prices move and D appears, but nothing opens or closes for good. It also
  //    finishes after the bell, which must not be enough to make it authoritative.
  const run2 = await fakeScan(runs, signals, rejections, {
    periodKey: '2026-07-16',
    asOf: '2026-07-16',
    finishedAt: DAY2_OVERRUN,
    periodClose: false,
    rows: [
      snap('ZZTEST-A', 105, { barsSinceValid: 1, validSince: '2026-07-15' }),
      snap('ZZTEST-B', 100, { barsSinceValid: 1, validSince: '2026-07-15' }),
      snap('ZZTEST-D', 100),
    ],
    noData: ['ZZTEST-F'],
  });
  const report2 = await tracker.applyRun(run2);
  check('session pass does not confirm', report2?.confirmed, false);
  // G and K both broke on the bar in progress. Nothing is realized until that bar finishes, and
  // C and L are open positions this pass evaluated and no longer reports.
  check(
    'session pass closes nothing for good',
    [report2?.closed, report2?.pendingClose, report2?.hidden, report2?.dropped],
    [0, 2, 2, 0],
  );
  check(
    'D is provisional',
    (await tracked.findOne({ yahooTicker: 'ZZTEST-D' }).lean<any>())?.provisional,
    true,
  );
  check(
    'A marked to market',
    (await tracked.findOne({ yahooTicker: 'ZZTEST-A' }).lean<any>())?.unrealizedUsd,
    50, // $100 risk / $10 stop distance = 10 shares × $5
  );

  // A break mid-period reads as closed on the Results screen and stays out of History.
  const closing = await tracked.findOne({ yahooTicker: 'ZZTEST-G' }).lean<any>();
  check(
    'a break on the bar in progress closes the trade for this period only',
    [closing?.status, closing?.provisionalClose, closing?.exitReason, closing?.exitPrice],
    ['active', true, 'sell_to_close', 168],
  );
  check(
    'CLOSED holds it, History does not',
    [
      (await results.list({ universe: 'Stocks', tf: 'Daily', bucket: 'closed' })).rows
        .filter((r) => TICKERS.includes(r.yahooTicker))
        .map((r) => r.symbol),
      (await history.report({ universe: 'Stocks', tf: 'Daily', groupBy: 'Daily' })).totals.closed,
    ],
    [['ZZTEST-G', 'ZZTEST-K'], 0],
  );

  // K's bar recovers before the bell, so the break it showed at 15:05 never happened.
  await save('ZZTEST-K', climbingSeries(RECOVERED_TAIL));

  // 3. Next period close: G realizes the break it showed mid-period, K's went away, B ran through
  //    its target and C through its stop without either ending anything, D was never confirmed and
  //    F is missing because Yahoo failed. A and C keep running, A rolls into VALID.
  const run3 = await fakeScan(runs, signals, rejections, {
    periodKey: '2026-07-16',
    asOf: '2026-07-16',
    finishedAt: DAY2_CLOSE,
    periodClose: true,
    rows: [
      snap('ZZTEST-A', 105, { barsSinceValid: 1, validSince: '2026-07-15' }),
      snap('ZZTEST-B', 100, { barsSinceValid: 1, validSince: '2026-07-15' }),
      snap('ZZTEST-K', 182, { barsSinceValid: 1, validSince: '2026-07-15' }),
    ],
    noData: ['ZZTEST-F'],
  });
  const report3 = await tracker.applyRun(run3);
  check(
    'day 2 close',
    [report3?.closed, report3?.pendingClose, report3?.hidden, report3?.dropped, report3?.refreshed],
    [1, 0, 0, 1, 3],
  );

  // TP and SL size a trade and say what it was worth. Price passing through either changes the
  // number the position is carrying, never whether the position is still on.
  const stopped = await tracked.findOne({ yahooTicker: 'ZZTEST-C' }).lean<any>();
  check(
    'a stop being taken out does not close a trade',
    [stopped?.status, stopped?.exitReason],
    ['active', undefined],
  );
  const target = await tracked.findOne({ yahooTicker: 'ZZTEST-B' }).lean<any>();
  check(
    'a target being reached does not close a trade',
    [target?.status, target?.exitReason],
    ['active', undefined],
  );
  const broke = await tracked.findOne({ yahooTicker: 'ZZTEST-G' }).lean<any>();
  check(
    'G sells to close on the bullish break',
    [broke?.status, broke?.provisionalClose, broke?.exitReason, broke?.exitPrice, broke?.exitDate, broke?.pnlUsd],
    ['closed', false, 'sell_to_close', 168, '2026-07-16', -72],
  );
  const recovered = await tracked.findOne({ yahooTicker: 'ZZTEST-K' }).lean<any>();
  check(
    'a break that does not survive the bar leaves the trade open',
    [recovered?.status, recovered?.provisionalClose, recovered?.exitReason, recovered?.pnlUsd],
    ['active', false, undefined, undefined],
  );
  check('D dropped', await tracked.countDocuments({ yahooTicker: 'ZZTEST-D' }), 0);
  // A scan that could not price a symbol says nothing about the trade, so F stays open.
  const outage = await tracked.findOne({ yahooTicker: 'ZZTEST-F' }).lean<any>();
  check('a data outage does not close a signal', [outage?.status, outage?.exitReason], ['active', undefined]);

  const buckets = await Promise.all(
    (['new', 'valid', 'closed'] as const).map((bucket) =>
      results.list({ universe: 'Stocks', tf: 'Daily', bucket }),
    ),
  );
  // C and L are still open but no scan reports their setup any more, so neither is on screen. F is
  // there on last week's numbers because no scan has managed to price it; only G is CLOSED.
  check(
    'buckets new/valid/closed',
    buckets.map((b) => b.rows.filter((r) => TICKERS.includes(r.yahooTicker)).map((r) => r.symbol)),
    [[], ['ZZTEST-B', 'ZZTEST-F', 'ZZTEST-A', 'ZZTEST-K'], ['ZZTEST-G']],
  );
  // Being dropped from a scan and being missed by one look identical from the outside, and they
  // must not be treated the same: only the first is the scan saying the setup is gone.
  check(
    'evaluated and gone is hidden, unevaluated keeps showing',
    (
      await tracked
        .find({ yahooTicker: { $in: ['ZZTEST-C', 'ZZTEST-F'] } })
        .sort({ yahooTicker: 1 })
        .select('status signalValid')
        .lean<any[]>()
    ).map((d) => [d.status, d.signalValid]),
    [
      ['active', false],
      ['active', true],
    ],
  );
  check(
    'valid marked to market',
    (await results.list({ universe: 'Stocks', tf: 'Daily', bucket: 'valid', sort: 'pnl' })).rows
      .filter((r) => TICKERS.includes(r.yahooTicker))
      .map((r) => [r.symbol, r.pnlUsd]),
    [
      ['ZZTEST-A', 50],
      ['ZZTEST-K', 12],
      ['ZZTEST-B', 0],
      ['ZZTEST-F', 0],
    ],
  );
  // A is one bar into its trade, and the card says so rather than counting scans.
  const aged = await results.list({ universe: 'Stocks', tf: 'Daily', bucket: 'valid' });
  check(
    'valid carries the age of the signal',
    aged.rows
      .filter((r) => r.yahooTicker === 'ZZTEST-A')
      .map((r) => [r.barsSinceValid, r.validSinceAsOf]),
    [[1, '2026-07-15']],
  );

  // CLOSED orders by RR at entry, the live buckets by the RR of the latest scan.
  const byRr = async (bucket: 'new' | 'valid' | 'closed', dir: 'asc' | 'desc') =>
    (await results.list({ universe: 'Stocks', tf: 'Daily', bucket, sort: 'rr', dir })).rows
      .filter((r) => TICKERS.includes(r.yahooTicker))
      .map((r) => r.symbol);
  check('closed sorted by RR desc', await byRr('closed', 'desc'), ['ZZTEST-G']);
  check('valid sorted by RR asc', await byRr('valid', 'asc'), [
    'ZZTEST-K',
    'ZZTEST-A',
    'ZZTEST-F',
    'ZZTEST-B',
  ]);

  const marked = await results.setInterest(buckets[1].rows[0].id, 'interested');
  check('interest saved', marked.interest, 'interested');
  check(
    'interest sorts first',
    (await results.list({ universe: 'Stocks', tf: 'Daily', bucket: 'valid', sort: 'interest' }))
      .rows[0].interest,
    'interested',
  );

  // 4. A close scan is missed (machine asleep at 16:15) so E stays provisional while the period
  //    rolls over, and the scan meets H and I for the first time: H has already been valid for
  //    four bars, I became valid on the bar being scanned. Only I is new on this bar.
  const run4 = await fakeScan(runs, signals, rejections, {
    periodKey: '2026-07-17',
    asOf: '2026-07-17',
    finishedAt: new Date('2026-07-17T15:00:00Z'),
    periodClose: false,
    rows: [snap('ZZTEST-E', 100)],
  });
  const report4 = await tracker.applyRun(run4);
  check('E opens NEW on the bar it became valid', await byRr('new', 'desc'), ['ZZTEST-E']);
  // A, B, F and K were on screen after the close scan and this pass no longer reports any of them.
  check('a live pass takes positions off the screen too', report4?.hidden, 4);

  // L broke on 2026-07-17 and nothing was scanning that afternoon. The bar is finished by the time
  // this session pass looks at it, so the trade is realized now rather than waiting for a close
  // scan — and the setup is a buy again on 2026-07-20, which starts the next trade in the same pass.
  await save('ZZTEST-L', climbingSeries(LATE_BREAK_TAIL));
  // M breaks on the bar this pass is scanning, and no scan has ever reported it as a buy — which
  // is the ordinary case, because a symbol closing today is not a buy today. N is a buy for the
  // first time, four months into a position the close scan has been holding since 2026-04-16.
  await save('ZZTEST-M', structuredSeries(BREAK_ON_20));
  const run5 = await fakeScan(runs, signals, rejections, {
    periodKey: '2026-07-20',
    asOf: '2026-07-20',
    finishedAt: new Date('2026-07-20T15:00:00Z'),
    periodClose: false,
    rows: [
      snap('ZZTEST-E', 100, { barsSinceValid: 1, validSince: '2026-07-17' }),
      snap('ZZTEST-H', 100, { barsSinceValid: 4, validSince: '2026-07-14' }),
      snap('ZZTEST-I', 100),
      snap('ZZTEST-L', 171),
      snap('ZZTEST-N', 182, { barsSinceValid: 4, validSince: '2026-07-14' }),
    ],
    closes: [closeRow('ZZTEST-M')],
  });
  const report5 = await tracker.applyRun(run5);
  check(
    'a break on a finished bar is realized by a live pass',
    [report5?.confirmed, report5?.closed, report5?.pendingClose, report5?.adopted],
    [false, 1, 1, 1],
  );

  // The whole trade goes in at once, priced from the bar the replay entered it on — the numbers
  // the Streamlit close scan reports, for a symbol this app had no record of.
  const adopted = await tracked.findOne({ yahooTicker: 'ZZTEST-M' }).lean<any>();
  check(
    'a break on a symbol we never opened is written down entry and exit together',
    [
      adopted?.status,
      adopted?.provisionalClose,
      adopted?.openedAsOf,
      adopted?.entry,
      adopted?.sl,
      adopted?.exitDate,
      adopted?.exitPrice,
      adopted?.pnlUsd,
    ],
    ['active', true, '2026-04-16', 137.5, 134.5, '2026-07-20', 150, 412.5],
  );
  check(
    'and it shows in CLOSED',
    (await results.list({ universe: 'Stocks', tf: 'Daily', bucket: 'closed' })).rows
      .filter((r) => TICKERS.includes(r.yahooTicker))
      .map((r) => r.symbol),
    ['ZZTEST-M'],
  );

  // N is met four bars into its buy signal and four months into its position. The record takes the
  // trade, not the day it was met: the scan's own 182 is only where the position is marked to.
  const joined = await tracked.findOne({ yahooTicker: 'ZZTEST-N' }).lean<any>();
  check(
    'a position met mid-trade is priced from the bar it started on',
    [
      joined?.entry,
      joined?.sl,
      joined?.tp,
      joined?.rrAtEntry,
      joined?.openedAsOf,
      joined?.openedPeriodKey,
      joined?.provisional,
      joined?.shares,
      joined?.unrealizedUsd,
    ],
    [137.5, 134.5, 150.5, 4.33, '2026-04-16', '2026-04-16', false, 33, 1468.5],
  );
  const settled = await tracked.findOne({ yahooTicker: 'ZZTEST-L', status: 'closed' }).lean<any>();
  check(
    'and it is a real close, filed under the bar it broke on',
    [
      settled?.provisionalClose,
      settled?.exitReason,
      settled?.exitDate,
      settled?.closedPeriodKey,
      settled?.pnlUsd,
    ],
    [false, 'sell_to_close', '2026-07-17', '2026-07-17', -60],
  );
  const restarted = await tracked.findOne({ yahooTicker: 'ZZTEST-L', status: 'active' }).lean<any>();
  check(
    'the same symbol starts its next trade in the same pass',
    [restarted?.entry, restarted?.openedAsOf, restarted?.provisional],
    [171, '2026-07-20', true],
  );

  const stale = await tracked.findOne({ yahooTicker: 'ZZTEST-E' }).lean<any>();
  check('E still provisional a period later', [stale?.provisional, stale?.openedPeriodKey], [true, '2026-07-17']);
  const rolled = await Promise.all(
    (['new', 'valid'] as const).map((bucket) =>
      results.list({ universe: 'Stocks', tf: 'Daily', bucket }),
    ),
  );
  check(
    'the bar the signal became valid on decides the bucket, not the record',
    rolled.map((b) => b.rows.filter((r) => TICKERS.includes(r.yahooTicker)).map((r) => r.symbol)),
    // Default sort is RR descending. E aged out of NEW without ever being confirmed and H arrived
    // already four bars old; the trades this scan did not report are open but off screen.
    [
      ['ZZTEST-I', 'ZZTEST-L'],
      ['ZZTEST-E', 'ZZTEST-H', 'ZZTEST-N'],
    ],
  );
  check('NEW sorted by RR desc', await byRr('new', 'desc'), ['ZZTEST-I', 'ZZTEST-L']);
  const arrivedOld = await results.list({ universe: 'Stocks', tf: 'Daily', bucket: 'valid' });
  check(
    'a signal first seen four bars into its run reports that age',
    arrivedOld.rows
      .filter((r) => r.yahooTicker === 'ZZTEST-H')
      .map((r) => [r.barsSinceValid, r.validSinceAsOf]),
    [[4, '2026-07-14']],
  );
  // NEW and VALID are complements among the signals still being reported, so nothing on screen can
  // fall between them or land in both. Everything a scan has dropped is open and hidden.
  check(
    'bucket counts add up to every signal still being reported',
    rolled[0].rows.length + rolled[1].rows.length,
    await tracked.countDocuments({
      universe: 'Stocks',
      tf: 'Daily',
      status: 'active',
      signalValid: { $ne: false },
      provisionalClose: { $ne: true },
    }),
  );
  check(
    'trades left open and off screen',
    await tracked.countDocuments({
      universe: 'Stocks',
      tf: 'Daily',
      status: 'active',
      signalValid: false,
      provisionalClose: { $ne: true },
    }),
    5, // A, B, C, F and K all still run: none of them broke.
  );

  const stats = await history.report({ universe: 'Stocks', tf: 'Daily', groupBy: 'Daily' });
  const day = stats.periods.find((p) => p.periodKey === '2026-07-16');
  // Only the break reaches History: the stop C took out and the target B reached are not exits.
  check('history bucket', [day?.trades, day?.wins, day?.pnlUsd], [1, 0, -72]);
  check('history avg RR at entry', day?.avgRrEntry, 1.5);
  check('every closed trade sold to close', stats.exitReasons, [{ reason: 'sell_to_close', count: 2 }]);
  // A, B, C, F, K, N and M: E, H, I and L's restart are active but still provisional, so none of
  // them is an open position yet. M's break is on the bar in progress, which can still take it
  // back, so it is a position until that bar finishes.
  check('history counts confirmed positions only', stats.totals.active, 7);
  check(
    'history periods sortable by RR',
    (await history.report({ universe: 'Stocks', tf: 'Daily', groupBy: 'Daily', sort: 'rr', dir: 'desc' }))
      .periods[0]?.periodKey,
    '2026-07-16',
  );
  check(
    'growth is reported per timeframe',
    stats.timeframes.map((t) => [t.tf, t.closed, t.pnlUsd, t.equity.length]),
    [
      ['Daily', 2, -132, 2],
      ['Weekly', 0, 0, 0],
      ['Monthly', 0, 0, 0],
    ],
  );

  // 5. A close scan catches up two periods late and finds a break that happened on 2026-07-17.
  //    The trade belongs to the period it ended in, not to the period the catch-up ran in.
  await save('ZZTEST-K', climbingSeries(LATE_BREAK_TAIL));
  const run6 = await fakeScan(runs, signals, rejections, {
    periodKey: '2026-07-20',
    asOf: '2026-07-20',
    finishedAt: new Date('2026-07-20T20:15:00Z'),
    periodClose: true,
    rows: [snap('ZZTEST-E', 100, { barsSinceValid: 1, validSince: '2026-07-17' })],
  });
  const report6 = await tracker.applyRun(run6);
  // K's old break, and M's break now that the bar it happened on has finished.
  check('the catch-up close finds the old break', report6?.closed, 2);
  const realized = await tracked.findOne({ yahooTicker: 'ZZTEST-M' }).lean<any>();
  check(
    'an adopted trade reaches History once its bar is final',
    [realized?.status, realized?.provisionalClose, realized?.pnlUsd],
    ['closed', false, 412.5],
  );
  const late = await tracked.findOne({ yahooTicker: 'ZZTEST-K' }).lean<any>();
  check(
    'a trade is filed under the period it ended in',
    [late?.status, late?.exitDate, late?.closedPeriodKey],
    ['closed', '2026-07-17', '2026-07-17'],
  );
  check(
    'and History buckets it there',
    (await history.report({ universe: 'Stocks', tf: 'Daily', groupBy: 'Daily' })).periods.map((p) => [
      p.periodKey,
      p.trades,
    ]),
    [
      // M's adopted trade, realized now that the bar it broke on is final.
      ['2026-07-20', 1],
      ['2026-07-17', 2],
      ['2026-07-16', 1],
    ],
  );

  // 5b. The break is still on M's last bar, so every later pass finds it again. Realizing it took
  //     the record out of the active set, which is what the adoption checks against — so this is
  //     where an hourly cadence would quietly stack a copy of the trade an hour.
  const run7 = await fakeScan(runs, signals, rejections, {
    periodKey: '2026-07-21',
    asOf: '2026-07-21',
    finishedAt: new Date('2026-07-21T20:15:00Z'),
    periodClose: true,
    rows: [],
    closes: [closeRow('ZZTEST-M')],
  });
  const report7 = await tracker.applyRun(run7);
  check('a close already written down is not adopted twice', report7?.adopted, 0);
  check(
    'and the symbol still has exactly one record',
    await tracked.countDocuments({ yahooTicker: 'ZZTEST-M' }),
    1,
  );

  // 6. Max risk is one number for every signal, so raising it re-sizes the open ones immediately.
  //    A: $200 risk over a $10 stop distance = 20 shares, and its $5 gain is worth $100.
  await settings.put({ maxRiskUsd: 200 });
  const resized = await tracked.findOne({ yahooTicker: 'ZZTEST-A' }).lean<any>();
  check(
    'raising max risk re-sizes open signals',
    [resized?.shares, resized?.riskUsd, resized?.unrealizedUsd],
    [20, 200, 100],
  );
  const untouched = await tracked.findOne({ yahooTicker: 'ZZTEST-G' }).lean<any>();
  check('closed signals keep their size', [untouched?.shares, untouched?.pnlUsd], [6, -72]);
  // History still re-sizes under the current Max risk: G is $18 to the stop, so $200 → 11 shares.
  const histAt200 = await history.report({ universe: 'Stocks', tf: 'Daily', groupBy: 'Daily' });
  check(
    'history P&L follows current max risk',
    histAt200.periods.find((p) => p.periodKey === '2026-07-16')?.pnlUsd,
    -132,
  );
  await settings.put({ maxRiskUsd: 100 });

  const drill = await history.trades({
    universe: 'Stocks',
    tf: 'Daily',
    groupBy: 'Daily',
    periodKey: '2026-07-16',
    sort: 'rr',
    dir: 'desc',
  });
  check(
    'history drill-down sorted by RR',
    drill.rows.map((r) => r.symbol),
    ['ZZTEST-G'],
  );

  await Promise.all([
    tracked.deleteMany({ yahooTicker: { $in: TICKERS } }),
    barSeries.deleteMany({ yahooTicker: { $in: TICKERS } }),
    signals.deleteMany({
      runId: { $in: [run1, run2, run3, run4, run5, run6, run7].map((id) => new Types.ObjectId(id)) },
    }),
    rejections.deleteMany({
      runId: { $in: [run1, run2, run3, run4, run5, run6, run7].map((id) => new Types.ObjectId(id)) },
    }),
    runs.deleteMany({ trigger: 'smoke' }),
  ]);

  await app.close();
  await mongoose.disconnect();
  await finish('TRACKER');
}

main().catch((err) => {
  console.error('TRACKER FAIL', err);
  process.exit(1);
});
