/**
 * End-to-end check of the signal lifecycle without touching Yahoo.
 *
 * Feeds synthetic scans through SignalTrackerService — a period close, a session pass, the next
 * period close and a period that rolls over with its close scan missed — then asserts the Results
 * buckets and the History aggregation.
 *
 *   npm run smoke:tracker -w @vova/api
 */
import * as net from 'node:net';
import { NestFactory } from '@nestjs/core';
import { getModelToken } from '@nestjs/mongoose';
import mongoose, { Types, type Model } from 'mongoose';
import { encodeSeries, type OhlcSeries } from '@vova/engine';
import { AppModule } from '../app.module';
import { resolveMongoUri } from '../db/local-mongo';
import { BAR_SERIES, REJECTION, SCAN_RUN, SIGNAL, TRACKED_SIGNAL } from '../db/schemas';
import { HistoryService } from '../tracking/history.service';
import { ResultsService } from '../tracking/results.service';
import { SignalTrackerService } from '../tracking/signal-tracker.service';
import { SettingsService } from '../settings/settings.module';

const TICKERS = [
  'ZZTEST-A',
  'ZZTEST-B',
  'ZZTEST-C',
  'ZZTEST-D',
  'ZZTEST-E',
  'ZZTEST-F',
  'ZZTEST-G',
];

/** 2026-07-15 and 2026-07-16 are a Wednesday and a Thursday; 20:15Z is 16:15 in New York. */
const DAY1_CLOSE = new Date('2026-07-15T20:15:00Z');
/** An hourly pass that started at 15:05 and ran past the bell — deliberately not a close scan. */
const DAY2_OVERRUN = new Date('2026-07-16T20:05:00Z');
const DAY2_CLOSE = new Date('2026-07-16T20:15:00Z');

const EMBEDDED_PORT = 27019;

let failures = 0;

function portIsOpen(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = net
      .connect({ host: '127.0.0.1', port })
      .on('connect', () => (socket.end(), resolve(true)))
      .on('error', () => resolve(false));
    socket.setTimeout(500, () => (socket.destroy(), resolve(false)));
  });
}

/**
 * Bucket boundaries follow the newest scan in the database, so the fixtures need a database of
 * their own — a real background scan sitting next to them would move NEW and CLOSED off the
 * synthetic period. The embedded Mongo has a persistent data directory shared with the dev
 * server, hence the separate database name rather than a separate server.
 */
async function useSmokeDatabase() {
  const base =
    process.env.MONGO_URI ??
    ((await portIsOpen(EMBEDDED_PORT))
      ? `mongodb://127.0.0.1:${EMBEDDED_PORT}/vova?directConnection=true`
      : await resolveMongoUri());
  const uri = new URL(base);
  uri.pathname = '/vova-smoke';
  process.env.MONGO_URI = uri.toString();
}

function check(label: string, actual: unknown, expected: unknown) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (!ok) failures += 1;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${JSON.stringify(actual)}` + (ok ? '' : ` (want ${JSON.stringify(expected)})`));
}

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

type Snapshot = { ticker: string; entry: number; tp: number; sl: number; rr: number };

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
        reason: 'NO_DATA',
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
          asOf: opts.asOf,
        },
      })),
    );
  }
  return runId;
}

async function main() {
  await useSmokeDatabase();
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
  // F goes missing because Yahoo failed, G sells to close on a bullish break.
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
    // Rides the climb, then closes back through the critical level on day 2.
    'ZZTEST-G': [
      { date: '2026-07-15', high: 180.5, low: 179.5, close: 180 },
      { date: '2026-07-16', high: 169, low: 167, close: 168 },
    ],
  };
  for (const ticker of TICKERS) {
    const build = ticker === 'ZZTEST-G' ? climbingSeries : series;
    await saveBars(barSeries, ticker, build(tails[ticker]));
  }

  /** Distinct RRs so the sort assertions have something to order by. */
  const RR: Record<string, number> = {
    'ZZTEST-A': 2,
    'ZZTEST-B': 5,
    'ZZTEST-C': 3,
    'ZZTEST-D': 1,
    'ZZTEST-E': 4,
    'ZZTEST-F': 2.5,
    'ZZTEST-G': 1.5,
  };
  const snap = (ticker: string, entry: number): Snapshot => ({
    ticker,
    entry,
    tp: entry * 1.2,
    sl: entry * 0.9,
    rr: RR[ticker],
  });

  // 1. Period close: A, B, C, F and G open as confirmed signals.
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
    ],
  });
  const report1 = await tracker.applyRun(run1);
  check('day 1 close confirms', [report1?.confirmed, report1?.opened], [true, 5]);
  check(
    'day 1 provisional count',
    await tracked.countDocuments({ yahooTicker: { $in: TICKERS }, provisional: true }),
    0,
  );

  // 2. Session pass: prices move and D appears, but nothing opens or closes for good. It also
  //    finishes after the bell, which must not be enough to make it authoritative.
  const run2 = await fakeScan(runs, signals, rejections, {
    periodKey: '2026-07-16',
    asOf: '2026-07-16',
    finishedAt: DAY2_OVERRUN,
    periodClose: false,
    rows: [snap('ZZTEST-A', 105), snap('ZZTEST-B', 100), snap('ZZTEST-D', 100)],
  });
  const report2 = await tracker.applyRun(run2);
  check('session pass does not confirm', report2?.confirmed, false);
  check('session pass closes nothing', [report2?.closed, report2?.dropped], [0, 0]);
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

  // 3. Next period close: B takes its target, C is stopped out, G sells to close on the bullish
  //    break, D was never confirmed, F is only missing because Yahoo failed, A rolls into VALID.
  const run3 = await fakeScan(runs, signals, rejections, {
    periodKey: '2026-07-16',
    asOf: '2026-07-16',
    finishedAt: DAY2_CLOSE,
    periodClose: true,
    rows: [snap('ZZTEST-A', 105), snap('ZZTEST-B', 100), snap('ZZTEST-G', 180)],
    noData: ['ZZTEST-F'],
  });
  const report3 = await tracker.applyRun(run3);
  check('day 2 close', [report3?.closed, report3?.dropped, report3?.refreshed], [3, 1, 1]);

  const closed = await tracked.findOne({ yahooTicker: 'ZZTEST-C' }).lean<any>();
  check('C stopped out', [closed?.status, closed?.exitReason, closed?.exitPrice], ['closed', 'SL', 90]);
  check('C realized P&L', [closed?.pnlUsd, closed?.pnlR], [-100, -1]);
  const won = await tracked.findOne({ yahooTicker: 'ZZTEST-B' }).lean<any>();
  check('B took its target', [won?.exitReason, won?.exitPrice, won?.pnlUsd, won?.pnlR], ['TP', 120, 200, 2]);
  // Neither stop nor target was touched: the structure break is what ends the trade.
  const broke = await tracked.findOne({ yahooTicker: 'ZZTEST-G' }).lean<any>();
  check(
    'G sells to close on the bullish break',
    [broke?.exitReason, broke?.exitPrice, broke?.exitDate, broke?.pnlUsd],
    ['sell_to_close', 168, '2026-07-16', -72],
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
  check(
    'buckets new/valid/closed',
    buckets.map((b) => b.rows.filter((r) => TICKERS.includes(r.yahooTicker)).length),
    [0, 2, 3],
  );
  check(
    'valid marked to market',
    (await results.list({ universe: 'Stocks', tf: 'Daily', bucket: 'valid', sort: 'pnl' })).rows
      .filter((r) => TICKERS.includes(r.yahooTicker))
      .map((r) => [r.symbol, r.pnlUsd]),
    [
      ['ZZTEST-A', 50],
      ['ZZTEST-F', 0],
    ],
  );

  // CLOSED orders by RR at entry, the live buckets by the RR of the latest scan.
  const byRr = async (bucket: 'new' | 'valid' | 'closed', dir: 'asc' | 'desc') =>
    (await results.list({ universe: 'Stocks', tf: 'Daily', bucket, sort: 'rr', dir })).rows
      .filter((r) => TICKERS.includes(r.yahooTicker))
      .map((r) => r.symbol);
  check('closed sorted by RR desc', await byRr('closed', 'desc'), ['ZZTEST-B', 'ZZTEST-C', 'ZZTEST-G']);
  check('closed sorted by RR asc', await byRr('closed', 'asc'), ['ZZTEST-G', 'ZZTEST-C', 'ZZTEST-B']);

  const marked = await results.setInterest(buckets[1].rows[0].id, 'interested');
  check('interest saved', marked.interest, 'interested');
  check(
    'interest sorts first',
    (await results.list({ universe: 'Stocks', tf: 'Daily', bucket: 'valid', sort: 'interest' }))
      .rows[0].interest,
    'interested',
  );

  // 4. A close scan is missed (machine asleep at 16:15) and E stays provisional while the period
  //    rolls over. Age alone must not promote it: VALID is earned by surviving a close.
  const run4 = await fakeScan(runs, signals, rejections, {
    periodKey: '2026-07-17',
    asOf: '2026-07-17',
    finishedAt: new Date('2026-07-17T15:00:00Z'),
    periodClose: false,
    rows: [snap('ZZTEST-E', 100)],
  });
  await tracker.applyRun(run4);
  const run5 = await fakeScan(runs, signals, rejections, {
    periodKey: '2026-07-20',
    asOf: '2026-07-20',
    finishedAt: new Date('2026-07-20T15:00:00Z'),
    periodClose: false,
    rows: [snap('ZZTEST-E', 100)],
  });
  await tracker.applyRun(run5);

  const stale = await tracked.findOne({ yahooTicker: 'ZZTEST-E' }).lean<any>();
  check('E still provisional a period later', [stale?.provisional, stale?.openedPeriodKey], [true, '2026-07-17']);
  const rolled = await Promise.all(
    (['new', 'valid'] as const).map((bucket) =>
      results.list({ universe: 'Stocks', tf: 'Daily', bucket }),
    ),
  );
  check(
    'unconfirmed signal stays in NEW',
    rolled.map((b) => b.rows.filter((r) => TICKERS.includes(r.yahooTicker)).map((r) => r.symbol)),
    // Default sort is RR descending: F still carries its opening 2.5, A was refreshed down to 2.
    [['ZZTEST-E'], ['ZZTEST-F', 'ZZTEST-A']],
  );
  check('NEW sorted by RR desc', await byRr('new', 'desc'), ['ZZTEST-E']);

  const stats = await history.report({ tf: 'Daily', groupBy: 'Daily' });
  const day = stats.periods.find((p) => p.periodKey === '2026-07-16');
  check('history bucket', [day?.trades, day?.wins, day?.pnlUsd], [3, 1, 28]);
  check('history avg RR at entry', day?.avgRrEntry, 3.17); // B 5, C 3, G 1.5
  // A and F: E is active but still provisional, so it is not an open position yet.
  check('history counts confirmed positions only', stats.totals.active, 2);
  check(
    'history periods sortable by RR',
    (await history.report({ tf: 'Daily', groupBy: 'Daily', sort: 'rr', dir: 'desc' })).periods[0]
      ?.periodKey,
    '2026-07-16',
  );

  // 5. Max risk is one number for every signal, so raising it re-sizes the open ones immediately.
  //    A: $200 risk over a $10 stop distance = 20 shares, and its $5 gain is worth $100.
  await settings.put({ maxRiskUsd: 200 });
  const resized = await tracked.findOne({ yahooTicker: 'ZZTEST-A' }).lean<any>();
  check(
    'raising max risk re-sizes open signals',
    [resized?.shares, resized?.riskUsd, resized?.unrealizedUsd],
    [20, 200, 100],
  );
  const untouched = await tracked.findOne({ yahooTicker: 'ZZTEST-C' }).lean<any>();
  check('closed signals keep their size', [untouched?.shares, untouched?.pnlUsd], [10, -100]);
  await settings.put({ maxRiskUsd: 100 });

  const drill = await history.trades({
    tf: 'Daily',
    groupBy: 'Daily',
    periodKey: '2026-07-16',
    sort: 'rr',
    dir: 'desc',
  });
  check(
    'history drill-down sorted by RR',
    drill.rows.map((r) => r.symbol),
    ['ZZTEST-B', 'ZZTEST-C', 'ZZTEST-G'],
  );

  await Promise.all([
    tracked.deleteMany({ yahooTicker: { $in: TICKERS } }),
    barSeries.deleteMany({ yahooTicker: { $in: TICKERS } }),
    signals.deleteMany({
      runId: { $in: [run1, run2, run3, run4, run5].map((id) => new Types.ObjectId(id)) },
    }),
    rejections.deleteMany({
      runId: { $in: [run1, run2, run3, run4, run5].map((id) => new Types.ObjectId(id)) },
    }),
    runs.deleteMany({ trigger: 'smoke' }),
  ]);

  await app.close();
  await mongoose.disconnect();
  console.log(failures ? `\n${failures} check(s) failed` : '\nTRACKER OK');
  process.exit(failures ? 1 : 0);
}

main().catch((err) => {
  console.error('TRACKER FAIL', err);
  process.exit(1);
});
