/**
 * End-to-end check of the signal lifecycle without touching Yahoo.
 *
 * Feeds three synthetic scans through SignalTrackerService — a period close, a session pass
 * and the next period close — then asserts the Results buckets and the History aggregation.
 *
 *   npm run smoke:tracker -w @vova/api
 */
import { NestFactory } from '@nestjs/core';
import { getModelToken } from '@nestjs/mongoose';
import mongoose, { Types, type Model } from 'mongoose';
import { encodeSeries, type OhlcSeries } from '@vova/engine';
import { AppModule } from '../app.module';
import { BAR_SERIES, SCAN_RUN, SIGNAL, TRACKED_SIGNAL } from '../db/schemas';
import { HistoryService } from '../tracking/history.service';
import { ResultsService } from '../tracking/results.service';
import { SignalTrackerService } from '../tracking/signal-tracker.service';

const TICKERS = ['ZZTEST-A', 'ZZTEST-B', 'ZZTEST-C', 'ZZTEST-D'];

/** 2026-07-15 and 2026-07-16 are a Wednesday and a Thursday; 20:15Z is 16:15 in New York. */
const DAY1_CLOSE = new Date('2026-07-15T20:15:00Z');
/** An hourly pass that started at 15:05 and ran past the bell — deliberately not a close scan. */
const DAY2_OVERRUN = new Date('2026-07-16T20:05:00Z');
const DAY2_CLOSE = new Date('2026-07-16T20:15:00Z');

let failures = 0;

function check(label: string, actual: unknown, expected: unknown) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (!ok) failures += 1;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${JSON.stringify(actual)}` + (ok ? '' : ` (want ${JSON.stringify(expected)})`));
}

/** Flat 100.00 series so the only interesting bars are the ones written by hand. */
function series(tail: Array<{ date: string; high: number; low: number; close: number }>): OhlcSeries {
  const bars: OhlcSeries = [];
  const start = Date.UTC(2026, 0, 5);
  for (let i = 0; i < 120; i++) {
    const date = new Date(start + i * 86_400_000).toISOString().slice(0, 10);
    bars.push({ date, open: 100, high: 100.5, low: 99.5, close: 100, volume: 1_000_000 });
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
  opts: {
    periodKey: string;
    asOf: string;
    finishedAt: Date;
    periodClose: boolean;
    rows: Snapshot[];
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
  const app = await NestFactory.createApplicationContext(AppModule, { logger: ['error'] });
  const tracker = app.get(SignalTrackerService);
  const results = app.get(ResultsService);
  const history = app.get(HistoryService);
  const runs = app.get<Model<any>>(getModelToken(SCAN_RUN));
  const signals = app.get<Model<any>>(getModelToken(SIGNAL));
  const tracked = app.get<Model<any>>(getModelToken(TRACKED_SIGNAL));
  const barSeries = app.get<Model<any>>(getModelToken(BAR_SERIES));

  const oldRuns = await runs.find({ trigger: 'smoke' }).select('_id').lean<any[]>();
  await Promise.all([
    tracked.deleteMany({ yahooTicker: { $in: TICKERS } }),
    barSeries.deleteMany({ yahooTicker: { $in: TICKERS } }),
    signals.deleteMany({ runId: { $in: oldRuns.map((r) => r._id) } }),
    runs.deleteMany({ trigger: 'smoke' }),
  ]);

  // A drifts up, B stays flat, C gaps down through its stop on day 2, D only ever shows mid-session.
  const tails: Record<string, Array<{ date: string; high: number; low: number; close: number }>> = {
    'ZZTEST-A': [
      { date: '2026-07-15', high: 101, low: 99, close: 100 },
      { date: '2026-07-16', high: 106, low: 100, close: 105 },
    ],
    'ZZTEST-B': [
      { date: '2026-07-15', high: 101, low: 99, close: 100 },
      { date: '2026-07-16', high: 101, low: 99, close: 100 },
    ],
    'ZZTEST-C': [
      { date: '2026-07-15', high: 101, low: 99, close: 100 },
      { date: '2026-07-16', high: 100, low: 88, close: 89 },
    ],
    'ZZTEST-D': [
      { date: '2026-07-15', high: 101, low: 99, close: 100 },
      { date: '2026-07-16', high: 101, low: 99, close: 100 },
    ],
  };
  for (const ticker of TICKERS) await saveBars(barSeries, ticker, series(tails[ticker]));

  const snap = (ticker: string, entry: number): Snapshot => ({
    ticker,
    entry,
    tp: entry * 1.2,
    sl: entry * 0.9,
    rr: 2,
  });

  // 1. Period close: A, B and C open as confirmed signals.
  const run1 = await fakeScan(runs, signals, {
    periodKey: '2026-07-15',
    asOf: '2026-07-15',
    finishedAt: DAY1_CLOSE,
    periodClose: true,
    rows: [snap('ZZTEST-A', 100), snap('ZZTEST-B', 100), snap('ZZTEST-C', 100)],
  });
  const report1 = await tracker.applyRun(run1);
  check('day 1 close confirms', [report1?.confirmed, report1?.opened], [true, 3]);
  check(
    'day 1 provisional count',
    await tracked.countDocuments({ yahooTicker: { $in: TICKERS }, provisional: true }),
    0,
  );

  // 2. Session pass: prices move and D appears, but nothing opens or closes for good. It also
  //    finishes after the bell, which must not be enough to make it authoritative.
  const run2 = await fakeScan(runs, signals, {
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

  // 3. Next period close: C is stopped out, D never confirmed, A and B roll into VALID.
  const run3 = await fakeScan(runs, signals, {
    periodKey: '2026-07-16',
    asOf: '2026-07-16',
    finishedAt: DAY2_CLOSE,
    periodClose: true,
    rows: [snap('ZZTEST-A', 105), snap('ZZTEST-B', 100)],
  });
  const report3 = await tracker.applyRun(run3);
  check('day 2 close', [report3?.closed, report3?.dropped, report3?.refreshed], [1, 1, 2]);

  const closed = await tracked.findOne({ yahooTicker: 'ZZTEST-C' }).lean<any>();
  check('C stopped out', [closed?.status, closed?.exitReason, closed?.exitPrice], ['closed', 'SL', 90]);
  check('C realized P&L', [closed?.pnlUsd, closed?.pnlR], [-100, -1]);
  check('D dropped', await tracked.countDocuments({ yahooTicker: 'ZZTEST-D' }), 0);

  const buckets = await Promise.all(
    (['new', 'valid', 'closed'] as const).map((bucket) =>
      results.list({ universe: 'Stocks', tf: 'Daily', bucket }),
    ),
  );
  check(
    'buckets new/valid/closed',
    buckets.map((b) => b.rows.filter((r) => TICKERS.includes(r.yahooTicker)).length),
    [0, 2, 1],
  );
  check(
    'valid sorted by P&L',
    (await results.list({ universe: 'Stocks', tf: 'Daily', bucket: 'valid', sort: 'pnl' })).rows
      .filter((r) => TICKERS.includes(r.yahooTicker))
      .map((r) => [r.symbol, r.pnlUsd]),
    [
      ['ZZTEST-A', 50],
      ['ZZTEST-B', 0],
    ],
  );

  const marked = await results.setInterest(buckets[1].rows[0].id, 'interested');
  check('interest saved', marked.interest, 'interested');
  check(
    'interest sorts first',
    (await results.list({ universe: 'Stocks', tf: 'Daily', bucket: 'valid', sort: 'interest' }))
      .rows[0].interest,
    'interested',
  );

  const stats = await history.report({ tf: 'Daily', groupBy: 'Daily' });
  const day = stats.periods.find((p) => p.periodKey === '2026-07-16');
  check('history bucket', [day?.trades, day?.wins, day?.pnlUsd], [1, 0, -100]);
  check('history active count', stats.totals.active >= 2, true);

  const drill = await history.trades({ tf: 'Daily', groupBy: 'Daily', periodKey: '2026-07-16' });
  check('history drill-down', drill.rows.map((r) => r.symbol), ['ZZTEST-C']);

  await Promise.all([
    tracked.deleteMany({ yahooTicker: { $in: TICKERS } }),
    barSeries.deleteMany({ yahooTicker: { $in: TICKERS } }),
    signals.deleteMany({ runId: { $in: [run1, run2, run3].map((id) => new Types.ObjectId(id)) } }),
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
