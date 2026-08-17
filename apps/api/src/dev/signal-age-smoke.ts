/**
 * The NEW / VALID split, end to end and without touching Yahoo.
 *
 * Two synthetic symbols per timeframe: one whose signal appears on the latest bar, one that has been
 * running for four bars by the time the scanner meets it for the first time. Nothing here writes an
 * age by hand — a real scan runs through `ScanRunnerService`, the real tracker opens the records and
 * the real Results reads sort them, so the whole chain is under test. A signal older than the record
 * that carries it belongs in VALID, which is the case that used to land in NEW.
 *
 * Two things the split has broken on before are checked as well: the chart badge must read the same
 * age as the tabs even with the default RR settings, and records written before the age existed must
 * be filled in on boot instead of sitting in VALID until the next scan of their timeframe.
 *
 *   npm run smoke:age -w @vova/api
 */
import { NestFactory } from '@nestjs/core';
import { getModelToken } from '@nestjs/mongoose';
import mongoose, { type Model } from 'mongoose';
import { encodeSeries, runSequenceVovaPine, type OhlcSeries, type Timeframe } from '@vova/engine';
import { AppModule } from '../app.module';
import { BAR_SERIES, SCAN_RUN, SIGNAL, TRACKED_SIGNAL } from '../db/schemas';
import { InstrumentsService } from '../instruments/instruments.service';
import { SignalAgeBackfill } from '../migrations/signal-age-backfill.service';
import { ScansService } from '../scans/scans.service';
import { ResultsService } from '../tracking/results.service';
import { SignalTrackerService } from '../tracking/signal-tracker.service';
import { check, finish, useSmokeDatabase } from './smoke-harness';

const TIMEFRAMES: Timeframe[] = ['Daily', 'Weekly', 'Monthly'];
const DAYS_PER_BAR: Record<Timeframe, number> = { Daily: 1, Weekly: 7, Monthly: 30 };
const INTERVAL: Record<Timeframe, string> = { Daily: '1d', Weekly: '1wk', Monthly: '1mo' };
/** RR is not a gate here, the same as in the background scans. */
const ENGINE_OPTS = {
  atr_len: 14,
  min_rr: 0,
  use_last_hl_sl: true,
  risk_dollars: 100,
  no_rr_req: true,
  direction: 'buy' as const,
};

/**
 * Flat, a climb that leaves an HH peak and an HL trough behind, a pullback deep enough to break the
 * sequence back down, then one bar that closes above the critical level. That last bar is where the
 * buy signal becomes valid, so `trail` bars after it is the age the engine should report.
 */
function ageSeries(trail: number, tf: Timeframe, lastDate: string): OhlcSeries {
  const closes = [
    ...Array.from({ length: 60 }, () => 100),
    ...Array.from({ length: 30 }, (_, i) => 100 + (i + 1) * 2),
    ...Array.from({ length: 6 }, (_, i) => 160 - (i + 1) * 6),
    150,
    ...Array.from({ length: trail }, (_, i) => 150 + (i + 1) * 2),
  ];

  const end = Date.parse(`${lastDate}T00:00:00Z`);
  const stepMs = DAYS_PER_BAR[tf] * 86_400_000;
  return closes.map((close, i) => ({
    date: new Date(end - (closes.length - 1 - i) * stepMs).toISOString().slice(0, 10),
    open: close,
    high: close + 0.5,
    low: close - 0.5,
    close,
    volume: 1_000_000,
  }));
}

/** Flat series with no sequence-up — used to prove Results drops a dead setup off NEW. */
function flatSeries(n: number, tf: Timeframe, lastDate: string): OhlcSeries {
  const end = Date.parse(`${lastDate}T00:00:00Z`);
  const stepMs = DAYS_PER_BAR[tf] * 86_400_000;
  return Array.from({ length: n }, (_, i) => {
    const close = 100;
    return {
      date: new Date(end - (n - 1 - i) * stepMs).toISOString().slice(0, 10),
      open: close,
      high: close + 0.5,
      low: close - 0.5,
      close,
      volume: 1_000_000,
    };
  });
}

async function saveBars(
  barSeries: Model<any>,
  yahooTicker: string,
  tf: Timeframe,
  bars: OhlcSeries,
) {
  const enc = encodeSeries(bars);
  await barSeries.updateOne(
    { yahooTicker, interval: INTERVAL[tf] },
    {
      $set: {
        yahooTicker,
        interval: INTERVAL[tf],
        firstDate: enc.firstDate,
        lastDate: enc.lastDate,
        barCount: enc.barCount,
        dates: Buffer.from(enc.dates),
        open: Buffer.from(enc.open),
        high: Buffer.from(enc.high),
        low: Buffer.from(enc.low),
        close: Buffer.from(enc.close),
        volume: Buffer.from(enc.volume),
        // Fresh, so the scan reads these bars instead of asking Yahoo for them.
        updatedAt: new Date(),
      },
    },
    { upsert: true },
  );
}

async function main() {
  await useSmokeDatabase('vova-smoke-age');
  const app = await NestFactory.createApplicationContext(AppModule, { logger: ['error'] });
  const scans = app.get(ScansService);
  const tracker = app.get(SignalTrackerService);
  const results = app.get(ResultsService);
  const instruments = app.get(InstrumentsService);
  const ageBackfill = app.get(SignalAgeBackfill);
  const runs = app.get<Model<any>>(getModelToken(SCAN_RUN));
  const signals = app.get<Model<any>>(getModelToken(SIGNAL));
  const tracked = app.get<Model<any>>(getModelToken(TRACKED_SIGNAL));
  const barSeries = app.get<Model<any>>(getModelToken(BAR_SERIES));

  for (const tf of TIMEFRAMES) {
    // Uppercase without separators: the manual-ticker parser normalises what it is given.
    const short = tf.slice(0, 1);
    const [freshTicker, oldTicker] = [`ZZAGENEW${short}`, `ZZAGEOLD${short}`];
    const tickers = [freshTicker, oldTicker];

    const lastDate = new Date().toISOString().slice(0, 10);
    const fresh = ageSeries(0, tf, lastDate);
    const old = ageSeries(4, tf, lastDate);
    const validSince = old[old.length - 5].date;

    const freshPine = runSequenceVovaPine(fresh, ENGINE_OPTS);
    const oldPine = runSequenceVovaPine(old, ENGINE_OPTS);
    check(
      `${tf}: the fresh series becomes valid on its last bar`,
      [freshPine?.Valid, freshPine?.bars_since_valid],
      [true, 0],
    );
    check(
      `${tf}: the old series is four bars into its run`,
      [oldPine?.Valid, oldPine?.bars_since_valid],
      [true, 4],
    );

    await tracked.deleteMany({ yahooTicker: { $in: tickers } });
    await saveBars(barSeries, freshTicker, tf, fresh);
    await saveBars(barSeries, oldTicker, tf, old);

    // Manual scan accepts one ticker. Two sequential starts reuse the same run doc, so keep
    // each pass's signals in memory and write them back before the tracker sees the run.
    const savedSignals: any[] = [];
    let runId = '';
    for (const ticker of tickers) {
      const res = await scans.start(
        {
          source: 'MANUAL SCAN',
          manualTickers: ticker,
          tf,
          direction: 'buy',
          minRr: 0,
          noRrReq: true,
          newOnly: false,
        },
        { wait: true },
      );
      runId = res.runId;
      savedSignals.push(...(await signals.find({ runId }).lean<any[]>()));
    }
    await signals.deleteMany({ runId });
    await signals.insertMany(
      savedSignals.map(({ _id, ...rest }) => ({
        ...rest,
        runId: new mongoose.Types.ObjectId(runId),
      })),
    );
    await runs.updateOne({ _id: runId }, { $set: { 'params.source': 'Stocks', periodTf: tf } });

    const payloads = await signals.find({ runId }).select('payload').lean<any[]>();
    check(
      `${tf}: the scan records the age the engine computed`,
      payloads
        .map((s) => [s.payload.yahooTicker, s.payload.barsSinceValid, s.payload.validSinceAsOf])
        .sort(),
      [
        [freshTicker, 0, fresh[fresh.length - 1].date],
        [oldTicker, 4, validSince],
      ].sort(),
    );

    check(`${tf}: both open as tracked signals`, (await tracker.applyRun(runId))?.opened, 2);

    const symbolsIn = (rows: Array<{ yahooTicker: string }>) =>
      rows.filter((r) => tickers.includes(r.yahooTicker)).map((r) => r.yahooTicker);
    const buckets = async () => {
      const [newList, validList] = await Promise.all([
        results.list({ universe: 'Stocks', tf, bucket: 'new' }),
        results.list({ universe: 'Stocks', tf, bucket: 'valid' }),
      ]);
      return { newList, validList, split: [symbolsIn(newList.rows), symbolsIn(validList.rows)] };
    };

    const first = await buckets();
    check(
      `${tf}: the new signal is NEW, the four-bar-old signal is VALID`,
      first.split,
      [[freshTicker], [oldTicker]],
    );
    check(
      `${tf}: the VALID row carries the age of the signal`,
      first.validList.rows
        .filter((r) => r.yahooTicker === oldTicker)
        .map((r) => [r.barsSinceValid, r.validSinceAsOf]),
      [[4, validSince]],
    );

    // Results revalidates age against the bar cache on every NEW/VALID read, so a setup that
    // dies between hourly scans leaves the list immediately — and returns to NEW if it
    // recovers on the same period with age 0.
    const dead = flatSeries(fresh.length, tf, lastDate);
    check(
      `${tf}: the dead series has no buy setup`,
      runSequenceVovaPine(dead, ENGINE_OPTS)?.Valid ?? false,
      false,
    );
    await saveBars(barSeries, freshTicker, tf, dead);
    const afterDeath = await buckets();
    check(
      `${tf}: a dead setup leaves NEW without waiting for a scan`,
      afterDeath.split,
      [[], [oldTicker]],
    );
    const hidden = await tracked
      .findOne({ yahooTicker: freshTicker, tf })
      .select('status signalValid barsSinceValid')
      .lean<any>();
    check(
      `${tf}: the position stays open (hidden) when the setup dies`,
      [hidden?.status, hidden?.signalValid, hidden?.barsSinceValid ?? null],
      ['active', false, null],
    );

    await saveBars(barSeries, freshTicker, tf, fresh);
    check(
      `${tf}: the same-period recovery returns to NEW`,
      (await buckets()).split,
      [[freshTicker], [oldTicker]],
    );

    await saveBars(barSeries, freshTicker, tf, old);
    check(
      `${tf}: age greater than one bar moves the row to VALID`,
      (await buckets()).split,
      [[], [freshTicker, oldTicker].sort()],
    );
    check(
      `${tf}: aging never closes the trade`,
      (await tracked.findOne({ yahooTicker: freshTicker, tf }).select('status').lean<any>())?.status,
      'active',
    );
    // Put the fresh signal back so the chart / backfill checks below still see age 0.
    await saveBars(barSeries, freshTicker, tf, fresh);
    await results.revalidateLiveAges('Stocks', tf);

    // The chart reads the same age as the tabs on the settings the chart screen actually sends —
    // `min_rr: 1.5` with the RR requirement on — because RR decides nothing about NEW vs VALID.
    check(
      `${tf}: the chart agrees with the tabs on the default RR settings`,
      [
        (await instruments.chart(freshTicker, tf)).pine?.barsSinceValid,
        (await instruments.chart(oldTicker, tf)).pine?.barsSinceValid,
      ],
      [0, 4],
    );

    // What a deploy finds in the database: records written before the age was stored. Hide the
    // bar cache so list() cannot revalidate, then the missing field can only land in VALID —
    // which empties NEW for a whole timeframe until its next scan or the boot backfill.
    await tracked.updateMany(
      { yahooTicker: { $in: tickers } },
      { $unset: { barsSinceValid: '', validSinceAsOf: '' } },
    );
    const cachedBars = await barSeries.find({ yahooTicker: { $in: tickers } }).lean<any[]>();
    await barSeries.deleteMany({ yahooTicker: { $in: tickers } });
    check(`${tf}: records with no age all fall into VALID`, (await buckets()).split, [
      [],
      [freshTicker, oldTicker].sort(),
    ]);

    for (const doc of cachedBars) {
      const { _id, ...rest } = doc;
      await barSeries.updateOne({ yahooTicker: rest.yahooTicker, interval: rest.interval }, { $set: rest }, {
        upsert: true,
      });
    }

    const report = await ageBackfill.run();
    check(`${tf}: the backfill fills both records`, [report.filled, report.skipped], [2, 0]);
    check(`${tf}: the split is back after the backfill`, (await buckets()).split, [
      [freshTicker],
      [oldTicker],
    ]);
    check(
      `${tf}: the backfilled age matches the scan`,
      (await buckets()).validList.rows
        .filter((r) => r.yahooTicker === oldTicker)
        .map((r) => [r.barsSinceValid, r.validSinceAsOf]),
      [[4, validSince]],
    );

    await Promise.all([
      tracked.deleteMany({ yahooTicker: { $in: tickers } }),
      barSeries.deleteMany({ yahooTicker: { $in: tickers } }),
      signals.deleteMany({ runId }),
      runs.deleteMany({ _id: runId }),
    ]);
  }

  await app.close();
  await mongoose.disconnect();
  await finish('SIGNAL AGE');
}

main().catch((err) => {
  console.error('SIGNAL AGE FAIL', err);
  process.exit(1);
});
