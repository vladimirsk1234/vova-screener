/**
 * The CLOSED tab against the live market, through the whole app rather than the engine alone.
 *
 * `scripts/check_live_close_parity.py` already proves the engine reads the same closes off the
 * same bars as Streamlit. That leaves everything between the engine and the screen — the scan
 * writing the breaks down, the tracker turning them into records, and the period key the Results
 * query filters on — and every one of those can quietly turn a full close list into an empty tab.
 * So this runs a real scan over a slice of the real universe and asks the Results service, the
 * way the app does, whether the tab shows what the scan found.
 *
 *   npm run smoke:close-live -w @vova/api -- --limit 300
 */
import { NestFactory } from '@nestjs/core';
import { getModelToken } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { AppModule } from '../app.module';
import { SIGNAL } from '../db/schemas';
import { ScansService } from '../scans/scans.service';
import { barPeriodKey } from '../scans/period';
import { ResultsService } from '../tracking/results.service';
import { check, finish, useSmokeDatabase } from './smoke-harness';

/** `npm run -w` swallows `--flags`, so the settings are read from the environment too. */
function arg(name: string, env: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0 && process.argv[i + 1]) return process.argv[i + 1];
  return process.env[env] || fallback;
}

async function main() {
  const tf = arg('tf', 'VOVA_SMOKE_TF', 'Weekly') as Timeframe;
  const limit = Number(arg('limit', 'VOVA_SMOKE_LIMIT', '300'));
  // The tracker is the thing under test; a cron firing a second pass mid-run would rewrite its
  // output underneath the assertions.
  process.env.VOVA_BACKGROUND_SCANS = 'off';

  await useSmokeDatabase('vova-close-live');
  const app = await NestFactory.createApplicationContext(AppModule, { logger: ['error', 'warn'] });
  const scans = app.get(ScansService);
  const results = app.get(ResultsService);
  const signals = app.get<Model<any>>(getModelToken(SIGNAL));
  // Start from nothing: yesterday's records would make today's closes look like ordinary exits
  // rather than the adoptions they are, which is the case this smoke exists to cover.
  await scans.resetHistory();

  console.log(`Scanning ${limit} Stocks on ${tf} — this downloads from Yahoo, give it a minute`);
  const { runId } = await scans.start(
    {
      source: 'Stocks',
      tf,
      direction: 'buy',
      minRr: 0,
      noRrReq: true,
      useLastHlSl: true,
      newOnly: false,
      riskPerTrade: 100,
      maxSymbols: limit,
      forceRefresh: true,
      barsMaxAgeHours: 0,
    },
    { trigger: 'scheduled', wait: true },
  );

  const run = await scans.get(runId);
  const counters = run.counters ?? {};
  console.log(
    `run ${run.status}: ${counters.evaluated}/${counters.total} evaluated, ` +
      `${counters.signals} buy signals, ${counters.closes} closes, newestAsOf ${run.newestAsOf}`,
  );

  // What the scan found: the Streamlit SELL TO CLOSE list for these symbols, verbatim.
  const found = await signals
    .find({ runId: new Types.ObjectId(runId), kind: 'sell' })
    .select('payload')
    .lean<any[]>()
    .exec();
  const scanned = new Map(found.map((row) => [row.payload.yahooTicker as string, row.payload]));
  console.log(`closes: ${[...scanned.keys()].sort().join(', ') || '(none)'}`);

  check('the scan writes down every close it counted', scanned.size, counters.closes ?? 0);
  check('the run reports the newest bar it saw', Boolean(run.newestAsOf), true);

  // The period CLOSED filters on has to be the one the closes were filed under. A stray series
  // stamped a day off the grid used to be able to move it on its own, which empties the tab.
  const meta = await results.scanMeta('Stocks', tf);
  const key = meta.newestAsOf ? barPeriodKey(tf, meta.newestAsOf) : null;
  check(
    'CLOSED is keyed to the period the closes were filed under',
    key,
    scanned.size ? barPeriodKey(tf, [...scanned.values()][0].exitAsOf) : key,
  );

  const closed = await results.list({ universe: 'Stocks', tf, bucket: 'closed', limit: 500 });
  const shown = new Set(closed.rows.map((row) => row.yahooTicker));
  const missing = [...scanned.keys()].filter((ticker) => !shown.has(ticker)).sort();
  const extra = [...shown].filter((ticker) => !scanned.has(ticker)).sort();

  // The whole point of the exercise: a symbol Streamlit lists is a symbol the tab lists.
  check('every close the scan found reaches the CLOSED tab', missing, []);
  check('and the tab invents nothing the scan did not find', extra, []);
  check('CLOSED is not empty when the scan found closes', closed.total > 0, scanned.size > 0);

  // Entry and exit come from the replay, so the row carries the trade, not just its ending.
  const priced = closed.rows.filter((row) => scanned.has(row.yahooTicker));
  const mispriced = priced
    .filter((row) => {
      const want = scanned.get(row.yahooTicker);
      return (
        row.exitDate !== want.exitAsOf ||
        Math.abs((row.exitPrice ?? 0) - want.exit) > 0.02 ||
        Math.abs(row.entry - want.entry) > 0.02 ||
        row.openedAsOf !== want.entryAsOf
      );
    })
    .map((row) => row.yahooTicker);
  check('each row carries the replay entry and exit Streamlit prices it from', mispriced, []);
  check('every row is a sell-to-close', [...new Set(priced.map((r) => r.exitReason))].sort(), [
    'sell_to_close',
  ]);

  const sample = closed.rows[0];
  if (sample) {
    console.log(
      `e.g. ${sample.yahooTicker}: in ${sample.openedAsOf} @ ${sample.entry}, ` +
        `out ${sample.exitDate} @ ${sample.exitPrice}, P&L ${sample.pnlUsd}`,
    );
  }

  // Scans run hourly against the same period, so every pass meets the same breaks again. The
  // second one has to recognise the trades it wrote down in the first.
  console.log('Scanning the same symbols again — the second pass must change nothing');
  await scans.start(
    {
      source: 'Stocks',
      tf,
      direction: 'buy',
      minRr: 0,
      noRrReq: true,
      useLastHlSl: true,
      newOnly: false,
      riskPerTrade: 100,
      maxSymbols: limit,
    },
    { trigger: 'scheduled', wait: true },
  );
  const again = await results.list({ universe: 'Stocks', tf, bucket: 'closed', limit: 500 });
  const counts = new Map<string, number>();
  for (const row of again.rows) {
    counts.set(row.yahooTicker, (counts.get(row.yahooTicker) ?? 0) + 1);
  }
  const twice = [...counts.entries()].filter(([, n]) => n > 1).map(([t]) => t);
  check('a second pass over the same period writes no second copy', twice, []);
  check(
    'and the tab still shows the same trades',
    again.rows.map((r) => r.yahooTicker).sort(),
    closed.rows.map((r) => r.yahooTicker).sort(),
  );

  await app.close();
  await finish('CLOSE SCAN LIVE');
}

void main();
