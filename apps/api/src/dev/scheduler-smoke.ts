/**
 * Pins the background scan schedule: how often each timeframe refreshes, and what each pass asks
 * the scanner for. The cadence is tuned to how much Yahoo will tolerate rather than to anything
 * visible in the app, so nothing else would notice if it drifted.
 *
 * Runs entirely on stubs — no database, no network.
 */
process.env.VOVA_BACKGROUND_SCANS = 'on';

import { CronJob } from 'cron';
import { MARKET_TZ } from '../scans/period';
import { PeriodSchedulerService, SESSION_CRONS } from '../scans/period-scheduler.service';
import { check, finish } from './smoke-harness';

type Started = { tf: string; source: string; forceRefresh: boolean; barsMaxAgeHours: number };

/** The times a cron fires on its next active day, as `Mon 10:05`, in market time. */
function firstDay(expr: string): string[] {
  const job = new CronJob(expr, () => undefined, null, false, MARKET_TZ);
  const fires = job.nextDates(40).map((d) => d.setZone(MARKET_TZ).toFormat('yyyy-MM-dd ccc HH:mm'));
  const date = fires[0].slice(0, 10);
  return fires.filter((f) => f.startsWith(date)).map((f) => f.slice(11));
}

function build(hold?: Promise<void>) {
  const started: Started[] = [];
  const scans = {
    start: async (p: any) => {
      started.push({
        tf: p.tf,
        source: p.source,
        forceRefresh: p.forceRefresh,
        barsMaxAgeHours: p.barsMaxAgeHours,
      });
      if (hold) await hold;
      return { runId: 'stub' };
    },
  };
  const runs = {
    findById: () => ({
      select: () => ({
        lean: () => ({ exec: async () => ({ status: 'completed', counters: {} }) }),
      }),
    }),
  };
  const settings = { get: async () => ({ maxRiskUsd: 1000 }) };
  return {
    service: new PeriodSchedulerService(runs as any, scans as any, settings as any),
    started,
  };
}

/** The cron methods are fire-and-forget, so let the scheduler's queue run before asserting. */
async function drain() {
  for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r));
}

async function main() {
  check('daily refreshes hourly through the session', firstDay(SESSION_CRONS.Daily), [
    'Mon 10:05',
    'Mon 11:05',
    'Mon 12:05',
    'Mon 13:05',
    'Mon 14:05',
    'Mon 15:05',
  ]);
  check('weekly refreshes three times a day', firstDay(SESSION_CRONS.Weekly), [
    'Mon 10:35',
    'Mon 12:35',
    'Mon 14:35',
  ]);
  // The bell is 16:00 ET.
  check('monthly refreshes once, 30 minutes before the close', firstDay(SESSION_CRONS.Monthly), [
    'Mon 15:30',
  ]);

  // Passes that collide are passes that get skipped, and a skipped pass is a lost refresh.
  const day = Object.values(SESSION_CRONS).flatMap(firstDay);
  check('no two session passes start at the same minute', day.length, new Set(day).size);
  check(
    'no session pass starts after the bell',
    day.every((f) => f.slice(4) < '16:00'),
    true,
  );

  const { service, started } = build();

  service.dailySession();
  await drain();
  check(
    'a daily session pass covers both universes on Daily alone',
    started.map((s) => [s.tf, s.source]),
    [
      ['Daily', 'Stocks'],
      ['Daily', 'ETF'],
    ],
  );
  check(
    'a session pass re-downloads without forcing it',
    [started[0].forceRefresh, started[0].barsMaxAgeHours],
    [false, 0.5],
  );

  started.length = 0;
  service.weeklySession();
  await drain();
  check(
    'a weekly session pass touches Weekly alone',
    started.map((s) => s.tf),
    ['Weekly', 'Weekly'],
  );

  started.length = 0;
  service.monthlySession();
  await drain();
  check(
    'a monthly session pass touches Monthly alone',
    started.map((s) => s.tf),
    ['Monthly', 'Monthly'],
  );

  // A close scan is the authoritative one, so it refuses the cache outright.
  started.length = 0;
  service.dailyClose();
  await drain();
  check(
    'the daily close scan forces fresh bars',
    [started.map((s) => s.tf), started[0].forceRefresh, started[0].barsMaxAgeHours],
    [['Daily', 'Daily'], true, 0],
  );

  // A pass that is still running must be skipped, not queued behind the one before it: catching up
  // on a missed hour is worth less than not doubling the load on Yahoo.
  let release: () => void = () => undefined;
  const stalled = build(new Promise<void>((r) => (release = r)));
  stalled.service.dailySession();
  await drain();
  stalled.service.weeklySession();
  await drain();
  check(
    'a session pass is skipped while an earlier one is still running',
    stalled.started.map((s) => s.tf),
    ['Daily'],
  );
  release();
  await drain();

  await finish('SCHEDULER');
}

main().catch((err) => {
  console.error('FAIL', err);
  process.exit(1);
});
