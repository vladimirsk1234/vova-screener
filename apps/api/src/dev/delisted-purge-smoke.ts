/**
 * Smoke: a tracked signal whose ticker has left STOCK-TICKERS.txt shows up in the delisted
 * preview and is removed by the purge, while a ticker still in the list is left alone.
 *
 *   npm run smoke:delisted -w @vova/api
 *
 * Env is set before AppModule loads (imports are dynamic) so the boot catch-up scan stays off.
 */
process.env.VOVA_BACKGROUND_SCANS = 'off';

async function main() {
  const { NestFactory } = await import('@nestjs/core');
  const { getModelToken } = await import('@nestjs/mongoose');
  const { AppModule } = await import('../app.module');
  const { TRACKED_SIGNAL } = await import('../db/schemas');
  const { ScansService } = await import('../scans/scans.service');
  const { UniverseService } = await import('../universe/universe.service');

  const GONE = 'ZZ-DELISTED-SMOKE';

  function check(label: string, got: unknown, expected: unknown) {
    const ok = JSON.stringify(got) === JSON.stringify(expected);
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${JSON.stringify(got)}`);
    if (!ok) {
      console.log(`     expected ${JSON.stringify(expected)}`);
      process.exitCode = 1;
    }
  }

  function closedRow(yahooTicker: string) {
    return {
      yahooTicker,
      symbol: yahooTicker,
      universe: 'Stocks',
      tf: 'Weekly',
      status: 'closed',
      openedPeriodKey: '2022-10-01',
      openedAsOf: '2022-10-01',
      entry: 9.96,
      tp: 9.94,
      sl: 9.87,
      shares: 1111,
      closedPeriodKey: '2025-01-01',
      exitDate: '2025-01-01',
      exitPrice: 5.84,
      exitReason: 'sell_to_close',
      pnlUsd: -4577.32,
      pnlR: -45.78,
    };
  }

  const app = await NestFactory.createApplicationContext(AppModule, { logger: false });
  const tracked = app.get(getModelToken(TRACKED_SIGNAL));
  const scans = app.get(ScansService);
  const universe = app.get(UniverseService);

  const live = await universe.resolveEntries('Stocks');
  const kept = live[0]?.yahoo;
  if (!kept) throw new Error('Stocks universe is empty — nothing to test against');

  await tracked.deleteMany({ yahooTicker: { $in: [GONE, kept] }, tf: 'Weekly' });
  await tracked.insertMany([closedRow(GONE), closedRow(kept)]);

  const preview = await scans.delistedPreview();
  check('preview finds the delisted ticker', preview.sample.includes(GONE), true);
  check('preview leaves the live ticker alone', preview.sample.includes(kept), false);

  const purge = await scans.purgeDelisted();
  check('purge reports at least the one row', purge.deletedSignals >= 1, true);
  check('delisted row is gone', await tracked.countDocuments({ yahooTicker: GONE }).exec(), 0);
  check(
    'live row survives',
    await tracked.countDocuments({ yahooTicker: kept, tf: 'Weekly' }).exec(),
    1,
  );

  const after = await scans.delistedPreview();
  check('nothing left to purge', after.records, 0);

  await tracked.deleteMany({ yahooTicker: { $in: [GONE, kept] }, tf: 'Weekly' });
  await app.close();
  process.exit(process.exitCode ?? 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
