/**
 * Checks that one ticker reads the same everywhere, and that one trade is one record.
 *
 * Seeds the two shapes the lists used to disagree on — a scan record carrying the TradingView
 * symbol and a journal copy of the same trade carrying a Yahoo company name — then runs the
 * migration and asserts the short symbol, the list company name, and that the duplicate is gone.
 *
 *   npm run smoke:normalize -w @vova/api
 */
import { NestFactory } from '@nestjs/core';
import { getModelToken } from '@nestjs/mongoose';
import mongoose, { type Model } from 'mongoose';
import { AppModule } from '../app.module';
import { INSTRUMENT, TRACKED_SIGNAL } from '../db/schemas';
import { NormalizeSymbols } from '../migrations/normalize-symbols.service';
import { ResultsService } from '../tracking/results.service';
import { check, finish, useSmokeDatabase } from './smoke-harness';

const TICKERS = ['ZZNORM-A', 'ZZNORM-B', 'ZZNORM-C'];

function row(over: Record<string, unknown>) {
  return {
    universe: 'Stocks',
    tf: 'Monthly',
    status: 'active',
    provisional: false,
    provisionalClose: false,
    imported: false,
    openedPeriodKey: '2026-07',
    openedAsOf: '2026-07-01',
    entry: 101.32,
    tp: 118.01,
    sl: 88.38,
    rrAtEntry: 1.29,
    shares: 8,
    riskUsd: 100,
    ...over,
  };
}

async function main() {
  await useSmokeDatabase('vova-normalize-smoke');
  const app = await NestFactory.createApplicationContext(AppModule, { logger: ['error'] });
  const migration = app.get(NormalizeSymbols);
  const results = app.get(ResultsService);
  const tracked = app.get<Model<any>>(getModelToken(TRACKED_SIGNAL));
  const instruments = app.get<Model<any>>(getModelToken(INSTRUMENT));

  await Promise.all([
    tracked.deleteMany({ yahooTicker: { $in: TICKERS } }),
    instruments.deleteMany({ yahooTicker: { $in: TICKERS } }),
  ]);
  await instruments.insertMany(
    TICKERS.map((yahooTicker) => ({
      yahooTicker,
      tvSymbol: `NASDAQ:${yahooTicker}`,
      companyName: `${yahooTicker} Inc`,
      universes: ['stocks'],
    })),
  );

  await tracked.insertMany([
    // One trade, two records: the break the scan is still settling on the bar in progress, and the
    // copy a History rebuild wrote beside it under the short symbol and a Yahoo name.
    row({
      yahooTicker: 'ZZNORM-A',
      symbol: 'NASDAQ:ZZNORM-A',
      tvSymbol: 'NASDAQ:ZZNORM-A',
      companyName: 'ZZNORM-A Inc',
      provisionalClose: true,
      exitDate: '2026-08-01',
      exitPrice: 79,
      closedPeriodKey: '2026-08',
      pnlUsd: -178.56,
    }),
    row({
      yahooTicker: 'ZZNORM-A',
      symbol: 'ZZNORM-A',
      companyName: 'ZZNORM-A, Inc. - Common Stock',
      status: 'closed',
      backfilled: true,
      exitDate: '2026-08-01',
      exitPrice: 83.54,
      closedPeriodKey: '2026-08',
      pnlUsd: -142.24,
    }),
    // Same trade again, this time disagreeing on the exit: matched on the bar it started on.
    row({
      yahooTicker: 'ZZNORM-B',
      symbol: 'ZZNORM-B',
      companyName: 'ZZNORM-B, Inc.',
      status: 'closed',
      imported: true,
      exitDate: '2026-08-04',
      exitPrice: 78.31,
      closedPeriodKey: '2026-08',
      pnlUsd: -63.8,
    }),
    row({
      yahooTicker: 'ZZNORM-B',
      symbol: 'NYSE:ZZNORM-B',
      tvSymbol: 'NYSE:ZZNORM-B',
      companyName: 'ZZNORM-B Inc',
      provisionalClose: true,
      exitDate: '2026-08-01',
      exitPrice: 78.66,
      closedPeriodKey: '2026-08',
      pnlUsd: -60.3,
    }),
    // A trade of its own on the same ticker as none of the above: nothing to merge it with.
    row({ yahooTicker: 'ZZNORM-C', symbol: 'NASDAQ:ZZNORM-C', tvSymbol: 'NASDAQ:ZZNORM-C' }),
  ]);

  const report = await migration.run();
  check('the pass renames every record and drops one copy per trade', [report.deduped >= 2], [true]);

  const rows = await tracked
    .find({ yahooTicker: { $in: TICKERS } })
    .select('yahooTicker symbol tvSymbol companyName status provisionalClose exitPrice')
    .sort({ yahooTicker: 1 })
    .lean<any[]>();

  check(
    'one record per trade',
    rows.map((r) => r.yahooTicker),
    TICKERS,
  );
  check(
    'every record shows the short symbol',
    rows.map((r) => r.symbol),
    TICKERS,
  );
  check(
    'the TradingView symbol is kept for the link',
    rows.map((r) => r.tvSymbol),
    TICKERS.map((t) => `NASDAQ:${t}`),
  );
  check(
    'company names come from the instrument list',
    rows.map((r) => r.companyName),
    TICKERS.map((t) => `${t} Inc`),
  );
  check(
    'the realized close is the copy that survives',
    [rows[0].status, rows[0].exitPrice, rows[1].status, rows[1].exitPrice],
    ['closed', 83.54, 'active', 78.66],
  );

  const second = await migration.run();
  check(
    'a second pass has nothing left to do',
    [second.normalized, second.deduped],
    [0, 0],
  );

  // The lists read through the same normalizer, so a record written by an older build still prints
  // the one format even before the migration reaches it.
  await tracked.updateOne(
    { yahooTicker: 'ZZNORM-C' },
    { $set: { symbol: 'NASDAQ:ZZNORM-C' } },
  );
  const valid = await results.list({ universe: 'Stocks', tf: 'Monthly', bucket: 'valid' });
  check(
    'the Results row prints the short symbol whatever is stored',
    valid.rows.filter((r) => r.yahooTicker === 'ZZNORM-C').map((r) => [r.symbol, r.tvSymbol]),
    [['ZZNORM-C', 'NASDAQ:ZZNORM-C']],
  );

  await Promise.all([
    tracked.deleteMany({ yahooTicker: { $in: TICKERS } }),
    instruments.deleteMany({ yahooTicker: { $in: TICKERS } }),
  ]);

  await app.close();
  await mongoose.disconnect();
  await finish('NORMALIZE SYMBOLS');
}

main().catch((err) => {
  console.error('NORMALIZE SYMBOLS FAIL', err);
  process.exit(1);
});
