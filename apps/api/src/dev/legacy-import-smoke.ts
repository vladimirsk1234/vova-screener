/**
 * Checks that the pre-tracking trade journal survives the move into `trackedSignals`.
 *
 * Seeds a `trades` collection in the shape the old app wrote, runs the import and asserts what
 * came across: realized P&L in History, interest marks, universes recovered from the instrument
 * list, discarded rows left behind, and a second pass adding nothing.
 *
 *   npm run smoke:legacy -w @vova/api
 */
import { NestFactory } from '@nestjs/core';
import { getConnectionToken, getModelToken } from '@nestjs/mongoose';
import mongoose, { type Connection, type Model } from 'mongoose';
import { AppModule } from '../app.module';
import { INSTRUMENT, TRACKED_SIGNAL } from '../db/schemas';
import { LegacyTradesMigration } from '../migrations/legacy-trades.service';
import { HistoryService } from '../tracking/history.service';
import { ResultsService } from '../tracking/results.service';
import { SettingsService } from '../settings/settings.module';
import { check, finish, useSmokeDatabase } from './smoke-harness';

const TICKERS = ['ZZOLD-WIN', 'ZZOLD-LOSS', 'ZZOLD-HAND', 'ZZOLD-OPEN', 'ZZOLD-MARK', 'ZZOLD-ETF', 'ZZOLD-WEEK', 'ZZOLD-DEAD', 'ZZOLD-JUNK', 'ZZOLD-DUP'];

/** The journal's own field names, which is the whole point of the exercise. */
type LegacyTrade = Record<string, unknown>;

const OPENED = new Date('2026-06-10T13:35:00Z');

function trade(over: LegacyTrade): LegacyTrade {
  return {
    symbol: over.yahooTicker,
    companyName: `${over.yahooTicker} Inc`,
    tf: 'Daily',
    openedAt: OPENED,
    asOf: '2026-06-10',
    periodKey: '2026-06-10',
    entry: 100,
    tp: 130,
    sl: 90,
    rrAtEntry: 3,
    shares: 10,
    riskUsd: 100,
    status: 'open',
    source: 'manual',
    ...over,
  };
}

async function main() {
  await useSmokeDatabase('vova-legacy-smoke');
  const app = await NestFactory.createApplicationContext(AppModule, { logger: ['error'] });
  const migration = app.get(LegacyTradesMigration);
  const history = app.get(HistoryService);
  const results = app.get(ResultsService);
  const settings = app.get(SettingsService);
  const tracked = app.get<Model<any>>(getModelToken(TRACKED_SIGNAL));
  const instruments = app.get<Model<any>>(getModelToken(INSTRUMENT));
  const trades = app.get<Connection>(getConnectionToken()).db!.collection('trades');

  await Promise.all([
    tracked.deleteMany({ yahooTicker: { $in: TICKERS } }),
    instruments.deleteMany({ yahooTicker: { $in: TICKERS } }),
    trades.deleteMany({ yahooTicker: { $in: TICKERS } }),
  ]);
  await settings.put({ maxRiskUsd: 100 });

  // Only the ETF ticker carries an ETF universe; the rest fall back to Stocks.
  await instruments.insertMany(
    TICKERS.map((yahooTicker) => ({
      yahooTicker,
      tvSymbol: yahooTicker,
      universes: yahooTicker === 'ZZOLD-ETF' ? ['etf'] : ['stocks'],
    })),
  );

  await trades.insertMany([
    trade({
      yahooTicker: 'ZZOLD-WIN',
      status: 'closed',
      exitPrice: 130,
      exitDate: '2026-06-19',
      exitReason: 'TP',
      pnlUsd: 300,
      pnlR: 3,
    }),
    trade({
      yahooTicker: 'ZZOLD-LOSS',
      status: 'closed',
      exitPrice: 90,
      exitDate: '2026-06-12',
      exitReason: 'SL',
      pnlUsd: -100,
      pnlR: -1,
    }),
    // The journal let you close by hand; the tracker has no such exit, so it must survive as one.
    trade({
      yahooTicker: 'ZZOLD-HAND',
      status: 'closed',
      exitPrice: 110,
      exitDate: '2026-06-15',
      exitReason: 'manual',
      pnlUsd: 100,
      pnlR: 1,
    }),
    trade({ yahooTicker: 'ZZOLD-OPEN', status: 'open', shares: 10 }),
    trade({ yahooTicker: 'ZZOLD-MARK', status: 'interested' }),
    trade({ yahooTicker: 'ZZOLD-ETF', status: 'open' }),
    // A weekly row whose period key is already in weekly shape.
    trade({ yahooTicker: 'ZZOLD-WEEK', tf: 'Weekly', periodKey: '2026-W24', status: 'open' }),
    trade({ yahooTicker: 'ZZOLD-DEAD', status: 'dismissed', exitReason: 'signal_invalid' }),
    // No entry price: nothing can be measured from it.
    trade({ yahooTicker: 'ZZOLD-JUNK', status: 'open', entry: null }),
    trade({ yahooTicker: 'ZZOLD-DUP', status: 'open' }),
  ]);

  // ZZOLD-DUP is already tracked by a live scan, so the journal copy has nothing to add.
  await tracked.create({
    yahooTicker: 'ZZOLD-DUP',
    symbol: 'ZZOLD-DUP',
    universe: 'Stocks',
    tf: 'Daily',
    status: 'active',
    provisional: false,
    openedPeriodKey: '2026-07-31',
    openedAsOf: '2026-07-31',
    entry: 200,
    sl: 180,
    shares: 5,
  });

  const first = await migration.run();
  check('import counts', [first?.imported, first?.superseded, first?.skipped], [7, 1, 1]);

  const win = await tracked.findOne({ yahooTicker: 'ZZOLD-WIN' }).lean<any>();
  check(
    'closed trade keeps its realized P&L',
    [win?.status, win?.exitReason, win?.exitPrice, win?.pnlUsd, win?.pnlR],
    ['closed', 'TP', 130, 300, 3],
  );
  check(
    'closed trade lands in the right period',
    [win?.openedPeriodKey, win?.closedPeriodKey, win?.holdPeriods],
    ['2026-06-10', '2026-06-19', 9],
  );
  check('closed trade is not provisional', [win?.provisional, win?.shares], [false, 10]);

  const hand = await tracked.findOne({ yahooTicker: 'ZZOLD-HAND' }).lean<any>();
  check('a hand-closed trade keeps that reason', hand?.exitReason, 'manual');

  // Sizing follows the one global Max Risk, so an open row is re-sized while closed ones are not.
  await settings.put({ maxRiskUsd: 200 });
  const open = await tracked.findOne({ yahooTicker: 'ZZOLD-OPEN' }).lean<any>();
  check('open trade joins the global Max Risk', [open?.status, open?.shares], ['active', 20]);
  check('closed trade keeps the size it closed at', (await tracked.findOne({ yahooTicker: 'ZZOLD-WIN' }).lean<any>())?.shares, 10);
  await settings.put({ maxRiskUsd: 100 });

  const marked = await tracked.findOne({ yahooTicker: 'ZZOLD-MARK' }).lean<any>();
  check('interest mark survives', [marked?.status, marked?.interest, marked?.interestRank], ['active', 'interested', 2]);

  check(
    'universe recovered from the instrument list',
    (await tracked.findOne({ yahooTicker: 'ZZOLD-ETF' }).lean<any>())?.universe,
    'ETF',
  );
  check(
    'weekly period key kept as it was',
    (await tracked.findOne({ yahooTicker: 'ZZOLD-WEEK' }).lean<any>())?.openedPeriodKey,
    '2026-W24',
  );

  check('dismissed rows stay behind', await tracked.countDocuments({ yahooTicker: 'ZZOLD-DEAD' }), 0);
  check('unusable rows stay behind', await tracked.countDocuments({ yahooTicker: 'ZZOLD-JUNK' }), 0);
  check(
    'unusable rows are stamped so they are not retried forever',
    (await trades.findOne({ yahooTicker: 'ZZOLD-JUNK' }))?.migratedAs,
    'unusable',
  );
  check('an already tracked symbol is not duplicated', await tracked.countDocuments({ yahooTicker: 'ZZOLD-DUP' }), 1);

  const second = await migration.run();
  check('a second pass has nothing left to do', second, null);
  check(
    'and nothing was duplicated',
    await tracked.countDocuments({ yahooTicker: { $in: TICKERS } }),
    8, // 7 imported + the one that was already tracked
  );

  const stats = await history.report({ tf: 'Daily', groupBy: 'Daily' });
  const closed = stats.periods.filter((p) => ['2026-06-12', '2026-06-15', '2026-06-19'].includes(p.periodKey));
  check(
    'imported trades show up in History',
    closed.map((p) => [p.periodKey, p.trades, p.wins, p.pnlUsd]),
    [
      ['2026-06-19', 1, 1, 300],
      ['2026-06-15', 1, 1, 100],
      ['2026-06-12', 1, 0, -100],
    ],
  );

  // Results shows what the latest scan reported, and no scan has priced an imported trade yet, so
  // the three open ones are tracked and waiting rather than on screen. They appear the first time
  // a scan finds their setup still valid, and until then only a break can end them.
  const valid = await results.list({ universe: 'Stocks', tf: 'Daily', bucket: 'valid' });
  check(
    'imported open trades are tracked, not yet on screen',
    [
      valid.rows.filter((r) => TICKERS.includes(r.yahooTicker)).length,
      (
        await tracked
          .find({ yahooTicker: { $in: TICKERS }, status: 'active', universe: 'Stocks', tf: 'Daily' })
          .select('symbol')
          .sort({ symbol: 1 })
          .lean<any[]>()
      ).map((d) => d.symbol),
    ],
    [0, ['ZZOLD-DUP', 'ZZOLD-MARK', 'ZZOLD-OPEN']],
  );

  await Promise.all([
    tracked.deleteMany({ yahooTicker: { $in: TICKERS } }),
    instruments.deleteMany({ yahooTicker: { $in: TICKERS } }),
    trades.deleteMany({ yahooTicker: { $in: TICKERS } }),
  ]);

  await app.close();
  await mongoose.disconnect();
  await finish('LEGACY IMPORT');
}

main().catch((err) => {
  console.error('LEGACY IMPORT FAIL', err);
  process.exit(1);
});
