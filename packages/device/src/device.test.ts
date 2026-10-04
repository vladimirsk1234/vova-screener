import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';
import { buildChart } from './chart';
import { assertNoFundamentalFields } from './index';
import { historyReport } from './history';
import { bucketCounts, listResults } from './results';
import { evaluateBars } from './scan';
import { openNodeStore } from './store';
import { applyScanResult, resizeActive, type ScanOutcome } from './tracker';
import type { ScanMeta, TrackedSignal } from './types';
import { parseYahooChart } from './yahoo';

function signal(partial: Partial<TrackedSignal> & Pick<TrackedSignal, 'id' | 'yahooTicker'>): TrackedSignal {
  return {
    symbol: partial.yahooTicker,
    tvSymbol: partial.yahooTicker,
    companyName: partial.yahooTicker,
    universe: 'Stocks',
    tf: 'Weekly',
    status: 'active',
    provisional: false,
    provisionalClose: false,
    signalValid: true,
    imported: false,
    openedPeriodKey: '2024-W01',
    openedAsOf: '2024-01-02',
    openedAt: '2024-01-02T12:00:00.000Z',
    entry: 10,
    tp: 12,
    sl: 9,
    rrAtEntry: 2,
    shares: 100,
    riskUsd: 100,
    lastSeenPeriodKey: '2024-W10',
    lastSeenAsOf: '2024-03-04',
    lastPrice: 11,
    lastRr: 1.5,
    barsSinceValid: 3,
    validSinceAsOf: '2024-02-12',
    isStrong: false,
    unrealizedUsd: 100,
    unrealizedR: 1,
    unrealizedPct: 10,
    closedPeriodKey: null,
    closedAt: null,
    exitDate: null,
    exitPrice: null,
    exitReason: null,
    pnlUsd: null,
    pnlR: null,
    pnlPct: null,
    holdPeriods: null,
    interest: null,
    interestRank: 1,
    ...partial,
  };
}

const scan: ScanMeta = {
  periodKey: '2024-W10',
  asOf: '2024-03-04',
  newestAsOf: '2024-03-04',
  finishedAt: '2024-03-04T21:00:00.000Z',
  running: false,
  status: 'completed',
};

describe('device store', () => {
  it('keeps settings, bars and signals across a reopened sqlite file', async () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'vova-'));
    const file = path.join(dir, 'vova.db');
    const store = await openNodeStore(file);
    await store.putSettings({ maxRiskUsd: 250, minRr: 1.2 });
    await store.putBars({
      yahooTicker: 'AAPL',
      tf: 'Weekly',
      bars: [{ date: '2024-01-02', open: 1, high: 2, low: 0.5, close: 1.5, volume: 10 }],
      fetchedAt: '2024-03-01T00:00:00.000Z',
      companyName: 'Apple',
    });
    const row = signal({ id: 's1', yahooTicker: 'AAPL', companyName: 'Apple' });
    await store.saveSignals([row]);
    const again = await openNodeStore(file);
    assert.deepEqual(await again.getSettings(), { maxRiskUsd: 250, minRr: 1.2 });
    assert.equal((await again.getBars('AAPL', 'Weekly'))?.companyName, 'Apple');
    assert.equal((await again.listSignals())[0].companyName, 'Apple');
    assertNoFundamentalFields(await again.listSignals());
  });
});

describe('results', () => {
  it('splits new and valid on bar age and keeps closed on the scan period', () => {
    const rows = [
      signal({ id: 'new', yahooTicker: 'NEW', barsSinceValid: 0, lastSeenPeriodKey: '2024-W10', lastRr: 2 }),
      signal({ id: 'old', yahooTicker: 'OLD', barsSinceValid: 4, lastRr: 1.2 }),
      signal({
        id: 'closed',
        yahooTicker: 'CLO',
        status: 'closed',
        closedPeriodKey: '2024-W10',
        exitDate: '2024-03-04',
        exitPrice: 12,
        pnlUsd: 200,
        rrAtEntry: 2,
      }),
      signal({ id: 'hidden', yahooTicker: 'HID', signalValid: false }),
    ];
    const counts = bucketCounts(rows, 'Stocks', 'Weekly', 0, scan);
    assert.deepEqual(counts, { new: 1, valid: 1, closed: 1 });
    const filtered = listResults({
      signals: rows,
      universe: 'Stocks',
      tf: 'Weekly',
      bucket: 'valid',
      sort: 'rr',
      dir: 'desc',
      minRr: 2,
      scan,
    });
    assert.equal(filtered.total, 0);
  });
});

describe('tracker', () => {
  it('opens a buy, hides a setup that disappeared, and adopts a close', () => {
    const existing = signal({ id: 'keep', yahooTicker: 'KEEP', barsSinceValid: 2 });
    const gone = signal({ id: 'gone', yahooTicker: 'GONE', signalValid: true });
    const keepBuy = {
      kind: 'buy' as const,
      symbol: 'KEEP',
      tvSymbol: 'KEEP',
      yahooTicker: 'KEEP',
      companyName: 'KEEP',
      tvUrl: 'https://example.test',
      entry: 10,
      tp: 12,
      sl: 9,
      rr: 1.5,
      shares: 100,
      positionValue: 1000,
      isNew: false,
      isStrong: false,
      barsSinceValid: 2,
      validSinceAsOf: '2024-02-12',
      atr: 1,
      asOf: '2024-03-04',
    };
    const outcomes: ScanOutcome[] = [
      {
        yahooTicker: 'KEEP',
        symbol: 'KEEP',
        tvSymbol: 'KEEP',
        companyName: 'KEEP',
        buy: keepBuy,
        sell: null,
        unevaluated: false,
      },
      {
        yahooTicker: 'NEW',
        symbol: 'NEW',
        tvSymbol: 'NASDAQ:NEW',
        companyName: 'New Co',
        buy: {
          kind: 'buy',
          symbol: 'NEW',
          tvSymbol: 'NASDAQ:NEW',
          yahooTicker: 'NEW',
          companyName: 'New Co',
          tvUrl: 'https://example.test',
          entry: 20,
          tp: 26,
          sl: 18,
          rr: 3,
          shares: 50,
          positionValue: 1000,
          isNew: true,
          isStrong: true,
          barsSinceValid: 0,
          validSinceAsOf: '2024-03-04',
          atr: 1,
          asOf: '2024-03-04',
        },
        sell: null,
        unevaluated: false,
      },
    ];
    const applied = applyScanResult({
      signals: [existing, gone],
      universe: 'Stocks',
      tf: 'Weekly',
      periodKey: '2024-W10',
      confirmed: true,
      maxRiskUsd: 100,
      outcomes,
      complete: true,
      barsFor: () => null,
    });
    const symbols = applied.signals.map((s) => s.yahooTicker).sort();
    assert.deepEqual(symbols, ['GONE', 'KEEP', 'NEW']);
    assert.equal(applied.signals.find((s) => s.yahooTicker === 'GONE')?.signalValid, false);
    assert.equal(applied.signals.find((s) => s.yahooTicker === 'NEW')?.barsSinceValid, 0);
    assert.equal(applied.report.opened, 1);
    assert.equal(applied.report.hidden, 1);
    assertNoFundamentalFields(applied.signals);

    const sized = resizeActive(applied.signals, 50);
    const open = sized.find((s) => s.yahooTicker === 'NEW');
    assert.equal(open?.riskUsd, 50);
    assert.equal(open?.shares, Math.round(50 / (20 - 18)));
  });
});

describe('history', () => {
  it('reports the existing statistics from closed trades and no fundamental fields', () => {
    const closedA = signal({
      id: 'a',
      yahooTicker: 'AAA',
      status: 'closed',
      entry: 10,
      sl: 9,
      shares: 100,
      exitDate: '2024-02-01',
      exitPrice: 12,
      exitReason: 'sell_to_close',
      pnlUsd: 200,
      pnlR: 2,
      pnlPct: 20,
      rrAtEntry: 2,
      holdPeriods: 4,
      openedAsOf: '2024-01-02',
    });
    const closedB = signal({
      id: 'b',
      yahooTicker: 'BBB',
      status: 'closed',
      entry: 10,
      sl: 8,
      shares: 50,
      exitDate: '2024-03-01',
      exitPrice: 9,
      exitReason: 'sell_to_close',
      pnlUsd: -50,
      pnlR: -0.5,
      pnlPct: -10,
      rrAtEntry: 1,
      holdPeriods: 8,
      openedAsOf: '2024-01-15',
    });
    const report = historyReport({
      signals: [closedA, closedB],
      universe: 'Stocks',
      tf: 'All',
      groupBy: 'Monthly',
      range: 'all',
      minRr: 0,
      maxRiskUsd: 100,
      benchmark: {
        symbol: 'SPY',
        bars: [
          { date: '2024-01-02', close: 100 },
          { date: '2024-03-01', close: 110 },
        ],
      },
      now: new Date('2024-04-01T00:00:00Z'),
    });
    assert.equal(report.totals.closed, 2);
    assert.equal(report.totals.winRatePct, 50);
    assert.equal(report.totals.pnlUsd, 150);
    assert.equal(report.totals.avgR, 0.75);
    assert.equal(report.totals.profitToRisk, 0.75);
    assert.ok(report.equity.length >= 1);
    assert.equal(report.totals.benchmarkSymbol, 'SPY');
    assert.equal(report.totals.benchmarkReturnPct, 10);
    assert.equal(report.timeframes.length, 2);
    assertNoFundamentalFields(report);
  });
});

describe('no FMP in the phone app', () => {
  it('does not request fundamentals, DCF, EPS tags, or Financial Modeling Prep', () => {
    const root = fileURLToPath(new URL('../../..', import.meta.url));
    const files = [
      ...walk(path.join(root, 'packages/device/src')).filter((file) => !file.endsWith(`${path.sep}index.ts`)),
      ...walk(path.join(root, 'apps/ios')).filter((file) => !file.endsWith('universeData.ts')),
    ];
    const text = files.map((file) => readFileSync(file, 'utf8')).join('\n');
    assert.equal(/financialmodelingprep|FMP_API_KEY|fundamentals-cards|fundamentals-screener|enrich-eps|enrich-premium|premiumPctAtEntry|epsAtEntry/i.test(text), false);
    const bundled = readFileSync(path.join(root, 'apps/ios/src/universeData.ts'), 'utf8');
    const stocks = readFileSync(path.join(root, 'STOCK-TICKERS.txt'), 'utf8');
    const etf = readFileSync(path.join(root, 'TV-LIST-ETF.txt'), 'utf8');
    assert.equal(bundled.includes(JSON.stringify(stocks)), true);
    assert.equal(bundled.includes(JSON.stringify(etf)), true);
  });
});

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === '.expo') continue;
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) out.push(...walk(full));
    else if (/\.(ts|tsx|js|json)$/.test(name) && !name.endsWith('.test.ts')) out.push(full);
  }
  return out;
}

describe('yahoo and chart', () => {
  it('parses a Yahoo chart payload into OHLC', () => {
    const parsed = parseYahooChart(
      {
        chart: {
          result: [
            {
              timestamp: [1_704_153_600, 1_704_758_400],
              indicators: {
                quote: [{ open: [10, 11], high: [12, 13], low: [9, 10], close: [11, 12], volume: [100, 110] }],
              },
              meta: { longName: 'Example' },
            },
          ],
        },
      },
      'Weekly',
    );
    assert.equal(parsed.companyName, 'Example');
    assert.ok(parsed.bars && parsed.bars.length >= 1);
  });

  it('builds a technical chart without P/E, market cap or a description', () => {
    const bars = Array.from({ length: 80 }, (_, i) => {
      const close = 20 + Math.sin(i / 5) * 2 + i * 0.05;
      return {
        date: new Date(Date.UTC(2022, 0, 3 + i * 7)).toISOString().slice(0, 10),
        open: close - 0.4,
        high: close + 0.6,
        low: close - 0.8,
        close,
        volume: 1000 + i,
      };
    });
    const chart = buildChart({
      yahooTicker: 'AAA',
      symbol: 'AAA',
      companyName: 'Acme',
      tf: 'Weekly',
      bars,
      dailyClose: 24,
      prevDailyClose: 23,
    });
    const text = chart.watermarkLines.join('\n');
    assert.equal(text.includes('PE:'), false);
    assert.equal(text.includes('Earn:'), false);
    assert.equal(text.includes('+'), true);
    assert.ok(chart.bars.length > 0);
    assert.ok(chart.overlay);
    assertNoFundamentalFields(chart);
  });

  it('marks a short series as unevaluated rather than a fundamental reject', () => {
    const outcome = evaluateBars({
      bars: null,
      yahooTicker: 'ZZZ',
      tvSymbol: 'ZZZ',
      companyName: 'Zed',
      tf: 'Weekly',
      riskPerTrade: 100,
    });
    assert.equal(outcome.unevaluated, true);
    assert.equal(outcome.reason, 'NO_DATA');
    assert.equal(outcome.buy, null);
  });
});
