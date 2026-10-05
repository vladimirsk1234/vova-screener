import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import type { Timeframe } from '../../engine/src/types.ts';
import { createDeviceApi } from './api';
import { createCachedStore } from './cachedStore';
import { assertNoFundamentalFields, openNodeStore } from './nodeStore.test-helper';
import { parseUniverse } from './scan';
import type { TrackedSignal } from './types';

function bars(count: number, stepDays: number) {
  return Array.from({ length: count }, (_, i) => {
    const close = 20 + Math.sin(i / 5) * 3 + i * 0.05;
    return {
      date: new Date(Date.UTC(2016, 0, 4 + i * stepDays)).toISOString().slice(0, 10),
      open: close - 0.4,
      high: close + 0.6,
      low: close - 0.8,
      close,
      volume: 1000 + i,
    };
  });
}

async function setup() {
  const dir = mkdtempSync(path.join(tmpdir(), 'vova-api-'));
  const disk = await openNodeStore(path.join(dir, 'vova.db'));
  const store = createCachedStore(disk);
  const fetches: string[] = [];
  const fetchBars = async (ticker: string, tf: Timeframe) => {
    fetches.push(`${ticker}|${tf}`);
    const step = tf === 'Monthly' ? 30 : tf === 'Weekly' ? 7 : 1;
    return { bars: bars(tf === 'Daily' ? 400 : 300, step), companyName: `${ticker} Inc` };
  };
  const fetchCloses = async () => [
    { date: '2016-01-04', close: 100 },
    { date: '2030-01-04', close: 150 },
  ];
  const universe = parseUniverse('NASDAQ:AAA|Alpha Corp\nNYSE:BBB|Beta Corp', 'AMEX:SPYX|ETF One');
  const api = createDeviceApi({ store, universe, fetchBars, fetchCloses });
  const call = async (method: string, url: string, body?: unknown) => {
    const res = await api.handle(method, url, body === undefined ? null : JSON.stringify(body));
    return { status: res.status, data: JSON.parse(res.body) };
  };
  return { store, disk, call, fetches };
}

function closedTrade(id: string, ticker: string, exitDate: string, exitPrice: number): TrackedSignal {
  return {
    id,
    yahooTicker: ticker,
    symbol: ticker,
    tvSymbol: `NASDAQ:${ticker}`,
    companyName: ticker,
    universe: 'Stocks',
    tf: 'Weekly',
    status: 'closed',
    provisional: false,
    provisionalClose: false,
    signalValid: false,
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
    lastSeenPeriodKey: '2024-W05',
    lastSeenAsOf: exitDate,
    lastPrice: exitPrice,
    lastRr: null,
    barsSinceValid: null,
    validSinceAsOf: null,
    isStrong: false,
    unrealizedUsd: null,
    unrealizedR: null,
    unrealizedPct: null,
    closedPeriodKey: '2024-W05',
    closedAt: '2024-02-01T00:00:00.000Z',
    exitDate,
    exitPrice,
    exitReason: 'sell_to_close',
    pnlUsd: (exitPrice - 10) * 100,
    pnlR: exitPrice - 10,
    pnlPct: (exitPrice - 10) * 10,
    holdPeriods: 4,
    interest: null,
    interestRank: 1,
  };
}

describe('device /api', () => {
  it('answers Results, summary and settings in the web shapes', async () => {
    const { call } = await setup();
    const settings = await call('GET', '/settings');
    assert.deepEqual(settings.data, { maxRiskUsd: 200, minRr: 2, fundamentalsFilter: 'all' });

    const page = await call('GET', '/results?universe=Stocks&tf=Weekly&bucket=new&sort=rr&dir=desc&limit=100&offset=0');
    assert.equal(page.status, 200);
    assert.equal(page.data.total, 0);
    assert.deepEqual(Object.keys(page.data.scan).sort(), ['asOf', 'finishedAt', 'newestAsOf', 'periodKey', 'running', 'status']);

    const summary = await call('GET', '/results/summary');
    assert.deepEqual(Object.keys(summary.data), ['Stocks', 'ETF']);
    assert.deepEqual(Object.keys(summary.data.Stocks), ['Weekly', 'Monthly']);
    assert.deepEqual(summary.data.Stocks.Weekly.counts, { new: 0, valid: 0, closed: 0 });
  });

  it('serves History statistics and trades from stored signals, with S&P from Yahoo', async () => {
    const { call, store } = await setup();
    await store.saveSignals([
      closedTrade('w', 'AAA', '2024-02-01', 12),
      closedTrade('l', 'BBB', '2024-03-01', 9),
    ]);
    const report = await call('GET', '/history?universe=Stocks&tf=All&groupBy=Monthly&range=all&sort=period&dir=desc');
    assert.equal(report.status, 200);
    assert.equal(report.data.totals.closed, 2);
    assert.equal(report.data.totals.winRatePct, 50);
    assert.equal(report.data.totals.benchmarkSymbol, 'SPY');
    assert.equal(report.data.timeframes.length, 2);
    assert.ok(report.data.equity.length >= 1);
    assertNoFundamentalFields(report.data);

    const trades = await call('GET', '/history/trades?universe=Stocks&tf=All&groupBy=Monthly&range=all&sort=pnl&dir=desc&limit=200');
    assert.equal(trades.data.total, 2);
    assert.equal(trades.data.rows[0].id, 'w');
    assert.equal(trades.data.rows[0].rr, 2);
    assert.equal(trades.data.rows[0].realized, true);
    assertNoFundamentalFields(trades.data);
  });

  it('marks interest on one signal and keeps it after a reopen', async () => {
    const { call, store, disk } = await setup();
    await store.saveSignals([closedTrade('w', 'AAA', '2024-02-01', 12)]);
    const marked = await call('PATCH', '/results/w/interest', { interest: 'interested' });
    assert.equal(marked.data.interest, 'interested');
    const reread = (await disk.listSignals()).find((row) => row.id === 'w');
    assert.equal(reread?.interestRank, 2);
    const byId = await call('GET', '/results/signal/w');
    assert.equal(byId.data.interest, 'interested');
  });

  it('builds the chart from stored bars without downloading them again', async () => {
    const { call, store, fetches } = await setup();
    await store.putBars({ yahooTicker: 'AAA', tf: 'Weekly', bars: bars(300, 7), fetchedAt: '2020-01-01T00:00:00.000Z', companyName: 'Alpha Corp' });
    const first = await call('GET', '/instruments/AAA/chart?tf=Weekly&riskPerTrade=100&minRr=1.5');
    assert.equal(first.status, 200);
    assert.equal(first.data.companyName, 'Alpha Corp');
    assert.equal(first.data.symbol, 'AAA');
    assert.ok(first.data.watermark.lines.length >= 4);
    // Listed ticker with stored weekly bars: only the missing daily series is fetched, once.
    assert.deepEqual(fetches, ['AAA|Daily']);
    await call('GET', '/instruments/AAA/chart?tf=Weekly&riskPerTrade=100&minRr=1.5');
    assert.deepEqual(fetches, ['AAA|Daily']);
    const snapshot = await call('GET', '/instruments/AAA/chart?tf=Weekly&asOf=2018-01-01');
    assert.equal(snapshot.data.asOf, '2018-01-01');
    assert.ok(snapshot.data.bars.every((b: { date: string }) => b.date <= '2018-01-01'));
  });

  it('runs a manual scan end to end with polled progress', async () => {
    const { call } = await setup();
    const started = await call('POST', '/scans', {
      source: 'MANUAL SCAN',
      manualTickers: 'AAA',
      tf: 'Weekly',
      direction: 'buy',
      minRr: 0,
      noRrReq: true,
      useLastHlSl: true,
      newOnly: false,
      riskPerTrade: 100,
    });
    const runId = started.data.runId as string;
    let run = (await call('GET', `/scans/${runId}`)).data;
    for (let i = 0; i < 50 && !['completed', 'failed', 'cancelled'].includes(run.status); i++) {
      await new Promise((r) => setTimeout(r, 20));
      run = (await call('GET', `/scans/${runId}`)).data;
    }
    assert.equal(run.status, 'completed');
    assert.equal(run.counters.evaluated, 1);
    const progress = await call('GET', `/scans/${runId}/progress`);
    assert.equal(progress.data.phase, 'completed');
    const signals = await call('GET', `/scans/${runId}/signals`);
    const rejections = await call('GET', `/scans/${runId}/rejections?limit=20`);
    assert.equal(signals.data.count + rejections.data.rows.length, 1);
  });

  it('does not serve fundamentals', async () => {
    const { call } = await setup();
    for (const url of [
      '/instruments/AAA/fundamentals?metric=eps',
      '/instruments/AAA/dcf',
      '/instruments/fundamentals-cards?tickers=AAA',
      '/instruments/fundamentals-screener',
    ]) {
      assert.equal((await call('GET', url)).status, 404, url);
    }
    assert.equal((await call('POST', '/history/enrich-eps?limit=40')).status, 404);
  });

  it('re-sizes open positions when Max risk changes', async () => {
    const { call, store } = await setup();
    const open = { ...closedTrade('o', 'AAA', '2024-02-01', 12), status: 'active' as const, exitDate: null, exitPrice: null, closedPeriodKey: null, lastPrice: 11 };
    await store.saveSignals([open]);
    const saved = await call('PUT', '/settings', { maxRiskUsd: 50 });
    assert.equal(saved.data.maxRiskUsd, 50);
    const row = (await store.listSignals())[0];
    assert.equal(row.shares, 50);
    assert.equal(row.riskUsd, 50);
  });
});
