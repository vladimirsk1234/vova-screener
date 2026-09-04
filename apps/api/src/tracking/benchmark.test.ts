import { describe, it, mock } from 'node:test';
import assert from 'node:assert/strict';
import {
  alphaPct,
  benchmarkFromSeries,
  loadBenchmarkSeries,
  periodReturnPct,
  withBenchmark,
  type DailyClose,
} from './benchmark.ts';

function bars(rows: Array<[string, number]>): DailyClose[] {
  return rows.map(([date, close]) => ({ date, close }));
}

describe('periodReturnPct', () => {
  it('is close-to-close percent over the window', () => {
    const series = bars([
      ['2026-01-02', 100],
      ['2026-06-15', 105],
      ['2026-09-04', 110],
    ]);
    assert.equal(periodReturnPct(series, '2026-01-02', '2026-09-04'), 10);
  });

  it('uses the last session on or before range start (YTD uses prior year-end)', () => {
    const series = bars([
      ['2025-12-31', 200],
      ['2026-01-02', 202],
      ['2026-09-04', 220],
    ]);
    assert.equal(periodReturnPct(series, '2026-01-01', '2026-09-04'), 10);
  });

  it('falls back to the first session in-window when nothing precedes from', () => {
    const series = bars([
      ['2026-03-02', 50],
      ['2026-06-01', 55],
    ]);
    assert.equal(periodReturnPct(series, '2026-01-01', '2026-06-01'), 10);
  });

  it('is null when prices or the window are unusable', () => {
    assert.equal(periodReturnPct([], '2026-01-01', '2026-02-01'), null);
    assert.equal(periodReturnPct(bars([['2026-01-02', 0]]), '2026-01-01', '2026-02-01'), null);
    assert.equal(periodReturnPct(bars([['2026-06-01', 10]]), '2026-01-01', '2026-02-01'), null);
    assert.equal(periodReturnPct(bars([['2026-01-02', 10]]), '2026-06-01', '2026-01-01'), null);
  });
});

describe('alphaPct / withBenchmark', () => {
  it('is ROI minus S&P, rounded', () => {
    assert.equal(alphaPct(15, 10), 5);
    assert.equal(alphaPct(8.33, 10.11), -1.78);
    assert.equal(alphaPct(null, 10), null);
    assert.equal(alphaPct(10, null), null);
  });

  it('fills both alphas from a mocked benchmark and leaves them null on failure', () => {
    assert.deepEqual(withBenchmark(20, 40, { symbol: 'SPY', returnPct: 12 }), {
      benchmarkSymbol: 'SPY',
      benchmarkReturnPct: 12,
      alphaVsBenchmarkPct: 8,
      alphaOnAvgPct: 28,
    });
    assert.deepEqual(withBenchmark(20, 40, null), {
      benchmarkSymbol: null,
      benchmarkReturnPct: null,
      alphaVsBenchmarkPct: null,
      alphaOnAvgPct: null,
    });
    assert.deepEqual(withBenchmark(null, null, { symbol: 'SPY', returnPct: 5 }), {
      benchmarkSymbol: 'SPY',
      benchmarkReturnPct: 5,
      alphaVsBenchmarkPct: null,
      alphaOnAvgPct: null,
    });
  });
});

describe('loadBenchmarkSeries / benchmarkFromSeries', () => {
  it('uses mocked Yahoo closes and does not call FMP', async () => {
    const fetchYahoo = mock.fn(async (symbol: string) => {
      assert.equal(symbol, 'SPY');
      return bars([
        ['2025-12-31', 100],
        ['2026-09-04', 108],
      ]);
    });
    const fetchFmp = mock.fn(async () => {
      throw new Error('FMP should not run');
    });
    const series = await loadBenchmarkSeries({
      fetchYahoo,
      fmpConfigured: () => true,
      fetchFmp,
    });
    assert.deepEqual(benchmarkFromSeries(series, '2026-01-01', '2026-09-04'), {
      symbol: 'SPY',
      returnPct: 8,
    });
    assert.equal(fetchYahoo.mock.callCount(), 1);
    assert.equal(fetchFmp.mock.callCount(), 0);
  });

  it('falls back to FMP when Yahoo returns nothing', async () => {
    const series = await loadBenchmarkSeries({
      fetchYahoo: async () => null,
      fmpConfigured: () => true,
      fetchFmp: async () =>
        bars([
          ['2026-01-02', 400],
          ['2026-09-04', 440],
        ]),
    });
    assert.deepEqual(benchmarkFromSeries(series, '2026-01-01', '2026-09-04'), {
      symbol: 'SPY',
      returnPct: 10,
    });
  });

  it('returns null when every source fails, so History stays up', async () => {
    const series = await loadBenchmarkSeries({
      fetchYahoo: async () => null,
      fmpConfigured: () => false,
      fetchFmp: async () => [],
    });
    assert.equal(series, null);
    assert.equal(benchmarkFromSeries(null, '2026-01-01', '2026-09-04'), null);
    assert.equal(benchmarkFromSeries(null, null, '2026-09-04'), null);
  });
});
