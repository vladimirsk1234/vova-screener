import { describe, it, mock } from 'node:test';
import assert from 'node:assert/strict';
import { alphaPct, periodReturnPct, withBenchmark, type DailyClose } from './benchmark.ts';
import { BenchmarkService } from './benchmark.service.ts';

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

describe('BenchmarkService', () => {
  it('computes the period return from mocked Yahoo closes and does not call FMP', async () => {
    const fetchDailyCloses = mock.fn(async () =>
      bars([
        ['2025-12-31', 100],
        ['2026-09-04', 108],
      ]),
    );
    const historicalCloses = mock.fn(async () => {
      throw new Error('FMP should not run');
    });
    const svc = new BenchmarkService(
      { fetchDailyCloses } as any,
      { configured: () => true, historicalCloses } as any,
    );
    const hit = await svc.periodReturn('2026-01-01', '2026-09-04');
    assert.deepEqual(hit, { symbol: 'SPY', returnPct: 8 });
    assert.equal(fetchDailyCloses.mock.callCount(), 1);
    assert.equal(historicalCloses.mock.callCount(), 0);
  });

  it('falls back to FMP when Yahoo returns nothing', async () => {
    const svc = new BenchmarkService(
      { fetchDailyCloses: async () => null } as any,
      {
        configured: () => true,
        historicalCloses: async () =>
          bars([
            ['2026-01-02', 400],
            ['2026-09-04', 440],
          ]),
      } as any,
    );
    const hit = await svc.periodReturn('2026-01-01', '2026-09-04');
    assert.deepEqual(hit, { symbol: 'SPY', returnPct: 10 });
  });

  it('returns null when every source fails, so History stays up', async () => {
    const svc = new BenchmarkService(
      { fetchDailyCloses: async () => null } as any,
      { configured: () => false, historicalCloses: async () => [] } as any,
    );
    assert.equal(await svc.periodReturn('2026-01-01', '2026-09-04'), null);
    assert.equal(await svc.periodReturn(null, '2026-09-04'), null);
  });
});
