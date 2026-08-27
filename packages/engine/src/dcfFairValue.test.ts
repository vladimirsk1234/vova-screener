import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildDcfChartSeries,
  expectedDcfFairValueByYear,
  expectedDcfFairValueToday,
} from './dcfFairValue.ts';

describe('expected DCF fair value by year', () => {
  const years = [
    { year: 2026, ufcf: 110 },
    { year: 2027, ufcf: 121 },
    { year: 2028, ufcf: 133.1 },
  ];
  const wacc = 0.1;
  const terminalValue = 2000;
  const netDebt = 100;
  const dilutedShares = 10;

  it('matches the discounted model at t=0 and rolls remaining value forward', () => {
    const today = expectedDcfFairValueToday({
      years,
      wacc,
      terminalValue,
      netDebt,
      dilutedShares,
    });
    const ev0 =
      110 / 1.1 +
      121 / 1.1 ** 2 +
      133.1 / 1.1 ** 3 +
      2000 / 1.1 ** 3;
    assert.ok(today != null);
    assert.ok(Math.abs(today - (ev0 - 100) / 10) < 1e-9);

    const byYear = expectedDcfFairValueByYear({
      years,
      wacc,
      terminalValue,
      netDebt,
      dilutedShares,
    });
    assert.equal(byYear.length, 3);

    const ev2026 = 121 / 1.1 + 133.1 / 1.1 ** 2 + 2000 / 1.1 ** 2;
    const ev2027 = 133.1 / 1.1 + 2000 / 1.1;
    const ev2028 = 2000;
    assert.ok(Math.abs(byYear[0]!.fairValuePerShare! - (ev2026 - 100) / 10) < 1e-9);
    assert.ok(Math.abs(byYear[1]!.fairValuePerShare! - (ev2027 - 100) / 10) < 1e-9);
    assert.ok(Math.abs(byYear[2]!.fairValuePerShare! - (ev2028 - 100) / 10) < 1e-9);
    assert.ok(byYear[0]!.fairValuePerShare! > today);
    assert.ok(byYear[1]!.fairValuePerShare! > byYear[0]!.fairValuePerShare!);
    assert.ok(byYear[2]!.fairValuePerShare! > byYear[1]!.fairValuePerShare!);
  });

  it('returns nulls when WACC, terminal, net debt, or shares are missing', () => {
    const rows = expectedDcfFairValueByYear({
      years,
      wacc: null,
      terminalValue,
      netDebt,
      dilutedShares,
    });
    assert.deepEqual(
      rows.map((r) => r.fairValuePerShare),
      [null, null, null],
    );
  });
});

describe('buildDcfChartSeries', () => {
  const years = [
    { year: 2026, ufcf: 110 },
    { year: 2027, ufcf: 121 },
    { year: 2028, ufcf: 133.1 },
  ];
  const base = {
    years,
    wacc: 0.1,
    terminalValue: 2000,
    netDebt: 100,
    dilutedShares: 10,
  };

  it('uses local today and a monotonic path after asOf, ignoring the FMP headline', () => {
    const today = expectedDcfFairValueToday(base);
    const pts = buildDcfChartSeries({
      ...base,
      asOf: '2026-08-19',
      fmpEquityValuePerShare: 408,
    });
    assert.ok(today != null);
    assert.equal(pts[0]!.date, '2026-08-19');
    assert.equal(pts[0]!.year, undefined);
    assert.ok(Math.abs(pts[0]!.fairValue - today) < 1e-9);
    assert.ok(Math.abs(pts[0]!.fairValue - 408) > 1);
    assert.deepEqual(
      pts.slice(1).map((p) => p.date),
      ['2026-12-31', '2027-12-31', '2028-12-31'],
    );
    assert.deepEqual(
      pts.slice(1).map((p) => p.year),
      [2026, 2027, 2028],
    );
    for (let i = 1; i < pts.length; i++) {
      assert.ok(pts[i]!.fairValue > pts[i - 1]!.fairValue);
    }
  });

  it('drops past fiscal years instead of shifting them to asOf+1', () => {
    const pts = buildDcfChartSeries({
      ...base,
      years: [{ year: 2025, ufcf: 100 }, ...years],
      asOf: '2026-08-19',
      fmpEquityValuePerShare: 250,
    });
    assert.equal(pts[0]!.date, '2026-08-19');
    assert.ok(!pts.some((p) => p.date === '2025-12-31'));
    assert.ok(!pts.some((p) => p.date === '2026-08-20'));
    assert.ok(pts.every((p) => p.date >= '2026-08-19'));
  });

  it('plots Sep-FY roll-forward dates from lastHistDate, not 12-31', () => {
    const pts = buildDcfChartSeries({
      ...base,
      asOf: '2026-08-19',
      lastHistDate: '2025-09-27',
    });
    assert.deepEqual(
      pts.slice(1).map((p) => p.date),
      ['2026-09-27', '2027-09-27', '2028-09-27'],
    );
  });

  it('falls back to the FMP headline when the local model cannot run', () => {
    const pts = buildDcfChartSeries({
      ...base,
      wacc: null,
      asOf: '2026-08-19',
      fmpEquityValuePerShare: 250,
    });
    assert.equal(pts.length, 1);
    assert.equal(pts[0]!.date, '2026-08-19');
    assert.equal(pts[0]!.fairValue, 250);
    assert.equal(pts[0]!.year, undefined);
  });
});
