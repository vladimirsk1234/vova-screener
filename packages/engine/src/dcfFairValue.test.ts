import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildDcfChartSeries,
  dcfYearNumber,
  expectedDcfFairValueByYear,
  expectedDcfFairValueToday,
  forecastDcfYears,
  type DcfFairValueInput,
  type DcfYearInput,
} from './dcfFairValue.ts';

function solveTerminalValue(
  input: Omit<DcfFairValueInput, 'terminalValue'> & { target: number },
): number {
  const fwd = forecastDcfYears({ ...input, terminalValue: 0 });
  const wacc = input.wacc!;
  let sumPv = 0;
  for (let i = 0; i < fwd.length; i++) {
    sumPv += fwd[i]!.ufcf! / Math.pow(1 + wacc, i + 1);
  }
  return (input.target * input.dilutedShares! + input.netDebt! - sumPv) * Math.pow(1 + wacc, fwd.length);
}

function histAndFwdYears(histUfcf: number[], fwdUfcf: number[]): DcfYearInput[] {
  const years: DcfYearInput[] = [];
  histUfcf.forEach((ufcf, i) => years.push({ year: 2021 + i, ufcf }));
  fwdUfcf.forEach((ufcf, i) => years.push({ year: 2026 + i, ufcf }));
  return years;
}

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

describe('dcfYearNumber', () => {
  it('parses numeric and string FMP years', () => {
    assert.equal(dcfYearNumber(2026), 2026);
    assert.equal(dcfYearNumber('2026'), 2026);
    assert.equal(dcfYearNumber('2025-09-27'), 2025);
    assert.equal(dcfYearNumber(' 2030 '), 2030);
  });
});

/**
 * Live FMP audit 2026-08-27: 10-row Custom DCF (history 2021–2025 + forecast 2026–2030).
 * Discounting every year understates today; forecast-only years match FMP equityValuePerShare.
 */
describe('forecast-only Custom DCF (AAPL / NOK FMP today)', () => {
  const aaplLastHist = '2025-09-27';
  const nokLastHist = '2025-12-31';
  const aaplYears = histAndFwdYears(
    [92e9, 99e9, 101e9, 108e9, 111e9],
    [118e9, 126e9, 135e9, 144e9, 154e9],
  );
  const aaplBaseCore = {
    years: aaplYears,
    wacc: 0.0937,
    netDebt: -62_000_000_000,
    dilutedShares: 15_116_819_000,
    lastHistDate: aaplLastHist,
  };
  const aaplBaseTv = solveTerminalValue({ ...aaplBaseCore, target: 148.54 });
  const aaplBase = { ...aaplBaseCore, terminalValue: aaplBaseTv };

  const aaplCons = (() => {
    const core = {
      years: histAndFwdYears(
        [92e9, 99e9, 101e9, 108e9, 111e9],
        [112e9, 114e9, 117e9, 119e9, 121e9],
      ),
      wacc: 0.1037,
      netDebt: -62_000_000_000,
      dilutedShares: 15_116_819_000,
      lastHistDate: aaplLastHist,
    };
    return { ...core, terminalValue: solveTerminalValue({ ...core, target: 87.05 }) };
  })();

  const aaplOpt = (() => {
    const core = {
      years: histAndFwdYears(
        [92e9, 99e9, 101e9, 108e9, 111e9],
        [122e9, 132e9, 143e9, 155e9, 168e9],
      ),
      wacc: 0.0887,
      netDebt: -62_000_000_000,
      dilutedShares: 15_116_819_000,
      lastHistDate: aaplLastHist,
    };
    return { ...core, terminalValue: solveTerminalValue({ ...core, target: 166.34 }) };
  })();

  const nokYears = histAndFwdYears(
    [1.1e9, 1.2e9, 1.15e9, 1.3e9, 1.4e9],
    [1.5e9, 1.58e9, 1.66e9, 1.75e9, 1.84e9],
  );
  const nokBaseCore = {
    years: nokYears,
    wacc: 0.09,
    netDebt: 1_200_000_000,
    dilutedShares: 5_600_000_000,
    lastHistDate: nokLastHist,
  };
  const nokBase = {
    ...nokBaseCore,
    terminalValue: solveTerminalValue({ ...nokBaseCore, target: 6.72 }),
  };
  const nokCons = (() => {
    const core = {
      years: histAndFwdYears(
        [1.1e9, 1.2e9, 1.15e9, 1.3e9, 1.4e9],
        [1.42e9, 1.45e9, 1.48e9, 1.51e9, 1.54e9],
      ),
      wacc: 0.1,
      netDebt: 1_200_000_000,
      dilutedShares: 5_600_000_000,
      lastHistDate: nokLastHist,
    };
    return { ...core, terminalValue: solveTerminalValue({ ...core, target: 5.78 }) };
  })();
  const nokOpt = (() => {
    const core = {
      years: histAndFwdYears(
        [1.1e9, 1.2e9, 1.15e9, 1.3e9, 1.4e9],
        [1.58e9, 1.7e9, 1.83e9, 1.97e9, 2.12e9],
      ),
      wacc: 0.085,
      netDebt: 1_200_000_000,
      dilutedShares: 5_600_000_000,
      lastHistDate: nokLastHist,
    };
    return { ...core, terminalValue: solveTerminalValue({ ...core, target: 7.58 }) };
  })();

  it('AAPL-like 10-row fixture: localAll ≠ 148.54, localFwd = 148.54', () => {
    const localAll = expectedDcfFairValueToday({ ...aaplBase, lastHistDate: null });
    const localFwd = expectedDcfFairValueToday(aaplBase);
    assert.ok(localAll != null && localFwd != null);
    assert.notEqual(Math.round(localAll * 100) / 100, 148.54);
    assert.ok(Math.abs(localFwd - 148.54) < 1e-9, `localFwd=${localFwd}`);
    assert.ok(Math.abs(localAll - 148.54) > 1);
  });

  it('NOK-like fixture: localFwd = 6.72', () => {
    const localAll = expectedDcfFairValueToday({ ...nokBase, lastHistDate: null });
    const localFwd = expectedDcfFairValueToday(nokBase);
    assert.ok(localAll != null && localFwd != null);
    assert.ok(Math.abs(localFwd - 6.72) < 1e-9, `localFwd=${localFwd}`);
    assert.ok(Math.abs(localAll - 6.72) > 0.01);
  });

  it('parses string years newest-first the way FMP sends them', () => {
    const newestFirst = [...aaplYears]
      .reverse()
      .map((r) => ({ year: String(r.year) as unknown as number, ufcf: r.ufcf }));
    const fwd = forecastDcfYears({
      ...aaplBase,
      years: newestFirst,
    });
    assert.deepEqual(
      fwd.map((r) => r.year),
      [2026, 2027, 2028, 2029, 2030],
    );
    const today = expectedDcfFairValueToday({ ...aaplBase, years: newestFirst });
    assert.ok(today != null);
    assert.ok(Math.abs(today - 148.54) < 1e-9);
  });

  it('AAPL Cons < Base < Opt on the live FMP today numbers', () => {
    const cons = expectedDcfFairValueToday(aaplCons);
    const base = expectedDcfFairValueToday(aaplBase);
    const opt = expectedDcfFairValueToday(aaplOpt);
    assert.ok(cons != null && base != null && opt != null);
    assert.ok(Math.abs(cons - 87.05) < 1e-9);
    assert.ok(Math.abs(base - 148.54) < 1e-9);
    assert.ok(Math.abs(opt - 166.34) < 1e-9);
    assert.ok(cons < base && base < opt);
  });

  it('NOK Cons < Base < Opt on the live FMP today numbers', () => {
    const cons = expectedDcfFairValueToday(nokCons);
    const base = expectedDcfFairValueToday(nokBase);
    const opt = expectedDcfFairValueToday(nokOpt);
    assert.ok(cons != null && base != null && opt != null);
    assert.ok(Math.abs(cons - 5.78) < 1e-9);
    assert.ok(Math.abs(base - 6.72) < 1e-9);
    assert.ok(Math.abs(opt - 7.58) < 1e-9);
    assert.ok(cons < base && base < opt);
  });

  it('chart today point equals the chip number for each AAPL scenario', () => {
    for (const [label, model] of [
      ['conservative', aaplCons],
      ['base', aaplBase],
      ['optimistic', aaplOpt],
    ] as const) {
      const chip = expectedDcfFairValueToday(model);
      const pts = buildDcfChartSeries({
        ...model,
        asOf: '2026-08-27',
        fmpEquityValuePerShare: chip,
      });
      assert.ok(chip != null, label);
      assert.equal(pts[0]!.date, '2026-08-27');
      assert.ok(Math.abs(pts[0]!.fairValue - chip) < 1e-9, label);
    }
  });

  it('roll-forward uses only forecast years', () => {
    const byYear = expectedDcfFairValueByYear(aaplBase);
    assert.deepEqual(
      byYear.map((r) => r.year),
      [2026, 2027, 2028, 2029, 2030],
    );
  });
});
