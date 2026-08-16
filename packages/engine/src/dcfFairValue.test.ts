import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { expectedDcfFairValueByYear, expectedDcfFairValueToday } from './dcfFairValue.ts';

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
