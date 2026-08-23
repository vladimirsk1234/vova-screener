import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  annualizedRorPct,
  buildForecastScenarios,
  dividendCoverage,
  forecastHorizonYears,
  futurePriceAt,
  marginOfSafetyPct,
  scoreAnalystBeats,
} from './forecastReturns.ts';

describe('forecast returns', () => {
  it('matches the Adobe video case at ~$230 / 2.25y / 12.8×', () => {
    const future = futurePriceAt(31.15, 12.8);
    assert.ok(future != null);
    const ror = annualizedRorPct(230, future, 2.25, 0);
    assert.ok(ror != null);
    assert.ok(ror > 25 && ror < 28, `ror=${ror}`);
    const ror21 = annualizedRorPct(230, futurePriceAt(31.15, 21), 2.25, 0);
    assert.ok(ror21 != null && ror21 > ror && ror21 > 50, `ror21=${ror21}`);
  });

  it('at today $275 is ~18%, not the old additive 5y ~13%', () => {
    const future = futurePriceAt(31.15, 12.8);
    const ror = annualizedRorPct(275.3, future, 2.25, 0);
    assert.ok(ror != null && ror > 16 && ror < 20, `ror=${ror}`);
    const oldAdditive = 12.95 + (Math.pow(262 / 275.3, 1 / 5) - 1) * 100;
    assert.ok(Math.abs(ror - oldAdditive) > 3);
  });

  it('uses the estimate FY-end as the horizon (Adobe Nov 2028 ≈ 2.25y from Aug 2026)', () => {
    const years = forecastHorizonYears('2026-08-23', '2028-11-30');
    assert.ok(years != null);
    assert.ok(years > 2.2 && years < 2.3, `years=${years}`);
  });

  it('builds peg + normal + custom scenarios', () => {
    const got = buildForecastScenarios({
      price: 230,
      fairValue: 292,
      fairValueRatio: 12.8,
      normalMultiple: 21,
      customMultiple: 15,
      dividendYieldPct: 0,
      estimates: [
        { year: 2026, date: '2026-11-30', eps: 24.41 },
        { year: 2028, date: '2028-11-30', eps: 31.15 },
      ],
      asOfIso: '2026-08-23',
    });
    assert.ok(got.horizonYears != null && got.horizonYears > 2.2);
    assert.equal(got.horizonEps, 31.15);
    assert.ok(got.rorPegPct != null && got.rorPegPct > 25 && got.rorPegPct < 28);
    assert.ok(got.rorNormalPct != null && got.rorNormalPct > 46);
    assert.ok(got.marginOfSafetyPct != null && got.marginOfSafetyPct > 20);
  });

  it('margin of safety is the inverse of premium', () => {
    assert.ok(Math.abs((marginOfSafetyPct(275, 262) ?? 0) + 4.96) < 0.1);
    assert.ok((marginOfSafetyPct(230, 292) ?? 0) > 20);
  });
});

describe('analyst scorecard', () => {
  it('counts beat / meet / miss in the lookback window', () => {
    const rows = [
      { date: '2025-09-01', epsActual: 1.1, epsEstimated: 1.0 },
      { date: '2025-12-01', epsActual: 1.0, epsEstimated: 1.0 },
      { date: '2026-03-01', epsActual: 0.9, epsEstimated: 1.0 },
      { date: '2024-09-01', epsActual: 2, epsEstimated: 1 },
    ];
    const y1 = scoreAnalystBeats(rows, 1, '2026-08-23');
    assert.equal(y1.total, 3);
    assert.equal(y1.beat, 1);
    assert.equal(y1.meet, 1);
    assert.equal(y1.miss, 1);
    const y2 = scoreAnalystBeats(rows, 2, '2026-08-23');
    assert.equal(y2.total, 4);
    assert.equal(y2.beat, 2);
  });
});

describe('dividend coverage', () => {
  it('marks covered when OCF is 2× the dividend', () => {
    const got = dividendCoverage({
      dividend: 1,
      dilutedShares: 100,
      operatingCashFlow: 200,
      freeCashFlow: 180,
    });
    assert.equal(got.ocfCover, 2);
    assert.equal(got.status, 'covered');
  });

  it('marks none when there is no dividend', () => {
    assert.equal(dividendCoverage({ dividend: 0, dilutedShares: 10, operatingCashFlow: 5 }).status, 'none');
  });
});
