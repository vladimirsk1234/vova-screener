import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  encodeAssumptionsForFmp,
  fmpYearNum,
  mapCustomDcf,
  sanitizeCustomDcfAssumptions,
} from './fmpCustomDcf.ts';

describe('encodeAssumptionsForFmp', () => {
  it('sends ERP / g as percents and growth as a decimal', () => {
    const sent = encodeAssumptionsForFmp({
      marketRiskPremium: 0.0472,
      longTermGrowthRate: 0.04,
      revenueGrowthPct: 0.0409,
    });
    assert.equal(sent.marketRiskPremium, 4.72);
    assert.equal(sent.longTermGrowthRate, 4);
    assert.equal(sent.revenueGrowthPct, 0.0409);
  });

  it('encodes AAPL conservative overrides the way FMP stores them', () => {
    const stored = sanitizeCustomDcfAssumptions({
      revenueGrowthPct: 0.02045,
      longTermGrowthRate: 0.02,
      marketRiskPremium: 0.0572,
    });
    const sent = encodeAssumptionsForFmp(stored);
    assert.equal(sent.revenueGrowthPct, 0.02045);
    assert.equal(sent.longTermGrowthRate, 2);
    assert.equal(sent.marketRiskPremium, 5.72);
  });

  it('accepts percent or decimal input, then encodes back to FMP units', () => {
    const fromPct = sanitizeCustomDcfAssumptions({
      marketRiskPremium: 4.72,
      longTermGrowthRate: 4,
      riskFreeRate: 3.64,
      revenueGrowthPct: 0.0409,
      ebitdaPct: 0.31,
    });
    assert.equal(fromPct.marketRiskPremium, 0.0472);
    assert.equal(fromPct.longTermGrowthRate, 0.04);
    const sent = encodeAssumptionsForFmp(fromPct);
    assert.equal(sent.marketRiskPremium, 4.72);
    assert.equal(sent.longTermGrowthRate, 4);
    assert.equal(sent.riskFreeRate, 3.64);
    assert.equal(sent.revenueGrowthPct, 0.0409);
    assert.equal(sent.ebitdaPct, 0.31);
  });
});

describe('fmpYearNum', () => {
  it('handles string years from FMP', () => {
    assert.equal(fmpYearNum('2026'), 2026);
    assert.equal(fmpYearNum('2030'), 2030);
    assert.equal(fmpYearNum('2025-09-27'), 2025);
    assert.equal(fmpYearNum(2021), 2021);
  });
});

describe('mapCustomDcf year sort', () => {
  it('sorts string years numerically even when FMP sends newest-first', () => {
    const raw = [
      { year: '2030', ufcf: 5, wacc: 9.37, equityValuePerShare: 148.54, dilutedShares: 10 },
      { year: '2026', ufcf: 1, wacc: 9.37, equityValuePerShare: 148.54, dilutedShares: 10 },
      { year: '2021', ufcf: 0.4, wacc: 9.37, equityValuePerShare: 148.54, dilutedShares: 10 },
    ];
    const payload = mapCustomDcf('AAPL', 'AAPL', raw, {});
    assert.deepEqual(
      payload.years.map((y) => y.year),
      [2021, 2026, 2030],
    );
  });
});
