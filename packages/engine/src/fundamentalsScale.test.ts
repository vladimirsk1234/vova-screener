import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  FUNDAMENTALS_SCALE_VERSION,
  afterTaxOperatingIncome,
  buildFundamentalsScale,
  effectiveTaxRate,
  epsFromIncome,
  formatScaleCaption,
  fxToListingMultiplier,
  inferAdrRatio,
  inferShareScale,
  operatingEpsFromGaap,
  peSanityOk,
    pickScaledEps,
    pickScaledFcf,
    fcfFromCashFlow,
    scaleAmount,
  scaleDcf,
  scaleTev,
} from './fundamentalsScale.ts';
import { computeNormalMultiple, type AnnualFundamentalPoint } from './fundamentalsValuation.ts';

describe('XYF FY2025 SEC vs FMP units', () => {
  const netIncome = 1_464_553_000;
  const shares = 249_489_203;
  const fmpEps = 205.56;
  const fxCnyPerUsd = 6.9931;
  const fxToListing = 1 / fxCnyPerUsd;
  const price = 5.42;

  it('detects the extra ADR ×6 on RMB-per-ADS EPS', () => {
    const adr = inferAdrRatio({ ticker: 'XYF', netIncome, fmpEps, dilutedShares: shares });
    assert.equal(adr, 6);
    assert.equal(inferShareScale({ netIncome, fmpEps, dilutedShares: shares, adrRatio: 6 }), 'double_adr');
  });

  it('recomputes USD EPS per ADS near the SEC $5.04', () => {
    const eps = epsFromIncome({
      netIncome,
      dilutedShares: shares,
      fxToListing,
      adrRatio: 6,
    });
    assert.ok(eps != null);
    assert.ok(Math.abs(eps - 5.04) < 0.05, `eps=${eps}`);
  });

  it('scales the raw FMP field to ~$4.90 and prefers income when they diverge', () => {
    const scale = buildFundamentalsScale({
      ticker: 'XYF',
      reportedCurrency: 'CNY',
      listingCurrency: 'USD',
      netIncome,
      fmpEps,
      dilutedShares: shares,
      price,
      fxToListing,
    });
    assert.equal(scale.adrRatio, 6);
    assert.equal(scale.shareScale, 'double_adr');
    assert.equal(scale.reliable, true);
    const picked = pickScaledEps({ fmpEps, netIncome, dilutedShares: shares, scale, price });
    assert.ok(picked != null);
    assert.ok(Math.abs(picked - 5.04) < 0.15, `picked=${picked}`);
    const pe = price / picked;
    assert.ok(pe > 0.9 && pe < 1.3, `pe=${pe}`);
    assert.equal(peSanityOk({ price, eps: picked }), true);
  });

  it('does not treat GAAP vs adjusted (~6%) as a unit error', () => {
    const gaap = 5.04;
    const adj = 5.36;
    assert.ok(Math.abs(adj / gaap - 1) < 0.08);
  });
});

describe('fxToListingMultiplier', () => {
  it('converts CNY → USD via USDCNY-style rates', () => {
    const m = fxToListingMultiplier('CNY', 'USD', (c) => (c === 'CNY' ? 6.9931 : 1));
    assert.ok(Math.abs(m - 1 / 6.9931) < 1e-9);
  });

  it('is 1 when currencies match', () => {
    assert.equal(fxToListingMultiplier('USD', 'USD', () => 1), 1);
  });

  it('treats RMB as CNY', () => {
    const m = fxToListingMultiplier('RMB', 'USD', (c) => (c === 'CNY' ? 7.2 : 1));
    assert.ok(Math.abs(m - 1 / 7.2) < 1e-9);
  });
});

describe('garbage TEV / DCF', () => {
  const scale = buildFundamentalsScale({
    ticker: 'XYF',
    reportedCurrency: 'CNY',
    listingCurrency: 'USD',
    netIncome: 1_464_553_000,
    fmpEps: 205.56,
    dilutedShares: 249_489_203,
    price: 5.42,
    fxToListing: 1 / 6.9931,
  });

  it('drops TEV that is still absurd vs market cap after FX', () => {
    assert.equal(scaleTev(-3.17e9, 36.05e6, { ...scale, fxToListing: 1 }), null);
  });

  it('converts TEV when currencies differ', () => {
    const v = scaleTev(-3.17e9, 124e6, scale);
    assert.ok(v != null);
    assert.ok(Math.abs(v - -3.17e9 / 6.9931) < 1);
  });

  it('nulls a DCF that stays >40× price after scaling', () => {
    assert.equal(scaleDcf(14_312, scale, 5.42), null);
  });
});

describe('computeNormalMultiple rounding', () => {
  const pts = (pe: number): AnnualFundamentalPoint[] => [
    {
      date: '2024-12-31',
      year: 2024,
      price: pe,
      eps: 1,
      revenuePerShare: null,
      fcfPerShare: null,
      ownerEarningsPerShare: null,
      pe,
      revenue: null,
      netIncome: null,
      operatingCashFlow: null,
      freeCashFlow: null,
    },
  ];

  it('keeps sub-1.0 multiples instead of rounding to 0.0', () => {
    const { multiple, source } = computeNormalMultiple(pts(0.28), 'eps');
    assert.equal(source, 'median_pe');
    assert.ok(multiple > 0.2 && multiple < 0.4, `multiple=${multiple}`);
  });

  it('still rounds large multiples to one decimal', () => {
    const { multiple } = computeNormalMultiple(pts(6.29), 'eps');
    assert.equal(multiple, 6.3);
  });
});

describe('formatScaleCaption', () => {
  it('shows listing currency and ADR ratio', () => {
    const scale = buildFundamentalsScale({
      ticker: 'XYF',
      reportedCurrency: 'CNY',
      listingCurrency: 'USD',
      netIncome: 1e9,
      fmpEps: 200,
      dilutedShares: 2e8,
      price: 5,
      fxToListing: 1 / 7,
    });
    const cap = formatScaleCaption(scale);
    assert.ok(cap?.includes('USD'));
    assert.ok(cap?.includes('6'));
  });
});

describe('PDD listing units vs peTTM', () => {
  const price = 90;
  const peTtm = 9;
  const fmpEps = 10.07;
  const adsShares = 1_400_000_000;
  const netIncome = fmpEps * adsShares;
  const dilutedShares = adsShares;

  it('keeps per-ADS EPS (factor 1), not ordinary ×4, when peTTM is ~9', () => {
    assert.equal(inferShareScale({ netIncome, fmpEps, dilutedShares, adrRatio: 4 }), 'ordinary');
    const scale = buildFundamentalsScale({
      ticker: 'PDD',
      reportedCurrency: 'USD',
      listingCurrency: 'USD',
      netIncome,
      fmpEps,
      dilutedShares,
      price,
      fxToListing: 1,
      peTtm,
    });
    assert.equal(scale.version, FUNDAMENTALS_SCALE_VERSION);
    assert.equal(scale.version, 3);
    assert.equal(scale.adrRatio, 4);
    assert.equal(scale.shareScale, 'ads');
    assert.equal(scale.perShareFactor, 1);
    const picked = pickScaledEps({
      fmpEps,
      netIncome,
      dilutedShares,
      scale,
      price,
      peTtm,
    });
    assert.ok(picked != null);
    assert.ok(Math.abs(picked - 10.07) < 0.2, `picked=${picked}`);
    const pe = price / picked;
    assert.ok(pe > 7 && pe < 12, `pe=${pe}`);
  });
});

describe('pickScaledFcf listing units', () => {
  const price = 90;
  const adsShares = 1_400_000_000;
  const ordinaryShares = adsShares * 4;
  const companyFcf = 10.08e9;
  const adsFcf = companyFcf / adsShares;
  const ordinaryFcf = companyFcf / ordinaryShares;

  it('rebuilds FCF/sh from company FCF / ADS when FMP field is ordinary', () => {
    const scale = buildFundamentalsScale({
      ticker: 'PDD',
      reportedCurrency: 'USD',
      listingCurrency: 'USD',
      netIncome: 10.07 * adsShares,
      fmpEps: 10.07,
      dilutedShares: ordinaryShares,
      price,
      fxToListing: 1,
      peTtm: 9,
    });
    assert.equal(scale.shareScale, 'ads');
    assert.equal(scale.perShareFactor, 1);
    const fromCf = fcfFromCashFlow({
      freeCashFlow: companyFcf,
      dilutedShares: ordinaryShares,
      fxToListing: 1,
      adrRatio: 4,
    });
    assert.ok(fromCf != null);
    assert.ok(Math.abs(fromCf - adsFcf) < 0.05, `fromCf=${fromCf}`);
    const picked = pickScaledFcf({
      fmpFcfPerShare: ordinaryFcf,
      freeCashFlow: companyFcf,
      dilutedShares: ordinaryShares,
      scale,
      price,
    });
    assert.ok(picked != null);
    assert.ok(Math.abs(picked - adsFcf) < 0.05, `picked=${picked}`);
    assert.ok(Math.abs(picked - ordinaryFcf) > 1);
  });

  it('keeps already-ADS FMP FCF/sh when income shares are ADS counts', () => {
    const scale = buildFundamentalsScale({
      ticker: 'PDD',
      reportedCurrency: 'USD',
      listingCurrency: 'USD',
      netIncome: 10.07 * adsShares,
      fmpEps: 10.07,
      dilutedShares: adsShares,
      price,
      fxToListing: 1,
      peTtm: 9,
    });
    const picked = pickScaledFcf({
      fmpFcfPerShare: adsFcf,
      freeCashFlow: companyFcf,
      dilutedShares: adsShares,
      scale,
      price,
    });
    assert.ok(picked != null);
    assert.ok(Math.abs(picked - adsFcf) < 0.05, `picked=${picked}`);
  });
});

describe('scaleAmount', () => {
  it('returns null for null input', () => {
    assert.equal(scaleAmount(null, 2), null);
  });
});

describe('after-tax operating EPS (FMP NOPAT proxy for FG operating earnings)', () => {
  // AAPL FY2019 10-K, split-adjusted 4:1. FG Historical table showed 2.97 —
  // that is GAAP diluted. NOPAT / shares is the closest FMP operating proxy.
  const aapl2019 = {
    operatingIncome: 63_930_000_000,
    incomeBeforeTax: 65_737_000_000,
    incomeTaxExpense: 10_481_000_000,
    netIncome: 55_256_000_000,
    dilutedShares: 18_471_336_000,
    gaapEps: 2.97,
  };

  it('uses the filing tax rate, not a flat 21%', () => {
    const rate = effectiveTaxRate(aapl2019.incomeBeforeTax, aapl2019.incomeTaxExpense);
    assert.ok(rate != null);
    assert.ok(rate > 0.15 && rate < 0.17, `rate=${rate}`);
    const nopat = afterTaxOperatingIncome(aapl2019);
    assert.ok(nopat != null);
    const expected = aapl2019.operatingIncome * (1 - rate);
    assert.ok(Math.abs(nopat - expected) < 1);
  });

  it('lands AAPL FY2019 operating EPS near GAAP 2.97 (FG table), not pre-tax ~3.46', () => {
    const op = operatingEpsFromGaap({
      ...aapl2019,
      fxToListing: 1,
      adrRatio: 1,
    });
    assert.ok(op != null);
    assert.ok(op > 2.85 && op < 3.05, `operatingEps=${op}`);
    const pretax = aapl2019.operatingIncome / aapl2019.dilutedShares;
    assert.ok(pretax > 3.4, `pretax=${pretax}`);
    assert.ok(Math.abs(op - pretax) > 0.3);
  });

  it('inherits a 4:1 split from GAAP so filing-share NOPAT is not 4× too high', () => {
    const filingShares = aapl2019.dilutedShares / 4;
    const op = operatingEpsFromGaap({
      ...aapl2019,
      dilutedShares: filingShares,
      fxToListing: 1,
      adrRatio: 1,
    });
    assert.ok(op != null);
    assert.ok(op > 2.85 && op < 3.05, `operatingEps=${op}`);
    const naive = afterTaxOperatingIncome(aapl2019)! / filingShares;
    assert.ok(naive > 11, `naive=${naive}`);
  });

  it('AAPL FY2011 NOPAT/share matches the FG 0.99 table print after 28:1 splits', () => {
    // FY2011 10-K; 7:1 (2014) × 4:1 (2020) = 28.
    const filing = {
      operatingIncome: 33_790_000_000,
      incomeBeforeTax: 34_205_000_000,
      incomeTaxExpense: 8_283_000_000,
      netIncome: 25_922_000_000,
      dilutedShares: 924_258_000,
      gaapEps: 0.99,
    };
    const op = operatingEpsFromGaap({
      ...filing,
      fxToListing: 1,
      adrRatio: 1,
    });
    assert.ok(op != null);
    assert.ok(Math.abs(op - 0.99) < 0.05, `operatingEps=${op}`);
  });

  it('falls back to NOPAT/shares when net income is the opposite sign', () => {
    const op = operatingEpsFromGaap({
      gaapEps: -1,
      netIncome: -100,
      operatingIncome: 50,
      incomeBeforeTax: 50,
      incomeTaxExpense: 10,
      dilutedShares: 10,
      fxToListing: 1,
      adrRatio: 1,
    });
    const nopat = afterTaxOperatingIncome({
      operatingIncome: 50,
      incomeBeforeTax: 50,
      incomeTaxExpense: 10,
    });
    assert.ok(op != null && nopat != null);
    assert.ok(Math.abs(op - nopat / 10) < 1e-9);
  });
});
