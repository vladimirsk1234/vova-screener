import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildValuationSeries,
  fairValueRatioFromGrowth,
  sliceToWindow,
  trailingMetricCagr,
  type AnnualFundamentalPoint,
} from './fundamentalsValuation.ts';

function fy(
  year: number,
  eps: number,
  price: number,
): AnnualFundamentalPoint {
  return {
    date: `${year}-12-31`,
    year,
    price,
    eps,
    revenuePerShare: null,
    fcfPerShare: null,
    ownerEarningsPerShare: null,
    pe: price / eps,
    revenue: null,
    netIncome: null,
    operatingCashFlow: null,
    freeCashFlow: null,
  };
}

/** 2015–2025: slow early decade, then a steep last 5 years. */
function mixedGrowthHistory(): AnnualFundamentalPoint[] {
  return [
    fy(2015, 1.0, 15),
    fy(2016, 1.05, 16),
    fy(2017, 1.1, 17),
    fy(2018, 1.15, 18),
    fy(2019, 1.2, 19),
    fy(2020, 1.25, 20),
    fy(2021, 1.8, 30),
    fy(2022, 2.6, 45),
    fy(2023, 3.7, 60),
    fy(2024, 5.3, 80),
    fy(2025, 7.6, 110),
  ];
}

describe('valuation windows', () => {
  it('keeps two fiscal years for a 1Y window', () => {
    const sliced = sliceToWindow(mixedGrowthHistory(), 1);
    assert.deepEqual(
      sliced.map((p) => p.year),
      [2024, 2025],
    );
  });

  it('uses YoY as the 1Y trailing growth rate', () => {
    const sliced = sliceToWindow(mixedGrowthHistory(), 1);
    const yoy = trailingMetricCagr(sliced, 'eps', 1);
    assert.ok(yoy != null);
    assert.ok(Math.abs(yoy - ((7.6 / 5.3 - 1) * 100)) < 1e-6);
  });

  it('computes a different 10Y CAGR than the last 5 years', () => {
    const hist = mixedGrowthHistory();
    const cagr5 = trailingMetricCagr(sliceToWindow(hist, 5), 'eps', 5);
    const cagr10 = trailingMetricCagr(sliceToWindow(hist, 10), 'eps', 10);
    assert.ok(cagr5 != null && cagr10 != null);
    assert.ok(cagr5 !== cagr10, `5Y=${cagr5} 10Y=${cagr10}`);
    assert.ok(cagr5 > cagr10, 'recent 5Y should be steeper than the full decade');
  });

  it('uses PE15 below 15% growth and Lynch PEG at/above 15%', () => {
    const below = fairValueRatioFromGrowth(14.9);
    assert.equal(below.rule, 'pe15');
    assert.equal(below.ratio, 15);

    const atFloor = fairValueRatioFromGrowth(15);
    assert.equal(atFloor.rule, 'lynch_peg');
    assert.equal(atFloor.ratio, 15);

    const fast = fairValueRatioFromGrowth(20);
    assert.equal(fast.rule, 'lynch_peg');
    assert.equal(fast.ratio, 20);
  });

  it('anchors fair value on today\'s price for the selected window', () => {
    const { summary } = buildValuationSeries(mixedGrowthHistory(), 'eps', {
      currentPrice: 125.5,
      windowYears: 5,
    });
    assert.equal(summary.windowYears, 5);
    assert.equal(summary.currentPrice, 125.5);
    assert.ok(summary.growthRatePct != null);
    assert.ok(summary.fairValue != null);
    assert.ok(summary.premiumPct != null);
  });
});
