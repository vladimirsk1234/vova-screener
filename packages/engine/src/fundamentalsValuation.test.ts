import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  appendForwardFairValue,
  buildValuationSeries,
  cagrPct,
  fairValueFromEstimate,
  fairValueRatioFromGrowth,
  seriesForFairValueChart,
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

  it('uses trailing window CAGR even when analyst estimates are present', () => {
    const hist = mixedGrowthHistory();
    const trailing = trailingMetricCagr(sliceToWindow(hist, 5), 'eps', 5);
    const { summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 110,
      windowYears: 5,
      forward: [
        { year: 2026, metric: 20 },
        { year: 2027, metric: 30 },
      ],
    });
    assert.equal(summary.growthSource, 'trailing');
    assert.ok(trailing != null && summary.growthRatePct != null);
    assert.ok(Math.abs(summary.growthRatePct - trailing) < 1e-9);
  });

  it('anchors fair value on TTM, not the first forward estimate', () => {
    const { summary } = buildValuationSeries(mixedGrowthHistory(), 'eps', {
      currentPrice: 110,
      windowYears: 5,
      ttmMetric: 8,
      forward: [{ year: 2026, metric: 20 }],
    });
    assert.equal(summary.fairValueAnchor, 8);
    assert.ok(summary.fairValueRatio != null);
    assert.ok(summary.fairValue != null);
    assert.ok(Math.abs(summary.fairValue - 8 * summary.fairValueRatio) < 1e-9);
  });

  it('produces a different 5Y fair value than 10Y on mixed growth', () => {
    const hist = mixedGrowthHistory();
    const five = buildValuationSeries(hist, 'eps', { currentPrice: 110, windowYears: 5 });
    const ten = buildValuationSeries(hist, 'eps', { currentPrice: 110, windowYears: 10 });
    assert.ok(five.summary.fairValue != null && ten.summary.fairValue != null);
    assert.ok(
      five.summary.fairValue !== ten.summary.fairValue,
      `5Y=${five.summary.fairValue} 10Y=${ten.summary.fairValue}`,
    );
    assert.ok(five.summary.growthRatePct != null && ten.summary.growthRatePct != null);
    assert.ok(five.summary.growthRatePct > ten.summary.growthRatePct);
  });

  it('pins the last chart point to the headline fair value', () => {
    const { series, summary } = buildValuationSeries(mixedGrowthHistory(), 'eps', {
      currentPrice: 110,
      windowYears: 5,
      ttmMetric: 8,
    });
    const chart = seriesForFairValueChart(series, summary, '2026-06-15');
    const last = chart[chart.length - 1];
    assert.ok(last);
    assert.equal(last.fairValue, summary.fairValue);
    assert.equal(last.date, '2026-06-15');
    assert.equal(last.estimated, true);
  });

  it('does not add a chart point when last FY already equals headline', () => {
    const { series, summary } = buildValuationSeries(mixedGrowthHistory(), 'eps', {
      currentPrice: 110,
      windowYears: 5,
    });
    const expected = cagrPct(1.25, 7.6, 5);
    assert.ok(expected != null && summary.growthRatePct != null);
    assert.ok(Math.abs(summary.growthRatePct - expected) < 1e-9);
    const chart = seriesForFairValueChart(series, summary, '2026-06-15');
    assert.equal(chart.length, series.length);
    assert.equal(chart[chart.length - 1]?.fairValue, summary.fairValue);
  });

  it('appends three forward fair-value years and leaves the TTM point solid', () => {
    const { series, summary } = buildValuationSeries(mixedGrowthHistory(), 'eps', {
      currentPrice: 110,
      windowYears: 5,
      ttmMetric: 8,
    });
    const chart = seriesForFairValueChart(series, summary, '2026-06-15');
    const withFwd = appendForwardFairValue(
      chart,
      [
        { year: 2026, date: '2026-12-31', eps: 9 },
        { year: 2027, date: '2027-12-31', eps: 10 },
        { year: 2028, date: '2028-12-31', eps: 12 },
        { year: 2029, date: '2029-12-31', eps: 14 },
      ],
      summary.fairValueRatio,
    );
    const forecast = withFwd.filter((p) => p.forecast);
    assert.equal(forecast.length, 3);
    assert.deepEqual(
      forecast.map((p) => p.year),
      [2026, 2027, 2028],
    );
    assert.ok(summary.fairValueRatio != null);
    assert.equal(forecast[0]?.fairValue, 9 * summary.fairValueRatio);
    assert.equal(forecast[1]?.fairValue, 10 * summary.fairValueRatio);
    assert.equal(forecast[2]?.fairValue, 12 * summary.fairValueRatio);
    const ttm = withFwd.find((p) => p.estimated && !p.forecast);
    assert.ok(ttm);
    assert.equal(ttm.date, '2026-06-15');
    assert.equal(ttm.fairValue, summary.fairValue);
    assert.equal(ttm.forecast, undefined);
  });

  it('bumps a forecast date that would land on or before the last solid point', () => {
    const { series, summary } = buildValuationSeries(mixedGrowthHistory(), 'eps', {
      currentPrice: 110,
      windowYears: 5,
      ttmMetric: 8,
    });
    const chart = seriesForFairValueChart(series, summary, '2026-08-16');
    const withFwd = appendForwardFairValue(
      chart,
      [{ year: 2026, date: '2026-07-31', eps: 9 }],
      15,
    );
    const fwd = withFwd.filter((p) => p.forecast);
    assert.equal(fwd.length, 1);
    assert.ok(fwd[0]!.date > '2026-08-16');
    assert.equal(fwd[0]!.fairValue, 9 * 15);
  });

  it('computes table fair value as EPS × ratio', () => {
    assert.equal(fairValueFromEstimate(23.83, 31.06), 23.83 * 31.06);
    assert.equal(fairValueFromEstimate(null, 15), null);
    assert.equal(fairValueFromEstimate(10, null), null);
  });
});
