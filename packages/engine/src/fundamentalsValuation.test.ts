import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  appendForwardFairValue,
  buildValuationSeries,
  cagrPct,
  closeOnOrBefore,
  fairValueFromEstimate,
  fairValueRatioFromGrowth,
  forwardEstimatesForMetric,
  isoDayDiff,
  seriesForFairValueChart,
  sliceToWindow,
  trailingMetricCagr,
  ttmFromQuarterly,
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

  it('always adds a today-point even when last FY equals headline', () => {
    const { series, summary } = buildValuationSeries(mixedGrowthHistory(), 'eps', {
      currentPrice: 110,
      windowYears: 5,
    });
    const expected = cagrPct(1.25, 7.6, 5);
    assert.ok(expected != null && summary.growthRatePct != null);
    assert.ok(Math.abs(summary.growthRatePct - expected) < 1e-9);
    const chart = seriesForFairValueChart(series, summary, '2026-06-15');
    assert.equal(chart.length, series.length + 1);
    const last = chart[chart.length - 1];
    assert.equal(last?.date, '2026-06-15');
    assert.equal(last?.fairValue, summary.fairValue);
    assert.equal(last?.estimated, true);
    assert.equal(chart[chart.length - 2]?.date, '2025-12-31');
    assert.equal(chart[chart.length - 2]?.fairValue, summary.fairValue);
  });

  it('does not duplicate a today-point that already matches headline', () => {
    const { series, summary } = buildValuationSeries(mixedGrowthHistory(), 'eps', {
      currentPrice: 110,
      windowYears: 5,
      ttmMetric: 8,
    });
    const once = seriesForFairValueChart(series, summary, '2026-06-15');
    const twice = seriesForFairValueChart(once, summary, '2026-06-15');
    assert.equal(twice.length, once.length);
    assert.equal(twice[twice.length - 1]?.date, '2026-06-15');
    assert.equal(twice[twice.length - 1]?.fairValue, summary.fairValue);
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
    assert.notEqual(forecast[0]?.fairValue, forecast[1]?.fairValue);
    assert.notEqual(forecast[1]?.fairValue, forecast[2]?.fairValue);
    assert.ok(
      forecast.every((p) => p.fairValue != null && p.fairValue !== summary.fairValue),
      'each year is its own EPS × ratio, not the single 3y headline',
    );
    const ttm = withFwd.find((p) => p.estimated && !p.forecast);
    assert.ok(ttm);
    assert.equal(ttm.date, '2026-06-15');
    assert.equal(ttm.fairValue, summary.fairValue);
    assert.equal(ttm.forecast, undefined);
  });

  it('copies annual dividend onto the valuation series', () => {
    const base = mixedGrowthHistory();
    const hist = base.map((p, i) => ({
      ...p,
      dividend: i === base.length - 1 ? 1.1 : 0.55,
    }));
    const { series, summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 110,
      windowYears: 5,
    });
    const withDiv = series.filter((p) => p.dividend != null && p.dividend > 0);
    assert.ok(withDiv.length >= 2);
    assert.equal(series[series.length - 1]?.dividend, 1.1);
    const chart = seriesForFairValueChart(series, summary, '2026-06-15');
    const today = chart[chart.length - 1];
    assert.equal(today?.estimated, true);
    assert.equal(today?.dividend ?? null, null);
  });

  it('sets Normal P/E forecast as EPS × window median multiple', () => {
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
      ],
      summary.fairValueRatio,
      3,
      summary.normalMultiple,
    );
    const forecast = withFwd.filter((p) => p.forecast);
    assert.equal(forecast.length, 3);
    assert.ok(summary.normalMultiple > 0);
    assert.equal(forecast[0]?.normalValue, 9 * summary.normalMultiple);
    assert.equal(forecast[1]?.normalValue, 10 * summary.normalMultiple);
    assert.equal(forecast[2]?.normalValue, 12 * summary.normalMultiple);
    assert.notEqual(forecast[0]?.normalValue, forecast[0]?.fairValue);
  });

  it('places forecast years on the last historical FY-end, not a stale FMP date', () => {
    const { series, summary } = buildValuationSeries(mixedGrowthHistory(), 'eps', {
      currentPrice: 110,
      windowYears: 5,
      ttmMetric: 8,
    });
    const chart = seriesForFairValueChart(series, summary, '2026-08-16');
    const withFwd = appendForwardFairValue(
      chart,
      [
        { year: 2026, date: '2026-02-15', eps: 9 },
        { year: 2027, date: '2027-02-20', eps: 10 },
      ],
      15,
    );
    const fwd = withFwd.filter((p) => p.forecast);
    assert.deepEqual(
      fwd.map((p) => p.date),
      ['2026-12-31', '2027-12-31'],
    );
    assert.equal(fwd[0]?.fairValue, 9 * 15);
    assert.equal(fwd[1]?.fairValue, 10 * 15);
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
    assert.equal(fwd[0]!.date, '2026-12-31');
    assert.ok(fwd[0]!.date > '2026-08-16');
    assert.equal(fwd[0]!.fairValue, 9 * 15);
  });

  it('aligns forecast FY-end to a non-December fiscal year', () => {
    const juneHist: AnnualFundamentalPoint[] = [
      { ...fy(2024, 4, 60), date: '2024-06-30' },
      { ...fy(2025, 5, 70), date: '2025-06-30' },
    ];
    const { series, summary } = buildValuationSeries(juneHist, 'eps', {
      currentPrice: 80,
      windowYears: 1,
      ttmMetric: 5.2,
    });
    const chart = seriesForFairValueChart(series, summary, '2026-08-16');
    const withFwd = appendForwardFairValue(
      chart,
      [
        { year: 2026, date: '2026-02-01', eps: 6 },
        { year: 2027, date: '2027-02-01', eps: 7 },
      ],
      summary.fairValueRatio,
    );
    const fwd = withFwd.filter((p) => p.forecast);
    assert.equal(fwd.length, 1);
    assert.equal(fwd[0]!.year, 2027);
    assert.equal(fwd[0]!.date, '2027-06-30');
  });

  it('sums the last four completed quarters for TTM', () => {
    const got = ttmFromQuarterly(
      [
        { date: '2025-08-28', eps: 2.1 },
        { date: '2025-11-27', eps: 1.8 },
        { date: '2026-02-26', eps: 1.5 },
        { date: '2026-05-28', eps: 1.9 },
        { date: '2026-08-28', eps: 9.0 },
      ],
      '2026-08-16',
    );
    assert.equal(got.ttm, 2.1 + 1.8 + 1.5 + 1.9);
    assert.equal(got.asOf, '2026-05-28');
  });

  it('does not form TTM across a missing-quarter gap', () => {
    const got = ttmFromQuarterly(
      [
        { date: '2024-05-28', eps: 1 },
        { date: '2025-08-28', eps: 2 },
        { date: '2025-11-27', eps: 3 },
        { date: '2026-05-28', eps: 4 },
      ],
      '2026-08-16',
    );
    assert.equal(got.ttm, null);
    assert.equal(got.asOf, null);
  });

  it('picks the fiscal year-end close, not December', () => {
    const bars = [
      { date: '2025-08-28', close: 120 },
      { date: '2025-08-29', close: 121 },
      { date: '2025-12-31', close: 200 },
    ];
    assert.equal(closeOnOrBefore(bars, '2025-08-28'), 120);
    assert.equal(closeOnOrBefore(bars, '2025-08-30'), 121);
  });

  it('does not plant a current-FY estimate 12 days after the TTM point', () => {
    const muHist: AnnualFundamentalPoint[] = [
      { ...fy(2024, 1.3, 90), date: '2024-08-29' },
      { ...fy(2025, 8.29, 140), date: '2025-08-28' },
    ];
    const { series, summary } = buildValuationSeries(muHist, 'eps', {
      currentPrice: 971,
      windowYears: 5,
      ttmMetric: 1.48,
    });
    const chart = seriesForFairValueChart(series, summary, '2026-08-16');
    const withFwd = appendForwardFairValue(
      chart,
      [
        { year: 2026, date: '2026-08-28', eps: 72.21 },
        { year: 2027, date: '2027-08-28', eps: 155.15 },
        { year: 2028, date: '2028-08-28', eps: 167.25 },
      ],
      summary.fairValueRatio,
      3,
      summary.normalMultiple,
    );
    const ttm = withFwd.find((p) => p.estimated && !p.forecast);
    assert.ok(ttm);
    assert.equal(ttm.date, '2026-08-16');
    const forecast = withFwd.filter((p) => p.forecast);
    assert.deepEqual(
      forecast.map((p) => p.year),
      [2027, 2028],
    );
    assert.equal(forecast[0]?.date, '2027-08-28');
    assert.ok(
      forecast.every((p) => isoDayDiff(ttm.date, p.date) > 90),
      'no forecast vertex within 90 days of the TTM today-point',
    );
    const spikeFv = 72.21 * (summary.fairValueRatio as number);
    assert.ok(!forecast.some((p) => p.fairValue === spikeFv));
  });

  it('does not attach EPS estimates to non-EPS metrics', () => {
    const estimates = [
      { year: 2026, date: '2026-12-31', eps: 72 },
      { year: 2027, date: '2027-12-31', eps: 80 },
    ];
    assert.equal(forwardEstimatesForMetric('eps', estimates).length, 2);
    assert.deepEqual(forwardEstimatesForMetric('fcf', estimates), []);
    assert.deepEqual(forwardEstimatesForMetric('revenue', estimates), []);
    assert.deepEqual(forwardEstimatesForMetric('ownerEarnings', estimates), []);
  });

  it('computes table fair value as EPS × ratio', () => {
    assert.equal(fairValueFromEstimate(23.83, 31.06), 23.83 * 31.06);
    assert.equal(fairValueFromEstimate(null, 15), null);
    assert.equal(fairValueFromEstimate(10, null), null);
  });
});
