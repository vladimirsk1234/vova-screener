import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  appendForwardFairValue,
  appendIntraYearTtmSteps,
  appendNextQuarterEstimate,
  buildValuationSeries,
  cagrPct,
  fairValueFromEstimate,
  fairValueRatioFromGrowth,
  growthOverrideFromSummary,
  nextQuarterIso,
  projectMetricByGrowth,
  seriesForFairValueChart,
  firstForecastDate,
  firstNonForecastDate,
  lastForecastDate,
  lastSeriesDate,
  sliceToWindow,
  forwardMetricCagr,
  trailingMetricCagr,
  ttmFromQuarterly,
  valuationChartLogicalRange,
  valuationChartRange,
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

  it('keeps four fiscal years for a 3Y window', () => {
    const sliced = sliceToWindow(mixedGrowthHistory(), 3);
    assert.deepEqual(
      sliced.map((p) => p.year),
      [2022, 2023, 2024, 2025],
    );
  });

  it('keeps nine fiscal years for an 8Y window', () => {
    const sliced = sliceToWindow(mixedGrowthHistory(), 8);
    assert.deepEqual(
      sliced.map((p) => p.year),
      [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
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

  it('uses GDF / GDF…P/E=G / P/E=G bands at 5% and 15%', () => {
    const slow = fairValueRatioFromGrowth(4.9);
    assert.equal(slow.rule, 'gdf');
    assert.equal(slow.ratio, 15);

    const atGraham = fairValueRatioFromGrowth(5);
    assert.equal(atGraham.rule, 'gdf_pe_g');
    assert.equal(atGraham.ratio, 15);

    const mid = fairValueRatioFromGrowth(10);
    assert.equal(mid.rule, 'gdf_pe_g');
    assert.equal(mid.ratio, 15);

    const belowLynch = fairValueRatioFromGrowth(14.9);
    assert.equal(belowLynch.rule, 'gdf_pe_g');
    assert.equal(belowLynch.ratio, 15);

    const atLynch = fairValueRatioFromGrowth(15);
    assert.equal(atLynch.rule, 'pe_g');
    assert.equal(atLynch.ratio, 15);

    const fast = fairValueRatioFromGrowth(20);
    assert.equal(fast.rule, 'pe_g');
    assert.equal(fast.ratio, 20);
  });

  it('blocks Lynch when CAGR span is under 2 years on a 5Y window (LYFT-style)', () => {
    const hist = [
      fy(2023, -0.88, 12),
      fy(2024, 0.06, 14),
      fy(2025, 6.81, 17),
    ];
    const { summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 17.12,
      windowYears: 5,
      ttmMetric: 6.88,
    });
    assert.equal(summary.growthSpanYears, 1);
    assert.ok(summary.growthRatePct != null && summary.growthRatePct > 1000);
    assert.equal(summary.fairValueRule, 'gdf_pe_g');
    assert.equal(summary.fairValueRatio, 15);
    assert.ok(summary.fairValue != null);
    assert.ok(Math.abs(summary.fairValue - 6.88 * 15) < 1e-6);
  });

  it('still allows Lynch on a 1Y YoY even when span is 1 year', () => {
    const hist = [
      fy(2024, 2.0, 30),
      fy(2025, 2.88, 40),
    ];
    const { summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 40,
      windowYears: 1,
      ttmMetric: 2.88,
    });
    assert.equal(summary.growthSpanYears, 1);
    assert.ok(summary.growthRatePct != null);
    assert.ok(Math.abs(summary.growthRatePct - 44) < 0.1);
    assert.equal(summary.fairValueRule, 'pe_g');
    assert.equal(summary.fairValueRatio, Math.round(summary.growthRatePct! * 100) / 100);
    assert.ok(summary.fairValue != null);
    assert.ok(Math.abs(summary.fairValue - 2.88 * (summary.fairValueRatio as number)) < 1e-6);
  });

  it('uses 15× when only one profitable FY exists in a multi-year window', () => {
    const hist = [fy(2025, 2.5, 40)];
    const { summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 40,
      windowYears: 5,
      ttmMetric: 2.5,
    });
    assert.equal(summary.growthRatePct, null);
    assert.equal(summary.growthSpanYears, null);
    assert.equal(summary.fairValueRule, 'gdf_pe_g');
    assert.equal(summary.fairValueRatio, 15);
    assert.equal(summary.fairValue, 2.5 * 15);
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

  it('uses forward CAGR through the last positive estimate when analysts are present', () => {
    const hist = mixedGrowthHistory();
    const windowed = sliceToWindow(hist, 5);
    const forward = [
      { year: 2026, metric: 20 },
      { year: 2027, metric: 30 },
    ];
    const fwd = forwardMetricCagr(windowed, forward, 'eps');
    const { summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 110,
      windowYears: 5,
      forward,
    });
    assert.equal(summary.growthSource, 'forward');
    assert.ok(fwd != null && summary.growthRatePct != null);
    assert.ok(Math.abs(summary.growthRatePct - fwd) < 1e-9);
  });

  it('PDD-like 3Y + estimates: ratio ~23× and FV near $230, not $1600', () => {
    const hist = [
      fy(2022, 4.08, 40),
      fy(2023, 6.5, 70),
      fy(2024, 9.4, 85),
      fy(2025, 10.07, 90),
    ];
    const { summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 90,
      windowYears: 3,
      ttmMetric: 10.07,
      forward: [
        { year: 2026, metric: 9.98 },
        { year: 2027, metric: 12.08 },
        { year: 2028, metric: 13.88 },
      ],
    });
    assert.equal(summary.growthSource, 'forward');
    assert.ok(summary.growthRatePct != null);
    assert.ok(
      summary.growthRatePct > 20 && summary.growthRatePct < 26,
      `g=${summary.growthRatePct}`,
    );
    assert.equal(summary.fairValueRule, 'pe_g');
    assert.ok(summary.fairValueRatio != null);
    assert.ok(
      summary.fairValueRatio > 20 && summary.fairValueRatio < 26,
      `ratio=${summary.fairValueRatio}`,
    );
    assert.ok(summary.fairValue != null);
    assert.ok(
      summary.fairValue > 210 && summary.fairValue < 280,
      `fv=${summary.fairValue}`,
    );
  });

  it('PDD-like FCF uses EPS forward CAGR, not lumpy trailing FCF', () => {
    const hist: AnnualFundamentalPoint[] = [
      { ...fy(2022, 4.08, 40), fcfPerShare: 0.5 },
      { ...fy(2023, 6.5, 70), fcfPerShare: 1.2 },
      { ...fy(2024, 9.4, 85), fcfPerShare: 3.0 },
      { ...fy(2025, 10.07, 90), fcfPerShare: 8.0 },
    ];
    const forward = [
      { year: 2026, metric: 9.98 },
      { year: 2027, metric: 12.08 },
      { year: 2028, metric: 13.88 },
    ];
    const trailing = buildValuationSeries(hist, 'fcf', {
      currentPrice: 90,
      windowYears: 3,
      ttmMetric: 8,
    });
    assert.ok(
      trailing.summary.fairValue != null && trailing.summary.fairValue > 400,
      `trailing fv=${trailing.summary.fairValue}`,
    );

    const eps = buildValuationSeries(hist, 'eps', {
      currentPrice: 90,
      windowYears: 3,
      ttmMetric: 10.07,
      forward,
    });
    const fcf = buildValuationSeries(hist, 'fcf', {
      currentPrice: 90,
      windowYears: 3,
      ttmMetric: 8,
      ...growthOverrideFromSummary(eps.summary),
    });
    assert.equal(fcf.summary.growthSource, 'forward');
    assert.ok(eps.summary.growthRatePct != null && fcf.summary.growthRatePct != null);
    assert.ok(Math.abs(fcf.summary.growthRatePct - eps.summary.growthRatePct) < 1e-9);
    assert.ok(fcf.summary.fairValueRatio != null);
    assert.ok(Math.abs((fcf.summary.fairValue ?? 0) - 8 * fcf.summary.fairValueRatio) < 1e-6);
    assert.ok(
      fcf.summary.fairValue != null && fcf.summary.fairValue > 160 && fcf.summary.fairValue < 280,
      `fcf fv=${fcf.summary.fairValue}`,
    );
    assert.notEqual(fcf.summary.fairValue, trailing.summary.fairValue);
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

  it('steps intra-year fair value on each reported quarter', () => {
    const muHist: AnnualFundamentalPoint[] = [
      { ...fy(2024, 1.3, 90), date: '2024-08-29' },
      { ...fy(2025, 8.29, 140), date: '2025-08-28' },
    ];
    const quarters = [
      { date: '2024-11-28', eps: 1.2 },
      { date: '2025-02-27', eps: 1.5 },
      { date: '2025-05-29', eps: 2.1 },
      { date: '2025-08-28', eps: 3.49 },
      { date: '2025-11-27', eps: 4.78 },
      { date: '2026-02-26', eps: 12.2 },
      { date: '2026-05-28', eps: 25.11 },
    ];
    const { series, summary } = buildValuationSeries(muHist, 'eps', {
      currentPrice: 971,
      windowYears: 5,
      ttmMetric: ttmFromQuarterly(quarters, '2026-08-16').ttm,
    });
    const histFv = series.map((p) => p.fairValue);
    const stepped = appendIntraYearTtmSteps(
      series,
      quarters,
      summary.fairValueRatio,
      '2026-08-16',
    );
    const intra = stepped.filter((p) => p.estimated && !p.forecast);
    assert.deepEqual(
      intra.map((p) => p.date),
      ['2025-11-27', '2026-02-26', '2026-05-28'],
    );
    assert.ok(summary.fairValueRatio != null);
    const fvNov = ttmFromQuarterly(quarters, '2025-11-27').ttm! * summary.fairValueRatio;
    const fvFeb = ttmFromQuarterly(quarters, '2026-02-26').ttm! * summary.fairValueRatio;
    const fvMay = ttmFromQuarterly(quarters, '2026-05-28').ttm! * summary.fairValueRatio;
    assert.ok(Math.abs((intra[0]?.fairValue ?? 0) - fvNov) < 1e-9);
    assert.ok(Math.abs((intra[1]?.fairValue ?? 0) - fvFeb) < 1e-9);
    assert.ok(Math.abs((intra[2]?.fairValue ?? 0) - fvMay) < 1e-9);
    assert.ok(fvMay > fvFeb && fvFeb > fvNov);
    assert.deepEqual(
      stepped.filter((p) => !p.estimated).map((p) => p.fairValue),
      histFv,
    );

    const noToday = seriesForFairValueChart(stepped, summary, '2026-08-16', {
      pinToday: false,
    });
    assert.equal(noToday[noToday.length - 1]?.date, '2026-05-28');
    assert.ok(!noToday.some((p) => p.date === '2026-08-16'));
    assert.equal(noToday[noToday.length - 1]?.fairValue, fvMay);

    const estimates = [
      { year: 2026, date: '2026-08-28', eps: 72.21 },
      { year: 2027, date: '2027-08-28', eps: 155.15 },
      { year: 2028, date: '2028-08-28', eps: 167.25 },
    ];
    const withNext = appendNextQuarterEstimate(
      noToday,
      '2026-09-23',
      estimates,
      summary.fairValueRatio,
    );
    const nextQ = withNext[withNext.length - 1];
    assert.equal(nextQ?.date, '2026-09-23');
    assert.equal(nextQ?.forecast, true);
    assert.ok(nextQ?.fairValue != null && nextQ.fairValue > fvMay);
    assert.ok(
      Math.abs((nextQ?.fairValue ?? 0) - 72.21 * (summary.fairValueRatio as number)) < 1e-9,
    );

    const withFwd = appendForwardFairValue(
      withNext,
      estimates,
      summary.fairValueRatio,
    );
    const forecast = withFwd.filter((p) => p.forecast);
    assert.ok(!forecast.some((p) => p.date === '2026-08-28'));
    assert.deepEqual(
      forecast.filter((p) => p.date !== '2026-09-23').map((p) => p.year),
      [2027, 2028],
    );
    assert.ok(
      forecast[0]!.fairValue != null &&
        forecast[1]!.fairValue != null &&
        forecast[0]!.fairValue !== forecast[1]!.fairValue,
    );
  });

  it('steps intra-year FCF fair value on each reported quarter', () => {
    const hist: AnnualFundamentalPoint[] = [
      { ...fy(2024, 1.3, 90), date: '2024-08-29', fcfPerShare: 1.1 },
      { ...fy(2025, 8.29, 140), date: '2025-08-28', fcfPerShare: 2.4 },
    ];
    const quarters = [
      { date: '2024-11-28', metric: 0.4 },
      { date: '2025-02-27', metric: 0.5 },
      { date: '2025-05-29', metric: 0.6 },
      { date: '2025-08-28', metric: 0.9 },
      { date: '2025-11-27', metric: 1.2 },
      { date: '2026-02-26', metric: 2.8 },
      { date: '2026-05-28', metric: 4.1 },
    ];
    const { series, summary } = buildValuationSeries(hist, 'fcf', {
      currentPrice: 971,
      windowYears: 5,
      ttmMetric: ttmFromQuarterly(quarters, '2026-08-16').ttm,
    });
    const stepped = appendIntraYearTtmSteps(
      series,
      quarters,
      summary.fairValueRatio,
      '2026-08-16',
    );
    const intra = stepped.filter((p) => p.estimated && !p.forecast);
    assert.deepEqual(
      intra.map((p) => p.date),
      ['2025-11-27', '2026-02-26', '2026-05-28'],
    );
    const fvNov = ttmFromQuarterly(quarters, '2025-11-27').ttm! * (summary.fairValueRatio as number);
    const fvMay = ttmFromQuarterly(quarters, '2026-05-28').ttm! * (summary.fairValueRatio as number);
    assert.ok(fvMay > fvNov);
    assert.ok(Math.abs((intra[0]?.fairValue ?? 0) - fvNov) < 1e-9);
    assert.ok(Math.abs((intra[2]?.fairValue ?? 0) - fvMay) < 1e-9);

    const noToday = seriesForFairValueChart(stepped, summary, '2026-08-16', {
      pinToday: false,
    });
    assert.equal(noToday[noToday.length - 1]?.date, '2026-05-28');
    assert.ok(!noToday.some((p) => p.date === '2026-08-16'));

    const ttmMay = ttmFromQuarterly(quarters, '2026-05-28').ttm;
    const projected = projectMetricByGrowth({
      lastMetric: ttmMay,
      lastYear: 2025,
      growthPct: summary.growthRatePct,
      years: [{ year: 2026, date: '2026-08-28' }],
    });
    const withNext = appendNextQuarterEstimate(
      noToday,
      '2026-09-23',
      projected,
      summary.fairValueRatio,
    );
    const nextQ = withNext[withNext.length - 1];
    assert.equal(nextQ?.date, '2026-09-23');
    assert.equal(nextQ?.forecast, true);
    assert.ok(nextQ?.fairValue != null && nextQ.fairValue !== fvMay);
    const epsAsFv = 12 * (summary.fairValueRatio as number);
    assert.ok(
      Math.abs((nextQ?.fairValue ?? 0) - epsAsFv) > 0.5,
      'next-quarter FCF FV must not be EPS × ratio',
    );
  });

  it('uses next earnings date, else +91 days', () => {
    assert.equal(nextQuarterIso('2026-05-31', '2026-09-23'), '2026-09-23');
    assert.equal(nextQuarterIso('2026-05-31', '2026-05-20'), '2026-08-30');
    assert.equal(nextQuarterIso('2026-05-31', null), '2026-08-30');
  });

  it('slopes the first dashed point toward the next FY estimate, not today', () => {
    const hist: AnnualFundamentalPoint[] = [
      { ...fy(2024, 8, 120), date: '2024-11-30' },
      { ...fy(2025, 10, 150), date: '2025-11-30' },
    ];
    const quarters = [
      { date: '2025-02-28', eps: 2.2 },
      { date: '2025-05-31', eps: 2.4 },
      { date: '2025-08-31', eps: 2.6 },
      { date: '2025-11-30', eps: 2.8 },
      { date: '2026-02-28', eps: 3.0 },
      { date: '2026-05-31', eps: 3.2 },
    ];
    const { series, summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 140,
      windowYears: 5,
      ttmMetric: ttmFromQuarterly(quarters, '2026-08-16').ttm,
    });
    const stepped = appendIntraYearTtmSteps(
      series,
      quarters,
      summary.fairValueRatio,
      '2026-08-16',
    );
    const lastQ = stepped[stepped.length - 1];
    assert.equal(lastQ?.date, '2026-05-31');
    const lastFv = lastQ?.fairValue as number;
    assert.ok(lastFv > 0);

    const chart = seriesForFairValueChart(stepped, summary, '2026-08-16', {
      pinToday: false,
    });
    assert.ok(!chart.some((p) => p.date === '2026-08-16'));

    const withNext = appendNextQuarterEstimate(
      chart,
      '2026-09-25',
      [{ year: 2026, date: '2026-11-30', eps: 14 }],
      summary.fairValueRatio,
    );
    const nextQ = withNext[withNext.length - 1];
    assert.equal(nextQ?.date, '2026-09-25');
    assert.equal(nextQ?.forecast, true);
    const target = 14 * (summary.fairValueRatio as number);
    assert.ok(nextQ?.fairValue != null);
    assert.ok(nextQ.fairValue !== lastFv, 'not a flat copy of last TTM');
    assert.ok(nextQ.fairValue !== target, 'partial quarter, not the full FY estimate');
    assert.ok(nextQ.fairValue > lastFv && nextQ.fairValue < target);

    const withFwd = appendForwardFairValue(
      withNext,
      [
        { year: 2026, date: '2026-11-30', eps: 14 },
        { year: 2027, date: '2027-11-30', eps: 16 },
      ],
      summary.fairValueRatio,
    );
    const fy2026 = withFwd.find((p) => p.forecast && p.date === '2026-11-30');
    assert.ok(fy2026);
    assert.equal(fy2026.fairValue, target);
    assert.ok(fy2026.fairValue !== nextQ.fairValue);
  });

  it('prefers estimate.metric over eps so FCF forecasts stay in FCF dollars', () => {
    const hist = [fy(2024, 8, 120), fy(2025, 10, 150)];
    hist[0]!.fcfPerShare = 4;
    hist[1]!.fcfPerShare = 5;
    const { series, summary } = buildValuationSeries(hist, 'fcf', {
      currentPrice: 150,
      windowYears: 5,
      ttmMetric: 5,
    });
    const mixed = appendForwardFairValue(
      series,
      [{ year: 2026, date: '2026-12-31', eps: 14, metric: 6 }],
      summary.fairValueRatio,
    );
    const fwd = mixed.find((p) => p.forecast && p.year === 2026);
    assert.ok(fwd);
    assert.equal(fwd.metric, 6);
    assert.equal(fwd.fairValue, 6 * (summary.fairValueRatio as number));
    assert.notEqual(fwd.fairValue, 14 * (summary.fairValueRatio as number));
  });

  it('projects FCF at (1+g)^Δt, not at the EPS estimate level', () => {
    const pts = projectMetricByGrowth({
      lastMetric: 8,
      lastYear: 2025,
      growthPct: 25,
      years: [
        { year: 2026, date: '2026-12-31' },
        { year: 2027, date: '2027-12-31' },
      ],
    });
    assert.equal(pts.length, 2);
    assert.ok(Math.abs((pts[0]?.metric ?? 0) - 10) < 1e-9);
    assert.ok(Math.abs((pts[1]?.metric ?? 0) - 12.5) < 1e-9);
    assert.notEqual(pts[0]?.metric, 14);
  });

  it('computes table fair value as EPS × ratio', () => {
    assert.equal(fairValueFromEstimate(23.83, 31.06), 23.83 * 31.06);
    assert.equal(fairValueFromEstimate(null, 15), null);
    assert.equal(fairValueFromEstimate(10, null), null);
  });
});

describe('valuationChartRange', () => {
  const firstBar = '2015-01-02';
  const lastBar = '2026-08-14';
  const firstFy5 = '2020-12-31';
  const firstFy8 = '2017-12-31';
  const lastForecast = '2028-12-31';

  it('keeps the full 5Y fiscal start and includes the +3y forecast tail', () => {
    const range = valuationChartRange({
      firstBarDate: firstBar,
      lastBarDate: lastBar,
      windowYears: 5,
      firstHistoricalDate: firstFy5,
      firstForecastDate: lastForecast,
      lastForecastDate: lastForecast,
    });
    assert.ok(range.from <= firstFy5, `from=${range.from} should include ${firstFy5}`);
    assert.ok(range.from < '2023-01-01', `from=${range.from} must not start in 2023`);
    assert.ok(range.to >= lastBar, `to=${range.to} should reach last price`);
    assert.equal(range.to, lastForecast);
  });

  it('keeps the full 8Y fiscal start', () => {
    const range = valuationChartRange({
      firstBarDate: firstBar,
      lastBarDate: lastBar,
      windowYears: 8,
      firstHistoricalDate: firstFy8,
      firstForecastDate: lastForecast,
    });
    assert.ok(range.from <= firstFy8, `from=${range.from} should include ${firstFy8}`);
  });

  it('never computes from as lastForecast minus N years', () => {
    const range = valuationChartRange({
      firstBarDate: firstBar,
      lastBarDate: lastBar,
      windowYears: 5,
      firstHistoricalDate: firstFy5,
      firstForecastDate: lastForecast,
    });
    assert.notEqual(range.from, '2023-12-31');
    assert.ok(range.from < '2023-01-01');
  });

  it('uses the first bar for MAX', () => {
    const range = valuationChartRange({
      firstBarDate: firstBar,
      lastBarDate: lastBar,
      windowYears: null,
      firstHistoricalDate: '2015-12-31',
      firstForecastDate: lastForecast,
    });
    assert.equal(range.from, firstBar);
    assert.equal(range.to, lastForecast);
  });

  it('keeps 5Y price history when the first FCF point is late', () => {
    const range = valuationChartRange({
      firstBarDate: firstBar,
      lastBarDate: lastBar,
      windowYears: 5,
      firstHistoricalDate: '2023-12-31',
      firstForecastDate: lastForecast,
    });
    assert.ok(range.from <= '2021-08-14', `from=${range.from} should be lastBar − 5y`);
    assert.ok(range.from < '2023-01-01');
  });

  it('includes a near forecast point in to', () => {
    const range = valuationChartRange({
      firstBarDate: firstBar,
      lastBarDate: lastBar,
      windowYears: 1,
      firstHistoricalDate: '2025-12-31',
      firstForecastDate: '2026-10-15',
      lastForecastDate: '2026-10-15',
    });
    assert.equal(range.to, '2026-10-15');
    assert.ok(range.from <= '2025-08-14');
  });

  it('includes the last DCF date in to', () => {
    const range = valuationChartRange({
      firstBarDate: firstBar,
      lastBarDate: lastBar,
      windowYears: 5,
      firstHistoricalDate: firstFy5,
      firstForecastDate: lastForecast,
      lastExtraDate: '2029-12-31',
    });
    assert.equal(range.to, '2029-12-31');
    assert.ok(range.from <= firstFy5);
  });

  it('reads first historical and forecast dates from a mixed series', () => {
    const series = [
      { date: '2020-12-31', forecast: false },
      { date: '2025-12-31', forecast: false },
      { date: '2026-10-15', forecast: true },
      { date: '2028-12-31', forecast: true },
    ];
    assert.equal(firstNonForecastDate(series), '2020-12-31');
    assert.equal(firstForecastDate(series), '2026-10-15');
    assert.equal(lastForecastDate(series), '2028-12-31');
    assert.equal(lastSeriesDate(series), '2028-12-31');
  });

  it('maps 8Y onto bar indices near 2017–2018, not 2021, and never goes negative', () => {
    const weekMs = 7 * 86_400_000;
    const start = Date.parse('2015-01-02T00:00:00Z');
    const end = Date.parse('2028-12-29T00:00:00Z');
    const timesMs: number[] = [];
    for (let t = start; t <= end; t += weekMs) timesMs.push(t);
    const range = valuationChartRange({
      firstBarDate: firstBar,
      lastBarDate: lastBar,
      windowYears: 8,
      firstHistoricalDate: firstFy8,
      firstForecastDate: lastForecast,
      lastForecastDate: lastForecast,
    });
    const logical = valuationChartLogicalRange(timesMs, range);
    assert.ok(logical.fromIdx >= 0);
    assert.ok(logical.toIdx >= logical.fromIdx);
    const fromIso = new Date(timesMs[logical.fromIdx]!).toISOString().slice(0, 10);
    const toIso = new Date(timesMs[logical.toIdx]!).toISOString().slice(0, 10);
    assert.ok(fromIso <= '2018-01-07', `fromIdx date ${fromIso} should be ~2017–2018`);
    assert.ok(fromIso < '2021-01-01', `fromIdx date ${fromIso} must not be 2021`);
    assert.ok(toIso >= '2028-12-29', `toIdx date ${toIso} should include the 3y forecast tail`);
  });

  it('plots 8Y fair value from the first window year even when early EPS is a loss', () => {
    const hist = [
      fy(2017, -0.4, 12),
      fy(2018, -0.2, 13),
      fy(2019, 0, 14),
      fy(2020, -0.1, 15),
      fy(2021, 1.8, 30),
      fy(2022, 2.6, 45),
      fy(2023, 3.7, 60),
      fy(2024, 5.3, 80),
      fy(2025, 7.6, 110),
    ];
    const { series, summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 110,
      windowYears: 8,
    });
    assert.equal(series[0]?.year, 2017);
    assert.equal(series[0]?.fairValue, 0);
    assert.equal(series[0]?.normalValue, 0);
    const firstProfit = series.find((p) => p.year === 2021);
    assert.ok(firstProfit && firstProfit.fairValue != null && firstProfit.fairValue > 0);
    assert.ok(summary.fairValue != null && summary.fairValue > 0);
  });

  it('steps Normal P/E on intra-year TTM quarters after the last FY', () => {
    const hist = [fy(2024, 5.3, 80), fy(2025, 7.6, 110)];
    const quarters = [
      { date: '2025-03-31', eps: 1.5 },
      { date: '2025-06-30', eps: 1.8 },
      { date: '2025-09-30', eps: 2.0 },
      { date: '2025-12-31', eps: 2.3 },
      { date: '2026-03-31', eps: 2.5 },
    ];
    const { series, summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 110,
      windowYears: 5,
    });
    const stepped = appendIntraYearTtmSteps(
      series,
      quarters,
      summary.fairValueRatio,
      '2026-08-16',
      summary.normalMultiple,
    );
    const lastQ = stepped.find((p) => p.date === '2026-03-31');
    assert.ok(lastQ?.estimated);
    assert.ok(!lastQ?.forecast);
    const ttm = ttmFromQuarterly(quarters, '2026-03-31').ttm;
    assert.ok(ttm != null && ttm > 0);
    assert.ok(Math.abs((lastQ?.normalValue ?? 0) - ttm * summary.normalMultiple) < 1e-9);
    const withFwd = appendForwardFairValue(
      stepped,
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
    assert.ok(forecast.every((p) => p.normalValue != null && p.normalValue > 0));
  });
});
