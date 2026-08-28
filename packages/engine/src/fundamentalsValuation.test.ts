import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  appendForwardFairValue,
  appendIntraYearTtmSteps,
  appendNextQuarterEstimate,
  buildFairValueChartSeries,
  buildCardValuation,
  buildValuationSeries,
  cagrPct,
  fairValueFromEstimate,
  fairValueRatioFromGrowth,
  grahamDoddMultiple,
  growthOverrideFromSummary,
  nextQuarterIso,
  pickMetric,
  projectMetricByGrowth,
  seriesForFairValueChart,
  firstForecastDate,
  firstNonForecastDate,
  lastForecastDate,
  lastSeriesDate,
  sliceToWindow,
  estimateChainChgPct,
  fiscalYearForDate,
  forecastGrowthFromEstimates,
  orangeBoxTrailingGrowth,
  ownerEarningsPerShareFromRow,
  sumDividendsByFiscalYear,
  closeOnOrBefore,
  dividendStreak,
  DEFAULT_VALUATION_WINDOW,
  DEFAULT_CHART_VALUATION_METRIC,
  CHART_VALUATION_METRICS,
  coerceChartValuationMetric,
  coerceValuationMetric,
  forwardMetricCagr,
  trailingMetricCagr,
  ttmFromQuarterly,
  valuationChartLogicalRange,
  valuationChartRange,
  VALUATION_WINDOW_CHIPS,
  VALUATION_WINDOW_STEPS,
  availableValuationWindows,
  clampValuationWindow,
  fundamentalsHistoryBounds,
  type AnnualFundamentalPoint,
} from './fundamentalsValuation.ts';
import {
  applyStreetConsensusHistory,
  defaultEpsTtm,
} from './fundamentalsScale.ts';

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
    const noGrowth = fairValueRatioFromGrowth(0);
    assert.equal(noGrowth.rule, 'gdf');
    assert.equal(noGrowth.ratio, 8.5);

    const twoPct = fairValueRatioFromGrowth(2);
    assert.equal(twoPct.rule, 'gdf');
    assert.equal(twoPct.ratio, 12.5);

    const slow = fairValueRatioFromGrowth(4.9);
    assert.equal(slow.rule, 'gdf');
    assert.equal(slow.ratio, grahamDoddMultiple(4.9));
    assert.ok(slow.ratio != null && slow.ratio > 18 && slow.ratio < 19);

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

  it('uses 15× when trailing growth is negative, not GDF floored at 1', () => {
    assert.equal(grahamDoddMultiple(-10), 1);
    const declining = fairValueRatioFromGrowth(-4);
    assert.equal(declining.rule, 'gdf_pe_g');
    assert.equal(declining.ratio, 15);
    const nokLike = fairValueRatioFromGrowth(-2.5);
    assert.equal(nokLike.rule, 'gdf_pe_g');
    assert.equal(nokLike.ratio, 15);
    const zero = fairValueRatioFromGrowth(0);
    assert.equal(zero.rule, 'gdf');
    assert.equal(zero.ratio, 8.5);
  });

  it('exposes MAX + 15Y / 10Y / 8Y / 5Y / 3Y / 1Y window chips', () => {
    assert.deepEqual(VALUATION_WINDOW_CHIPS, [null, 15, 10, 8, 5, 3, 1]);
    assert.deepEqual([...VALUATION_WINDOW_STEPS], [1, 3, 5, 8, 10, 15]);
  });

  it('offers only lookbacks that fit FMP history, always including MAX', () => {
    assert.deepEqual(availableValuationWindows(null), [null, 15, 10, 8, 5, 3, 1]);
    assert.deepEqual(availableValuationWindows(20), [null, 15, 10, 8, 5, 3, 1]);
    assert.deepEqual(availableValuationWindows(12), [null, 10, 8, 5, 3, 1]);
    assert.deepEqual(availableValuationWindows(2), [null, 1]);
    assert.deepEqual(availableValuationWindows(0), [null]);
    assert.equal(clampValuationWindow(5, 20), 5);
    assert.equal(clampValuationWindow(5, 2), null);
    assert.equal(clampValuationWindow(null, 2), null);
    assert.equal(clampValuationWindow(15, 15), 15);
  });

  it('measures FMP history span from complete fiscal years, not a 20Y pad', () => {
    const longHist = mixedGrowthHistory();
    const long = fundamentalsHistoryBounds(longHist);
    assert.equal(long.firstDate, '2015-12-31');
    assert.equal(long.lastDate, '2025-12-31');
    assert.equal(long.spanYears, 10);
    assert.deepEqual(availableValuationWindows(long.spanYears), [null, 10, 8, 5, 3, 1]);

    const ipo = [fy(2024, 0.4, 22), fy(2025, 0.7, 28)];
    const short = fundamentalsHistoryBounds(ipo);
    assert.equal(short.firstDate, '2024-12-31');
    assert.equal(short.lastDate, '2025-12-31');
    assert.equal(short.spanYears, 1);
    assert.deepEqual(availableValuationWindows(short.spanYears), [null, 1]);
    assert.equal(clampValuationWindow(5, short.spanYears), null);
    assert.equal(clampValuationWindow(DEFAULT_VALUATION_WINDOW, long.spanYears), 5);
  });

  it('keeps eight fiscal years for a 7Y window and twenty for 19Y when history exists', () => {
    const hist = Array.from({ length: 22 }, (_, i) => fy(2004 + i, 1 + i * 0.1, 15 + i));
    const seven = sliceToWindow(hist, 7);
    assert.equal(seven[0]?.year, 2018);
    assert.equal(seven[seven.length - 1]?.year, 2025);
    assert.equal(seven.length, 8);
    const nineteen = sliceToWindow(hist, 19);
    assert.equal(nineteen[0]?.year, 2006);
    assert.equal(nineteen.length, 20);
  });

  it('computes 7Y trailing CAGR on the selected span, not a fixed 5Y', () => {
    const hist = mixedGrowthHistory();
    const cagr5 = trailingMetricCagr(sliceToWindow(hist, 5), 'eps', 5);
    const cagr7 = trailingMetricCagr(sliceToWindow(hist, 7), 'eps', 7);
    assert.ok(cagr5 != null && cagr7 != null);
    assert.ok(cagr5 !== cagr7, `5Y=${cagr5} 7Y=${cagr7}`);
  });

  it('defaults the EPS metric to GAAP and keeps NOPAT on operatingEps', () => {
    const point: AnnualFundamentalPoint = {
      ...fy(2019, 2.97, 73.49),
      operatingEps: 8.87,
      gaapEps: 2.97,
    };
    assert.equal(pickMetric(point, 'eps'), 2.97);
    assert.equal(pickMetric(point, 'operatingEps'), 8.87);
    const legacy = fy(2019, 2.97, 73.49);
    assert.equal(pickMetric(legacy, 'eps'), 2.97);
    assert.equal(pickMetric(legacy, 'operatingEps'), 2.97);
  });

  it('prefers Street consensus on eps over gaapEps when they differ', () => {
    const point: AnnualFundamentalPoint = {
      ...fy(2025, 0.26, 10.5),
      gaapEps: 0.13,
    };
    assert.equal(pickMetric(point, 'eps'), 0.26);
  });

  it('does not use NOPAT for trailing EPS CAGR when GAAP is present', () => {
    const hist: AnnualFundamentalPoint[] = [
      { ...fy(2018, 2.98, 58), operatingEps: 3.1, gaapEps: 2.98 },
      { ...fy(2019, 2.97, 73), operatingEps: 3.4, gaapEps: 2.97 },
    ];
    const gaap = trailingMetricCagr(hist, 'eps', 1);
    const op = trailingMetricCagr(hist, 'operatingEps', 1);
    assert.ok(op != null && gaap != null);
    assert.ok(Math.abs(gaap - ((2.97 / 2.98 - 1) * 100)) < 1e-6);
    assert.ok(Math.abs(op - ((3.4 / 3.1 - 1) * 100)) < 1e-6);
    assert.ok(op !== gaap);
  });

  it('maps dividend cash dates onto a September fiscal year', () => {
    assert.equal(fiscalYearForDate('2025-08-14', 9), 2025);
    assert.equal(fiscalYearForDate('2025-11-13', 9), 2026);
    assert.equal(fiscalYearForDate('2025-12-31', 12), 2025);
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

  it('applies classic GDF 8.5+2g on a slow-growth 5Y window', () => {
    const hist = [
      fy(2020, 4.0, 60),
      fy(2021, 4.08, 61),
      fy(2022, 4.16, 62),
      fy(2023, 4.24, 63),
      fy(2024, 4.32, 64),
      fy(2025, 4.4, 65),
    ];
    const { summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 65,
      windowYears: 5,
      ttmMetric: 4.4,
    });
    assert.equal(summary.growthSource, 'trailing');
    assert.ok(summary.growthRatePct != null && summary.growthRatePct < 5);
    assert.equal(summary.fairValueRule, 'gdf');
    assert.ok(summary.fairValueRatio != null);
    assert.equal(summary.fairValueRatio, grahamDoddMultiple(summary.growthRatePct!));
    assert.ok(summary.fairValueRatio < 15);
    assert.ok(summary.fairValue != null);
    assert.ok(Math.abs(summary.fairValue - 4.4 * summary.fairValueRatio) < 1e-6);
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

  it('keeps Historical orange-box growth on trailing CAGR when Street estimates exist', () => {
    const hist = mixedGrowthHistory();
    const windowed = sliceToWindow(hist, 5);
    const forward = [
      { year: 2026, metric: 20 },
      { year: 2027, metric: 30 },
    ];
    const trail = trailingMetricCagr(windowed, 'eps', 5);
    const { summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 110,
      windowYears: 5,
      forward,
    });
    assert.equal(summary.growthSource, 'trailing');
    assert.ok(trail != null && summary.growthRatePct != null);
    assert.ok(Math.abs(summary.growthRatePct - trail) < 1e-9);
    const street = forwardMetricCagr(windowed, forward, 'eps');
    assert.ok(street != null && Math.abs(street - 50) < 1e-6);
    assert.notEqual(summary.growthRatePct, street);
  });

  it('PDD-like 3Y Historical uses trailing, Forecasting uses Street-to-Street', () => {
    const hist = [
      fy(2022, 4.08, 40),
      fy(2023, 6.5, 70),
      fy(2024, 9.4, 85),
      fy(2025, 10.07, 90),
    ];
    const forward = [
      { year: 2026, metric: 9.98 },
      { year: 2027, metric: 12.08 },
      { year: 2028, metric: 13.88 },
    ];
    const { summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 90,
      windowYears: 3,
      ttmMetric: 10.07,
      forward,
    });
    assert.equal(summary.growthSource, 'trailing');
    const trail = cagrPct(4.08, 10.07, 3);
    assert.ok(trail != null && summary.growthRatePct != null);
    assert.ok(Math.abs(summary.growthRatePct - trail) < 1e-6);
    const box = forecastGrowthFromEstimates(forward);
    const street = cagrPct(9.98, 13.88, 2);
    assert.ok(street != null && box.growthRatePct != null);
    assert.ok(Math.abs(box.growthRatePct - street) < 1e-6);
    assert.ok(box.growthRatePct > 16 && box.growthRatePct < 20, `g=${box.growthRatePct}`);
    assert.equal(box.fairValueRule, 'pe_g');
    assert.notEqual(summary.fairValueRatio, box.fairValueRatio);
  });

  it('does not mix GAAP history into Street estimate CAGR (Adobe)', () => {
    const hist = [
      fy(2020, 10.83, 500),
      fy(2021, 10.02, 650),
      fy(2022, 10.1, 340),
      fy(2023, 11.82, 580),
      fy(2024, 12.36, 440),
      fy(2025, 16.7, 520),
    ];
    const forward = [
      { year: 2026, metric: 24.41 },
      { year: 2027, metric: 27.49 },
      { year: 2028, metric: 31.15 },
    ];
    const { summary } = buildValuationSeries(hist, 'eps', {
      currentPrice: 275.3,
      windowYears: 5,
      ttmMetric: 17.47,
      forward,
    });
    assert.equal(summary.growthSource, 'trailing');
    const box = forecastGrowthFromEstimates(forward);
    const street = cagrPct(24.41, 31.15, 2);
    assert.ok(street != null && box.growthRatePct != null);
    assert.ok(Math.abs(box.growthRatePct - street) < 0.05);
    assert.ok(box.growthRatePct > 12 && box.growthRatePct < 14);
    const fakeGaapToStreet = ((24.41 - 16.7) / 16.7) * 100;
    assert.ok(box.growthRatePct < fakeGaapToStreet / 2);
    assert.ok(summary.growthRatePct != null);
    assert.notEqual(summary.growthRatePct, box.growthRatePct);
  });

  it('PDD-like FCF can borrow Historical EPS trailing, not lumpy FCF', () => {
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
    assert.equal(fcf.summary.growthSource, 'trailing');
    assert.ok(eps.summary.growthRatePct != null && fcf.summary.growthRatePct != null);
    assert.ok(Math.abs(fcf.summary.growthRatePct - eps.summary.growthRatePct) < 1e-9);
    assert.ok(fcf.summary.fairValueRatio != null);
    assert.ok(Math.abs((fcf.summary.fairValue ?? 0) - 8 * fcf.summary.fairValueRatio) < 1e-6);
    assert.ok(
      fcf.summary.fairValue != null && fcf.summary.fairValue > 250 && fcf.summary.fairValue < 320,
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

  /**
   * CRM (Aug 2026): 10Y endpoint CAGR +45.9% → P/E=G 45.9×, FV 771 vs price 252,
   * 3y tail off the chart toward 1800+. 5Y +12.2% stays GDF…P/E=G 15×, FV ≈ price.
   * FG 10Y Adjusted Operating does not explode — the 10Y stub base must not set
   * the orange box. P/E=G itself stays unbounded (AAPL 25.67×).
   */
  function crmLikeOpEpsHistory(): AnnualFundamentalPoint[] {
    const price = 252;
    const rows: Array<[number, number]> = [
      [2011, 0.13],
      [2012, -0.2],
      [2013, -0.4],
      [2014, -0.3],
      [2015, -0.15],
      [2016, 0.419],
      [2017, 0.55],
      [2018, 0.72],
      [2019, 1.1],
      [2020, 2.4],
      [2021, 9.44],
      [2022, 10.59],
      [2023, 11.88],
      [2024, 13.33],
      [2025, 14.96],
      [2026, 16.79],
    ];
    return rows.map(([year, op]) => ({
      ...fy(year, op, price),
      date: `${year}-01-31`,
      operatingEps: op,
    }));
  }

  it('CRM-like 10Y stub CAGR must not produce a 45.9× FV ~3× price or a 3y tail off the chart', () => {
    const hist = crmLikeOpEpsHistory();
    const price = 252;
    const ttm = 16.79;
    const windowed10 = sliceToWindow(hist, 10);
    const raw10 = trailingMetricCagr(windowed10, 'operatingEps', 10);
    assert.ok(raw10 != null && raw10 > 40 && raw10 < 50, `raw 10Y CAGR=${raw10}`);
    const explodedRatio = fairValueRatioFromGrowth(raw10);
    assert.equal(explodedRatio.rule, 'pe_g');
    assert.ok(explodedRatio.ratio != null && explodedRatio.ratio > 40);
    const explodedFv = ttm * explodedRatio.ratio;
    assert.ok(explodedFv > price * 2.8 && explodedFv < price * 3.3, `exploded fv=${explodedFv}`);

    const five = buildValuationSeries(hist, 'operatingEps', {
      currentPrice: price,
      windowYears: 5,
      ttmMetric: ttm,
    });
    assert.ok(five.summary.growthRatePct != null);
    assert.ok(
      five.summary.growthRatePct > 11 && five.summary.growthRatePct < 14,
      `5Y g=${five.summary.growthRatePct}`,
    );
    assert.equal(five.summary.fairValueRule, 'gdf_pe_g');
    assert.equal(five.summary.fairValueRatio, 15);
    assert.ok(five.summary.fairValue != null);
    assert.ok(Math.abs(five.summary.fairValue - ttm * 15) < 1e-6);
    assert.ok(Math.abs(five.summary.fairValue - 251.85) < 0.05);
    assert.ok(Math.abs(five.summary.fairValue - price) < 1);

    const ten = buildValuationSeries(hist, 'operatingEps', {
      currentPrice: price,
      windowYears: 10,
      ttmMetric: ttm,
    });
    const orange10 = orangeBoxTrailingGrowth(windowed10, 'operatingEps', 10);
    assert.ok(orange10.growthPct != null);
    assert.ok(Math.abs(orange10.growthPct - five.summary.growthRatePct!) < 1e-9);
    assert.ok(ten.summary.growthRatePct != null);
    assert.ok(Math.abs(ten.summary.growthRatePct - five.summary.growthRatePct!) < 1e-9);
    assert.equal(ten.summary.fairValueRule, 'gdf_pe_g');
    assert.equal(ten.summary.fairValueRatio, 15);
    assert.ok(ten.summary.fairValue != null);
    assert.ok(Math.abs(ten.summary.fairValue - ttm * 15) < 1e-6);
    assert.ok(ten.summary.fairValue < price * 1.2, `10Y fv=${ten.summary.fairValue} must not be ~3× price`);
    assert.ok(ten.summary.fairValueRatio < 20);
    assert.notEqual(ten.summary.fairValueRatio, explodedRatio.ratio);

    const chart = buildFairValueChartSeries({
      series: ten.series,
      summary: ten.summary,
      metric: 'operatingEps',
      estimates: [
        { year: 2027, date: '2027-01-31' },
        { year: 2028, date: '2028-01-31' },
        { year: 2029, date: '2029-01-31' },
      ],
      asOfIso: '2026-08-16',
    });
    const tail = chart.filter((p) => p.forecast).map((p) => p.fairValue ?? 0);
    assert.ok(tail.length >= 3, `forecast count=${tail.length}`);
    const maxTail = Math.max(...tail);
    const explodedTail = ttm * Math.pow(1 + raw10 / 100, 3) * explodedRatio.ratio!;
    assert.ok(explodedTail > 1800, `pre-fix 3y tail would be ${explodedTail}`);
    assert.ok(maxTail < 1000, `3y tail ${maxTail} must stay on a FG-like 0–1000 axis`);
    assert.ok(maxTail < price * 2, `3y tail ${maxTail} must not shoot off vs price ${price}`);
    assert.ok(maxTail > five.summary.fairValue!, '3y overlay still slopes up');

    for (const w of [8, 15, null] as const) {
      const long = buildValuationSeries(hist, 'operatingEps', {
        currentPrice: price,
        windowYears: w,
        ttmMetric: ttm,
      });
      assert.equal(long.summary.fairValueRule, 'gdf_pe_g', `window=${w}`);
      assert.equal(long.summary.fairValueRatio, 15, `window=${w}`);
      assert.ok(
        long.summary.fairValue != null && Math.abs(long.summary.fairValue - ttm * 15) < 1e-6,
        `window=${w} fv=${long.summary.fairValue}`,
      );
    }
  });

  it('still uses unbounded P/E=G on a long window when the 5Y path is also Lynch', () => {
    const hist = Array.from({ length: 11 }, (_, i) => {
      const year = 2015 + i;
      const eps = 2 * Math.pow(1.2, i);
      return { ...fy(year, eps, eps * 22), operatingEps: eps };
    });
    const last = hist[hist.length - 1]!;
    const ten = buildValuationSeries(hist, 'operatingEps', {
      currentPrice: last.price,
      windowYears: 10,
      ttmMetric: last.operatingEps,
    });
    const five = buildValuationSeries(hist, 'operatingEps', {
      currentPrice: last.price,
      windowYears: 5,
      ttmMetric: last.operatingEps,
    });
    assert.ok(five.summary.growthRatePct != null && five.summary.growthRatePct >= 15);
    assert.equal(five.summary.fairValueRule, 'pe_g');
    assert.equal(ten.summary.fairValueRule, 'pe_g');
    assert.ok(ten.summary.fairValueRatio != null && ten.summary.fairValueRatio >= 15);
    assert.ok(Math.abs((ten.summary.growthRatePct ?? 0) - 20) < 0.05);
    assert.ok(
      ten.summary.fairValueRatio != null &&
        Math.abs(ten.summary.fairValueRatio - Math.round((ten.summary.growthRatePct as number) * 100) / 100) <
          1e-9,
    );
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

  it('uses the first bar for MAX when FMP history is as old as the series', () => {
    const range = valuationChartRange({
      firstBarDate: firstBar,
      lastBarDate: lastBar,
      windowYears: null,
      firstHistoricalDate: '2015-12-31',
      historyStartDate: '2015-12-31',
      firstForecastDate: lastForecast,
    });
    assert.equal(range.from, firstBar);
    assert.equal(range.to, lastForecast);
  });

  it('clamps MAX and 15Y to the first FMP FY year on a short-history IPO', () => {
    const ipoFirstFy = '2024-12-31';
    const paddedFirstBar = '2006-01-06';
    const maxRange = valuationChartRange({
      firstBarDate: paddedFirstBar,
      lastBarDate: lastBar,
      windowYears: null,
      firstHistoricalDate: ipoFirstFy,
      historyStartDate: ipoFirstFy,
      firstForecastDate: lastForecast,
      lastForecastDate: lastForecast,
    });
    assert.equal(maxRange.from, '2024-01-01');
    assert.ok(maxRange.from > '2010-01-01', `MAX from=${maxRange.from} must not be a 20Y pad`);
    assert.equal(maxRange.to, lastForecast);

    const y15 = valuationChartRange({
      firstBarDate: paddedFirstBar,
      lastBarDate: lastBar,
      windowYears: 15,
      firstHistoricalDate: ipoFirstFy,
      historyStartDate: ipoFirstFy,
      firstForecastDate: lastForecast,
      lastForecastDate: lastForecast,
    });
    assert.equal(y15.from, '2024-01-01');
  });

  it('does not pull a 5Y window back to the first FMP year on long history', () => {
    const range = valuationChartRange({
      firstBarDate: firstBar,
      lastBarDate: lastBar,
      windowYears: 5,
      firstHistoricalDate: firstFy5,
      historyStartDate: '2005-09-24',
      firstForecastDate: lastForecast,
      lastForecastDate: lastForecast,
    });
    assert.ok(range.from <= firstFy5, `from=${range.from} should include ${firstFy5}`);
    assert.ok(range.from > '2018-01-01', `from=${range.from} should stay a 5Y window, not MAX`);
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

  it('maps a short-history IPO onto recent bars, not a 20Y padded axis', () => {
    const weekMs = 7 * 86_400_000;
    const start = Date.parse('2006-01-06T00:00:00Z');
    const end = Date.parse('2028-12-29T00:00:00Z');
    const timesMs: number[] = [];
    for (let t = start; t <= end; t += weekMs) timesMs.push(t);
    const range = valuationChartRange({
      firstBarDate: '2006-01-06',
      lastBarDate: lastBar,
      windowYears: null,
      firstHistoricalDate: '2024-12-31',
      historyStartDate: '2024-12-31',
      firstForecastDate: lastForecast,
      lastForecastDate: lastForecast,
    });
    const logical = valuationChartLogicalRange(timesMs, range);
    const fromIso = new Date(timesMs[logical.fromIdx]!).toISOString().slice(0, 10);
    assert.ok(fromIso >= '2023-12-01', `IPO fromIdx date ${fromIso} must not be 2006`);
    assert.ok(fromIso <= '2024-01-10', `IPO fromIdx date ${fromIso} should be the first FMP year`);
    const long = valuationChartRange({
      firstBarDate: firstBar,
      lastBarDate: lastBar,
      windowYears: 10,
      firstHistoricalDate: '2015-12-31',
      historyStartDate: '2005-09-24',
      firstForecastDate: lastForecast,
      lastForecastDate: lastForecast,
    });
    assert.ok(long.from < '2017-01-01', `long-history 10Y from=${long.from}`);
    assert.ok(long.from > '2014-01-01', `long-history 10Y should not jump to 2005 FMP start`);
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

  it('leaves the first Street estimate % Chg blank so GAAP is not mixed in', () => {
    const chg = estimateChainChgPct([
      { year: 2026, eps: 24.41 },
      { year: 2027, eps: 27.49 },
      { year: 2028, eps: 31.15 },
    ]);
    assert.equal(chg[0], null);
    assert.ok(chg[1] != null && Math.abs(chg[1] - ((27.49 / 24.41 - 1) * 100)) < 0.01);
    assert.ok(chg[2] != null && chg[2] > 12 && chg[2] < 15);
  });
});

describe('FMP field mapping', () => {
  it('reads FMP ownersEarningsPerShare (plural) before the older ownerEarnings key', () => {
    assert.equal(
      ownerEarningsPerShareFromRow({ ownersEarningsPerShare: 2.3, ownerEarnings: 99 }),
      2.3,
    );
    assert.equal(ownerEarningsPerShareFromRow({ ownerEarningsPerShare: 2.3 }), 2.3);
    assert.equal(
      ownerEarningsPerShareFromRow({ ownersEarnings: 230, averageSharesOutstanding: 100 }),
      2.3,
    );
    assert.equal(
      ownerEarningsPerShareFromRow({ ownerEarnings: 230, shares: 100 }),
      2.3,
    );
    assert.equal(ownerEarningsPerShareFromRow({ ownerEarnings: 230 }), null);
  });

  it('uses FY-end close, not December, for a September fiscal year', () => {
    const close = closeOnOrBefore(
      [
        { date: '2025-09-26', close: 227.5 },
        { date: '2025-12-31', close: 250 },
      ],
      '2025-09-27',
    );
    assert.equal(close, 227.5);
  });

  it('counts consecutive fiscal dividend increases from the annual series', () => {
    const streak = dividendStreak([
      { year: 2020, dividend: 0.8 },
      { year: 2021, dividend: 0.85 },
      { year: 2022, dividend: 0.9 },
      { year: 2023, dividend: 0.94 },
      { year: 2024, dividend: 0.98 },
      { year: 2025, dividend: 1.02 },
    ]);
    assert.equal(streak.consecPaid, 6);
    assert.equal(streak.consecIncreases, 5);
    assert.ok(streak.avgGrowthPct != null && streak.avgGrowthPct > 4 && streak.avgGrowthPct < 6);
  });

  it('keeps the default Value / Summary window at 5Y so cards match the chart', () => {
    assert.equal(DEFAULT_VALUATION_WINDOW, 5);
  });

  /**
   * ALS-like: GAAP/Street EPS grew fast (Lynch, card used to say undervalued) while
   * trailing Op. EPS declined (15×, tiny FV, 2000%+ overvalued on the chart).
   * Results / Value cards must follow the Summary default, not leftover `eps`.
   */
  function alsLikeSplitHistory(): AnnualFundamentalPoint[] {
    const rows: Array<[number, number, number, number]> = [
      [2020, 0.56, 0.52, 25],
      [2021, 0.9, 0.43, 30],
      [2022, 1.45, 0.36, 38],
      [2023, 2.34, 0.3, 48],
      [2024, 3.77, 0.25, 58],
      [2025, 6.13, 0.207, 67.51],
    ];
    return rows.map(([year, eps, operatingEps, price]) => ({
      ...fy(year, eps, price),
      operatingEps,
      gaapEps: eps,
    }));
  }

  it('cards use 5Y Op. EPS trailing so GAAP/Street growth cannot flip undervalued vs the chart', () => {
    const hist = alsLikeSplitHistory();
    const price = 67.51;
    const ttmOp = 0.207;
    const ttmGaap = 6.13;
    const chart = buildValuationSeries(hist, DEFAULT_CHART_VALUATION_METRIC, {
      currentPrice: price,
      windowYears: DEFAULT_VALUATION_WINDOW,
      forward: [],
      ttmMetric: ttmOp,
    });
    const card = buildCardValuation(hist, { currentPrice: price, ttmOperatingEps: ttmOp });
    const leftoverGaap = buildValuationSeries(hist, 'eps', {
      currentPrice: price,
      windowYears: DEFAULT_VALUATION_WINDOW,
      forward: [
        { year: 2026, metric: 8 },
        { year: 2027, metric: 13 },
      ],
      ttmMetric: ttmGaap,
    });
    const street = forecastGrowthFromEstimates([
      { year: 2026, eps: 8 },
      { year: 2027, eps: 13 },
    ]);

    assert.equal(card.summary.metric, 'operatingEps');
    assert.equal(card.summary.windowYears, 5);
    assert.equal(card.summary.growthSource, 'trailing');
    assert.equal(card.summary.premiumPct, chart.summary.premiumPct);
    assert.equal(card.summary.growthRatePct, chart.summary.growthRatePct);
    assert.equal(card.summary.fairValue, chart.summary.fairValue);
    assert.ok(chart.summary.growthRatePct != null && chart.summary.growthRatePct < 0);
    assert.equal(chart.summary.fairValueRatio, 15);
    assert.ok(chart.summary.premiumPct != null && chart.summary.premiumPct > 1000);
    assert.ok(leftoverGaap.summary.growthRatePct != null && leftoverGaap.summary.growthRatePct > 50);
    assert.ok(leftoverGaap.summary.premiumPct != null && leftoverGaap.summary.premiumPct < 0);
    assert.ok(street.growthRatePct != null && street.growthRatePct > 50);
    assert.notEqual(Math.sign(chart.summary.premiumPct), Math.sign(leftoverGaap.summary.premiumPct));
    assert.ok(Math.abs(card.summary.growthRatePct - street.growthRatePct) > 40);
  });

  it('defaults the Summary chart chip to Op. EPS and drops Sales / Owner / GAAP EPS', () => {
    assert.equal(DEFAULT_CHART_VALUATION_METRIC, 'operatingEps');
    assert.deepEqual([...CHART_VALUATION_METRICS], ['operatingEps', 'fcf']);
    assert.equal(coerceChartValuationMetric('eps'), 'operatingEps');
    assert.equal(coerceChartValuationMetric('revenue'), 'operatingEps');
    assert.equal(coerceChartValuationMetric('ownerEarnings'), 'operatingEps');
    assert.equal(coerceChartValuationMetric('fcf'), 'fcf');
    assert.equal(coerceValuationMetric('eps'), 'eps');
    assert.equal(coerceValuationMetric('operatingEps'), 'operatingEps');
    assert.equal(coerceValuationMetric('sales'), 'operatingEps');
  });

  it('sums AAPL-like adjDividend into September fiscal years, not calendar 2025', () => {
    const byFy = sumDividendsByFiscalYear(
      [
        { date: '2024-11-14', adjDividend: 0.25 },
        { date: '2025-02-13', adjDividend: 0.25 },
        { date: '2025-05-15', adjDividend: 0.26 },
        { date: '2025-08-14', adjDividend: 0.26 },
        { date: '2025-11-13', adjDividend: 0.26 },
      ],
      9,
    );
    assert.ok(Math.abs((byFy.get(2025) ?? 0) - 1.02) < 1e-9);
    assert.ok(Math.abs((byFy.get(2026) ?? 0) - 0.26) < 1e-9);
    const calendar2025 = 0.25 + 0.26 + 0.26 + 0.26;
    assert.notEqual(byFy.get(2025), calendar2025);
  });
});

/**
 * Live FAST Graphs (user login, 26 Aug 2026) vs formulas on captured FMP numbers.
 * Do not invent other FG figures. Normal P/E needs FY-end prices — not in this fixture.
 */
describe('FAST Graphs live 26 Aug 2026 (AAPL / MSFT)', () => {
  const FG = {
    aapl: {
      price: 313.45,
      histGrowth: 25.67,
      histFv: 25.67,
      histNormalPe: 23.27,
      fy25: 7.46,
      fcstGrowth: 10.15,
      fcstFv: 15,
      street: [
        { year: 2026, metric: 8.85 },
        { year: 2027, metric: 9.54 },
        { year: 2028, metric: 10.68 },
      ],
      histTable: [
        { year: 2006, eps: 0.08 },
        { year: 2025, eps: 7.46 },
      ],
    },
    msft: {
      price: 496.37,
      histGrowth: 14.47,
      histFv: 15,
      histNormalPe: 22,
      fy25: 13.64,
      fcstGrowth: 18.23,
      fcstFv: 18.23,
      street: [
        { year: 2027, metric: 19.58 },
        { year: 2028, metric: 23.38 },
        { year: 2029, metric: 28.4 },
      ],
    },
  };
  const FMP = {
    aaplGaap: {
      2019: 2.97,
      2020: 3.28,
      2021: 5.61,
      2022: 6.11,
      2023: 6.13,
      2024: 6.08,
      2025: 7.46,
    },
    aaplNopatFy25: 8.87,
    aaplStreet: [
      { year: 2026, metric: 8.83 },
      { year: 2027, metric: 9.52 },
      { year: 2028, metric: 10.62 },
    ],
    msftGaap: {
      2022: 9.65,
      2023: 9.68,
      2024: 11.8,
      2025: 13.64,
      2026: 17.95,
    },
    msftNopatFy25: 17.22,
    msftStreet: [
      { year: 2027, metric: 19.72 },
      { year: 2028, metric: 23.48 },
      { year: 2029, metric: 28.6 },
    ],
  };

  it('keeps FMP GAAP FY25 as the closest proxy — it matches FG operating EPS', () => {
    assert.equal(FMP.aaplGaap[2025], FG.aapl.fy25);
    assert.equal(FMP.msftGaap[2025], FG.msft.fy25);
    assert.notEqual(FMP.aaplNopatFy25, FG.aapl.fy25);
    assert.notEqual(FMP.msftNopatFy25, FG.msft.fy25);
    assert.ok(FMP.aaplNopatFy25 > FG.aapl.fy25);
    assert.ok(FMP.msftNopatFy25 > FG.msft.fy25);
  });

  it('documents year-level GAAP vs FG gaps without inventing a patched series', () => {
    assert.equal(FMP.aaplGaap[2024], 6.08);
    assert.equal(FG.aapl.fy25, 7.46);
    assert.notEqual(FMP.aaplGaap[2024], 6.75);
    assert.equal(FMP.msftGaap[2022], 9.65);
    assert.notEqual(FMP.msftGaap[2022], 9.21);
    assert.equal(FMP.msftGaap[2023], 9.68);
    assert.notEqual(FMP.msftGaap[2023], 9.81);
    assert.equal(FMP.msftGaap[2026], 17.95);
    assert.notEqual(FMP.msftGaap[2026], 17.28);
    assert.equal(FMP.aaplGaap[2019], 2.97);
    assert.equal(FMP.aaplGaap[2020], 3.28);
    assert.equal(FMP.aaplGaap[2023], 6.13);
  });

  it('applies the Historical / Forecasting rule split to FG Graph Key growth', () => {
    const aaplHist = fairValueRatioFromGrowth(FG.aapl.histGrowth);
    assert.equal(aaplHist.rule, 'pe_g');
    assert.equal(aaplHist.ratio, FG.aapl.histFv);

    const aaplFcst = fairValueRatioFromGrowth(FG.aapl.fcstGrowth, {
      spanYears: 2,
      windowYears: 1,
    });
    assert.equal(aaplFcst.rule, 'gdf_pe_g');
    assert.equal(aaplFcst.ratio, FG.aapl.fcstFv);

    const msftHist = fairValueRatioFromGrowth(FG.msft.histGrowth);
    assert.equal(msftHist.rule, 'gdf_pe_g');
    assert.equal(msftHist.ratio, FG.msft.histFv);

    const msftFcst = fairValueRatioFromGrowth(FG.msft.fcstGrowth, {
      spanYears: 2,
      windowYears: 1,
    });
    assert.equal(msftFcst.rule, 'pe_g');
    assert.equal(msftFcst.ratio, FG.msft.fcstFv);
  });

  it('AAPL Historical endpoint CAGR on the FG table is P/E=G but not FG 25.67% (FG is fitted)', () => {
    const g = cagrPct(FG.aapl.histTable[0]!.eps, FG.aapl.histTable[1]!.eps, 19);
    assert.ok(g != null);
    const box = fairValueRatioFromGrowth(g);
    assert.equal(box.rule, 'pe_g');
    assert.ok(g > 26 && g < 28, `endpoint CAGR=${g}`);
    assert.ok(Math.abs(g - FG.aapl.histGrowth) > 0.5);
    assert.ok(box.ratio != null && Math.abs(box.ratio - Math.round(g * 100) / 100) < 1e-9);
  });

  it('AAPL Forecasting Street-to-Street from FMP estimates flips to 15× like FG', () => {
    const box = forecastGrowthFromEstimates(FMP.aaplStreet);
    assert.ok(box.growthRatePct != null);
    assert.ok(box.growthRatePct > 9 && box.growthRatePct < 11, `g=${box.growthRatePct}`);
    assert.equal(box.fairValueRule, 'gdf_pe_g');
    assert.equal(box.fairValueRatio, FG.aapl.fcstFv);
    const fgPrinted = forecastGrowthFromEstimates(FG.aapl.street);
    assert.equal(fgPrinted.fairValueRatio, 15);
    assert.ok(fgPrinted.growthRatePct != null && Math.abs(fgPrinted.growthRatePct - 10.15) > 0.1);
  });

  it('MSFT Forecasting FMP Street-to-Street stays P/E=G and is near FG 18.23×', () => {
    const box = forecastGrowthFromEstimates(FMP.msftStreet);
    assert.equal(box.fairValueRule, 'pe_g');
    assert.ok(box.growthRatePct != null && box.growthRatePct > 19 && box.growthRatePct < 22);
    assert.ok(
      box.fairValueRatio != null &&
        Math.abs(box.fairValueRatio - Math.round(box.growthRatePct * 100) / 100) < 1e-9,
    );
    assert.ok(Math.abs((box.fairValueRatio ?? 0) - FG.msft.fcstFv) > 1);
    const fgPrinted = forecastGrowthFromEstimates(FG.msft.street);
    assert.equal(fgPrinted.fairValueRule, 'pe_g');
    assert.ok(fgPrinted.growthRatePct != null && fgPrinted.growthRatePct > 18);
    // FG 18.23% is closer to last-actual→last-est (17.28→28.40 / 3y ≈ 18%). We stay Street-to-Street.
  });

  it('does not compute Normal P/E without FY-end prices (FG 23.27× / 22.00×)', () => {
    const aapl = buildValuationSeries(
      [
        fy(2024, FMP.aaplGaap[2024], 0),
        fy(2025, FMP.aaplGaap[2025], 0),
      ],
      'eps',
      { currentPrice: FG.aapl.price, windowYears: null },
    );
    assert.equal(aapl.summary.latestMetric, FG.aapl.fy25);
    assert.equal(aapl.summary.normalMultipleSource, 'fallback');
    assert.notEqual(aapl.summary.normalMultiple, FG.aapl.histNormalPe);
    assert.notEqual(aapl.summary.normalMultiple, FG.msft.histNormalPe);
  });
});

/**
 * NOK NYSE ADR: EUR GAAP vs USD listing. Street consensus is already USD.
 * Without (A)+(B)+(C) this was EPS×FV 1.0× and FV $0.14 (“7073% overvalued”).
 */
describe('NOK-like ADR / foreign filing (EUR GAAP + USD Street)', () => {
  const fx = 1.165;
  const price = 10.5;
  const gaapEur: Record<number, number> = {
    2021: 0.2,
    2022: 0.18,
    2023: 0.12,
    2024: 0.23,
    2025: 0.11,
  };
  const streetUsd: Record<number, number> = {
    2021: 0.42,
    2022: 0.48,
    2023: 0.3,
    2024: 0.33,
    2025: 0.26,
  };
  const scale = { reportedCurrency: 'EUR', listingCurrency: 'USD' };

  function nokHistory(): AnnualFundamentalPoint[] {
    return [2021, 2022, 2023, 2024, 2025].map((year) => {
      const gaapUsd = gaapEur[year]! * fx;
      return {
        ...fy(year, gaapUsd, price),
        gaapEps: gaapUsd,
      };
    });
  }

  it('does not overlay Street onto AAPL-like same-currency GAAP', () => {
    const hist: AnnualFundamentalPoint[] = [
      { ...fy(2025, 7.46, 313), gaapEps: 7.46 },
    ];
    const out = applyStreetConsensusHistory(
      hist,
      [{ year: 2025, eps: 8.85 }],
      { reportedCurrency: 'USD', listingCurrency: 'USD' },
    );
    assert.equal(out[0]?.eps, 7.46);
    assert.equal(out[0]?.gaapEps, 7.46);
    assert.equal(pickMetric(out[0]!, 'eps'), 7.46);
  });

  it('overlays USD Street on default eps and keeps FX-scaled GAAP on gaapEps', () => {
    const overlaid = applyStreetConsensusHistory(
      nokHistory(),
      Object.entries(streetUsd).map(([year, eps]) => ({ year: Number(year), eps })),
      scale,
    );
    const fy25 = overlaid.find((p) => p.year === 2025);
    assert.equal(fy25?.eps, 0.26);
    assert.ok(fy25?.gaapEps != null && Math.abs(fy25.gaapEps - 0.11 * fx) < 1e-9);
    assert.equal(pickMetric(fy25!, 'eps'), 0.26);
  });

  it('falls back to FX-scaled GAAP when a year has no consensus', () => {
    const overlaid = applyStreetConsensusHistory(
      nokHistory(),
      [
        { year: 2023, eps: 0.3 },
        { year: 2024, eps: 0.33 },
        { year: 2025, eps: 0.26 },
      ],
      scale,
    );
    const fy21 = overlaid.find((p) => p.year === 2021);
    assert.ok(fy21?.eps != null && Math.abs(fy21.eps - 0.2 * fx) < 1e-9);
    assert.equal(overlaid.find((p) => p.year === 2025)?.eps, 0.26);
  });

  it('produces FG-like FV: 15× on Street ~0.26, several dollars, not $0.14', () => {
    const overlaid = applyStreetConsensusHistory(
      nokHistory(),
      Object.entries(streetUsd).map(([year, eps]) => ({ year: Number(year), eps })),
      scale,
    );
    const ttm = defaultEpsTtm(overlaid, 0.11 * fx, scale);
    assert.ok(ttm != null && ttm >= 0.26 && ttm <= 0.35, `ttm=${ttm}`);
    const { summary } = buildValuationSeries(overlaid, 'eps', {
      currentPrice: price,
      windowYears: 5,
      ttmMetric: ttm,
    });
    assert.ok(summary.growthRatePct != null && summary.growthRatePct < 0, `g=${summary.growthRatePct}`);
    assert.equal(summary.fairValueRule, 'gdf_pe_g');
    assert.equal(summary.fairValueRatio, 15);
    assert.ok(
      summary.latestMetric != null &&
        summary.latestMetric >= 0.26 &&
        summary.latestMetric <= 0.35,
      `eps=${summary.latestMetric}`,
    );
    assert.ok(summary.fairValue != null);
    assert.ok(summary.fairValue > 3.5 && summary.fairValue < 8, `fv=${summary.fairValue}`);
    assert.ok(Math.abs(summary.fairValue - 0.26 * 15) < 1e-6);
    assert.ok(summary.premiumPct != null && summary.premiumPct < 400);
    assert.ok(Math.abs(summary.fairValue - 0.14) > 1);

    const broken = buildValuationSeries(nokHistory(), 'eps', {
      currentPrice: price,
      windowYears: 5,
      ttmMetric: 0.11 * fx,
    });
    assert.ok((broken.summary.fairValue ?? 0) < 3, `gaap-only fv=${broken.summary.fairValue}`);
  });
});

/** Distinct per-share series so leftover EPS FV is obvious. */
function multiMetricHistory(): AnnualFundamentalPoint[] {
  const rows: Array<[number, number, number, number, number]> = [
    [2020, 2.0, 2.4, 1.5, 40],
    [2021, 2.4, 2.9, 1.8, 50],
    [2022, 3.0, 3.6, 2.2, 62],
    [2023, 3.6, 4.4, 2.8, 75],
    [2024, 4.3, 5.2, 3.4, 88],
    [2025, 5.0, 6.0, 4.0, 100],
  ];
  return rows.map(([year, eps, operatingEps, fcf, price]) => ({
    ...fy(year, eps, price),
    operatingEps,
    fcfPerShare: fcf,
  }));
}

describe('buildFairValueChartSeries per Summary metric', () => {
  const asOf = '2026-08-16';
  const street = [
    { year: 2026, date: '2026-12-31', eps: 8 },
    { year: 2027, date: '2027-12-31', eps: 9 },
    { year: 2028, date: '2028-12-31', eps: 10 },
  ];

  function chartFor(
    metric: 'eps' | 'operatingEps' | 'fcf',
    opts: { ttmMetric?: number | null; growthRatePct?: number | null; growthSource?: 'trailing' | 'forward' } = {},
  ) {
    const hist = multiMetricHistory();
    const valuation = buildValuationSeries(hist, metric, {
      currentPrice: 100,
      windowYears: 5,
      ttmMetric: opts.ttmMetric,
      growthRatePct: opts.growthRatePct,
      growthSource: opts.growthSource,
    });
    const series = buildFairValueChartSeries({
      series: valuation.series,
      summary: valuation.summary,
      metric,
      estimates: street,
      asOfIso: asOf,
    });
    return { valuation, series };
  }

  it('Op. EPS plots operating FV, current value, and 3y — not leftover GAAP EPS', () => {
    const eps = chartFor('eps', { ttmMetric: 5 });
    const op = chartFor('operatingEps', { ttmMetric: 6 });
    const lastFy = op.series.find((p) => p.year === 2025 && !p.forecast && !p.estimated);
    assert.ok(op.valuation.summary.fairValueRatio);
    assert.equal(lastFy?.fairValue, 6.0 * op.valuation.summary.fairValueRatio);
    assert.notEqual(lastFy?.fairValue, 5 * (eps.valuation.summary.fairValueRatio as number));
    assert.equal(op.valuation.summary.fairValue, 6 * (op.valuation.summary.fairValueRatio as number));
    const fwd = op.series.filter((p) => p.forecast);
    assert.ok(fwd.length >= 3, `op.eps forecast count=${fwd.length}`);
    const y2026 = fwd.find((p) => p.date === '2026-12-31');
    const g = op.valuation.summary.growthRatePct as number;
    const expectedMetric = 6 * Math.pow(1 + g / 100, 1);
    assert.ok(y2026?.metric != null);
    assert.ok(Math.abs(y2026.metric - expectedMetric) < 1e-6);
    assert.notEqual(y2026.metric, 8);
    assert.equal(y2026.fairValue, expectedMetric * (op.valuation.summary.fairValueRatio as number));
    assert.ok(y2026.normalValue != null);
  });

  it('FCF/sh plots FCF FV and 3y in FCF dollars, not EPS × ratio', () => {
    const eps = chartFor('eps', { ttmMetric: 5 });
    const fcf = chartFor('fcf', { ttmMetric: 4 });
    const lastFy = fcf.series.find((p) => p.year === 2025 && !p.forecast && !p.estimated);
    assert.ok(fcf.valuation.summary.fairValueRatio);
    assert.equal(lastFy?.fairValue, 4.0 * fcf.valuation.summary.fairValueRatio);
    assert.notEqual(lastFy?.fairValue, 5 * (eps.valuation.summary.fairValueRatio as number));
    assert.equal(fcf.valuation.summary.fairValue, 4 * (fcf.valuation.summary.fairValueRatio as number));
    const fwd = fcf.series.filter((p) => p.forecast);
    assert.ok(fwd.length >= 3, `fcf forecast count=${fwd.length}`);
    const y2026 = fwd.find((p) => p.date === '2026-12-31');
    const g = fcf.valuation.summary.growthRatePct as number;
    const expectedMetric = 4 * Math.pow(1 + g / 100, 1);
    assert.ok(y2026?.metric != null);
    assert.ok(Math.abs(y2026.metric - expectedMetric) < 1e-6);
    assert.notEqual(y2026.metric, 8);
    assert.equal(y2026.fairValue, expectedMetric * (fcf.valuation.summary.fairValueRatio as number));
  });

  it('Forecasting tab uses Street growth and still draws a 3y overlay', () => {
    const trailing = chartFor('operatingEps', { ttmMetric: 6 });
    const box = forecastGrowthFromEstimates(street);
    assert.ok(box.growthRatePct != null);
    const forecast = chartFor('operatingEps', {
      ttmMetric: 6,
      growthRatePct: box.growthRatePct,
      growthSource: 'forward',
    });
    assert.equal(forecast.valuation.summary.growthSource, 'forward');
    assert.equal(forecast.valuation.summary.growthRatePct, box.growthRatePct);
    assert.notEqual(
      forecast.valuation.summary.fairValueRatio,
      trailing.valuation.summary.fairValueRatio,
    );
    const fwd = forecast.series.filter((p) => p.forecast);
    assert.ok(fwd.length >= 3, `forecast-tab overlay count=${fwd.length}`);
    assert.ok(fwd.every((p) => p.fairValue != null && p.fairValue > 0));
    const y2026 = fwd.find((p) => p.date === '2026-12-31');
    assert.ok(y2026?.metric != null && y2026.fairValue != null);
    assert.ok(
      Math.abs(y2026.fairValue / y2026.metric - (forecast.valuation.summary.fairValueRatio as number)) <
        1e-6,
    );
    assert.notEqual(y2026.metric, 8);
  });

  it('internal EPS overlay still uses Street estimate levels, not (1+g) projection', () => {
    const { valuation, series } = chartFor('eps', { ttmMetric: 5 });
    const y2026 = series.find((p) => p.forecast && p.date === '2026-12-31');
    assert.ok(y2026);
    assert.equal(y2026.metric, 8);
    assert.equal(y2026.fairValue, 8 * (valuation.summary.fairValueRatio as number));
    const projected = 5 * Math.pow(1 + (valuation.summary.growthRatePct as number) / 100, 1);
    assert.ok(Math.abs((y2026.metric as number) - projected) > 0.2);
  });
});

