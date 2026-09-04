/** S&P period return and alpha vs History capital-pool ROI. */

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

export const BENCHMARK_SYMBOL = 'SPY';

export type DailyClose = { date: string; close: number };

export type BenchmarkReturn = {
  symbol: string;
  returnPct: number;
};

/**
 * Close-to-close percent over [from, to]. Start is the last session on or
 * before `from` (so YTD Jan 1 uses the prior year-end close); if none, the
 * first session on or after `from`. End is the last session on or before `to`.
 */
export function periodReturnPct(bars: DailyClose[], from: string, to: string): number | null {
  if (!bars.length || !from || !to || from > to) return null;

  let start: DailyClose | null = null;
  let end: DailyClose | null = null;
  let firstInWindow: DailyClose | null = null;
  for (const bar of bars) {
    if (!bar.date || !(bar.close > 0)) continue;
    if (bar.date <= from) start = bar;
    if (bar.date <= to) end = bar;
    if (!firstInWindow && bar.date >= from && bar.date <= to) firstInWindow = bar;
  }
  if (!start) start = firstInWindow;
  if (!start || !end || end.date < start.date) return null;
  return round2((end.close / start.close - 1) * 100);
}

/** Strategy ROI minus the S&P period return; null if either side is missing. */
export function alphaPct(roiPct: number | null, benchmarkPct: number | null): number | null {
  if (roiPct == null || benchmarkPct == null) return null;
  if (!Number.isFinite(roiPct) || !Number.isFinite(benchmarkPct)) return null;
  return round2(roiPct - benchmarkPct);
}

/** Attach S&P + both alphas. A missing/failed benchmark leaves every field null. */
export function withBenchmark(
  roiOnPeakPct: number | null,
  roiOnAvgPct: number | null,
  benchmark: BenchmarkReturn | null,
): {
  benchmarkSymbol: string | null;
  benchmarkReturnPct: number | null;
  alphaVsBenchmarkPct: number | null;
  alphaOnAvgPct: number | null;
} {
  const returnPct = benchmark?.returnPct ?? null;
  return {
    benchmarkSymbol: benchmark?.symbol ?? null,
    benchmarkReturnPct: returnPct,
    alphaVsBenchmarkPct: alphaPct(roiOnPeakPct, returnPct),
    alphaOnAvgPct: alphaPct(roiOnAvgPct, returnPct),
  };
}
