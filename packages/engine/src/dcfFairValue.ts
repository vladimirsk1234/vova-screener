/**
 * Roll-forward DCF fair value at the end of each forecast year.
 * Pure math — Custom DCF cash flows come from FMP via the API layer.
 */

export type DcfYearInput = {
  year: number;
  ufcf: number | null;
};

export type DcfFairValueInput = {
  years: DcfYearInput[];
  wacc: number | null;
  terminalValue: number | null;
  netDebt: number | null;
  dilutedShares: number | null;
};

export type DcfYearFairValue = {
  year: number;
  fairValuePerShare: number | null;
};

export type DcfChartSeriesInput = DcfFairValueInput & {
  asOf: string;
  /** FMP headline — used only when the local t=0 model cannot run. */
  fmpEquityValuePerShare?: number | null;
  /** Last complete FY-end (YYYY-MM-DD) so Sep-FY names are not plotted on 12-31. */
  lastHistDate?: string | null;
};

export type DcfChartPoint = {
  date: string;
  /** FY-end year for chart markers; omitted on the today point. */
  year?: number;
  fairValue: number;
};

function fyEndIso(year: number, lastHistDate?: string | null): string {
  const md = lastHistDate && /^\d{4}-(\d{2}-\d{2})/.exec(lastHistDate.slice(0, 10));
  return `${year}-${md?.[1] ?? '12-31'}`;
}

function finite(n: unknown): n is number {
  return typeof n === 'number' && Number.isFinite(n);
}

/**
 * Expected equity value per share at the end of year t (after that year's UFCF).
 *
 * EV_t = Σ_{i=t+1..N} UFCF_i / (1+WACC)^{i-t} + TV / (1+WACC)^{N-t}
 * FV_t = (EV_t − NetDebt) / DilutedShares
 *
 * The last year is undiscounted terminal value minus net debt. Today's
 * equity value (t = 0) is expectedDcfFairValueToday — buildDcfChartSeries
 * uses that for the asOf point so the path stays on one model.
 */
export function expectedDcfFairValueByYear(input: DcfFairValueInput): DcfYearFairValue[] {
  const { years, wacc, terminalValue, netDebt, dilutedShares } = input;
  const n = years.length;
  const ready =
    finite(wacc) &&
    wacc > -0.99 &&
    finite(terminalValue) &&
    finite(netDebt) &&
    finite(dilutedShares) &&
    dilutedShares > 0;

  return years.map((row, t) => {
    if (!ready) return { year: row.year, fairValuePerShare: null };
    let ev = 0;
    for (let i = t + 1; i < n; i++) {
      const ufcf = years[i]!.ufcf;
      if (!finite(ufcf)) return { year: row.year, fairValuePerShare: null };
      ev += ufcf / Math.pow(1 + wacc, i - t);
    }
    ev += terminalValue / Math.pow(1 + wacc, n - 1 - t);
    return { year: row.year, fairValuePerShare: (ev - netDebt) / dilutedShares };
  });
}

export function expectedDcfFairValueToday(input: DcfFairValueInput): number | null {
  const { years, wacc, terminalValue, netDebt, dilutedShares } = input;
  if (
    !finite(wacc) ||
    wacc <= -0.99 ||
    !finite(terminalValue) ||
    !finite(netDebt) ||
    !finite(dilutedShares) ||
    dilutedShares <= 0
  ) {
    return null;
  }
  let ev = 0;
  for (let i = 0; i < years.length; i++) {
    const ufcf = years[i]!.ufcf;
    if (!finite(ufcf)) return null;
    ev += ufcf / Math.pow(1 + wacc, i + 1);
  }
  ev += terminalValue / Math.pow(1 + wacc, years.length);
  return (ev - netDebt) / dilutedShares;
}

/**
 * Chart path: local FV today at asOf, then year-end roll-forward points strictly
 * after asOf. Past FY-ends are dropped (not shifted to asOf+1) so the line cannot
 * form a one-day V against the today point.
 */
export function buildDcfChartSeries(input: DcfChartSeriesInput): DcfChartPoint[] {
  const asOf = input.asOf.slice(0, 10);
  const localToday = expectedDcfFairValueToday(input);
  const fmp = input.fmpEquityValuePerShare;
  const today =
    localToday != null && localToday > 0
      ? localToday
      : finite(fmp) && fmp > 0
        ? fmp
        : null;

  const points: DcfChartPoint[] = [];
  if (today != null) {
    points.push({ date: asOf, fairValue: today });
  }

  for (const row of expectedDcfFairValueByYear(input)) {
    if (
      row.fairValuePerShare == null ||
      !Number.isFinite(row.fairValuePerShare) ||
      row.fairValuePerShare <= 0
    ) {
      continue;
    }
    const date = fyEndIso(row.year, input.lastHistDate);
    if (!asOf || date <= asOf) continue;
    points.push({ date, year: row.year, fairValue: row.fairValuePerShare });
  }
  return points;
}
