/**
 * Roll-forward DCF fair value at the end of each forecast year.
 * Pure math — Custom DCF cash flows come from FMP via the API layer.
 *
 * FMP `/custom-discounted-cash-flow` mixes ~5 history years with ~5 forecast
 * years (strings, newest first). Only year > last completed FY is future UFCF.
 * Discounting 2021–2025 as if they were t=1…n understates today vs FMP.
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
  /**
   * Last complete FY-end (YYYY-MM-DD) or a year-like value. Forecast years
   * are those with year > this FY (and > max historical year in `years`).
   */
  lastHistDate?: string | null;
};

export type DcfYearFairValue = {
  year: number;
  fairValuePerShare: number | null;
};

export type DcfChartSeriesInput = DcfFairValueInput & {
  asOf: string;
  /** FMP headline — used only when the local t=0 model cannot run. */
  fmpEquityValuePerShare?: number | null;
};

export type DcfChartPoint = {
  date: string;
  /** FY-end year for chart markers; omitted on the today point. */
  year?: number;
  fairValue: number;
};

function finite(n: unknown): n is number {
  return typeof n === 'number' && Number.isFinite(n);
}

/** FMP sends year as a string ("2026"); lastHistDate is YYYY-MM-DD. */
export function dcfYearNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return Math.trunc(value);
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const direct = Number(trimmed);
    if (Number.isFinite(direct) && Math.abs(direct) >= 1000) return Math.trunc(direct);
    const m = /(\d{4})/.exec(trimmed);
    return m ? Number(m[1]) : null;
  }
  return null;
}

function fyEndIso(year: number, lastHistDate?: string | null): string {
  const md = lastHistDate && /^\d{4}-(\d{2}-\d{2})/.exec(lastHistDate.slice(0, 10));
  return `${year}-${md?.[1] ?? '12-31'}`;
}

/**
 * Last completed fiscal year: year of `lastHistDate`, or the max year in
 * `years` that is on or before that FY (history rows in the FMP payload).
 */
export function lastCompletedFiscalYear(input: Pick<DcfFairValueInput, 'lastHistDate' | 'years'>): number | null {
  const fromDate = dcfYearNumber(input.lastHistDate);
  const years = input.years
    .map((r) => dcfYearNumber(r.year))
    .filter((y): y is number => y != null);
  if (fromDate != null) {
    const historical = years.filter((y) => y <= fromDate);
    return historical.length ? Math.max(...historical) : fromDate;
  }
  return null;
}

/** Oldest-first forecast rows only (year > last completed FY). */
export function forecastDcfYears(input: DcfFairValueInput): DcfYearInput[] {
  const rows: DcfYearInput[] = [];
  for (const r of input.years) {
    const year = dcfYearNumber(r.year);
    if (year == null) continue;
    rows.push({ year, ufcf: r.ufcf });
  }
  rows.sort((a, b) => a.year - b.year);
  const lastFy = lastCompletedFiscalYear({ lastHistDate: input.lastHistDate, years: rows });
  if (lastFy == null) return rows;
  return rows.filter((r) => r.year > lastFy);
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
  const years = forecastDcfYears(input);
  const { wacc, terminalValue, netDebt, dilutedShares } = input;
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
  const years = forecastDcfYears(input);
  const { wacc, terminalValue, netDebt, dilutedShares } = input;
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
