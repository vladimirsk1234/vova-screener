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
 * equityValuePerShare (t = 0) is not returned — callers keep the FMP headline.
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
