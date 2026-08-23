/**
 * Fast Graphs–style forecast math: future price = horizon EPS × target multiple,
 * annualized ROR over the actual estimate horizon (not a fixed 5 years).
 * Pure math — no I/O.
 */

function finite(n: unknown): n is number {
  return typeof n === 'number' && Number.isFinite(n);
}

function isoDayDiff(fromIso: string, toIso: string): number {
  const a = Date.parse(`${fromIso.slice(0, 10)}T00:00:00Z`);
  const b = Date.parse(`${toIso.slice(0, 10)}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return Number.POSITIVE_INFINITY;
  return (b - a) / 86_400_000;
}

/** Floor so a FY-end a few months away does not explode the annualized number. */
export const MIN_FORECAST_HORIZON_YEARS = 0.5;

export function futurePriceAt(
  epsHorizon: number | null | undefined,
  targetMultiple: number | null | undefined,
): number | null {
  if (!finite(epsHorizon) || !finite(targetMultiple) || epsHorizon <= 0 || targetMultiple <= 0) {
    return null;
  }
  return epsHorizon * targetMultiple;
}

/** (FV − price) / FV. Positive = margin of safety (price below fair value). */
export function marginOfSafetyPct(
  price: number | null | undefined,
  fairValue: number | null | undefined,
): number | null {
  if (!finite(price) || !finite(fairValue) || fairValue <= 0) return null;
  return ((fairValue - price) / fairValue) * 100;
}

export function forecastHorizonYears(
  asOfIso: string,
  estimateFyEndIso: string | null | undefined,
): number | null {
  if (!estimateFyEndIso) return null;
  const days = isoDayDiff(asOfIso.slice(0, 10), estimateFyEndIso.slice(0, 10));
  if (!Number.isFinite(days) || days <= 0) return null;
  return Math.max(MIN_FORECAST_HORIZON_YEARS, days / 365.25);
}

/**
 * Chuck / FG Forecasting: (P_future / P_now)^(1/years) − 1 + dividend yield.
 * `divYieldPct` is already in percent points (1.2 = 1.2%).
 */
export function annualizedRorPct(
  price: number | null | undefined,
  futurePrice: number | null | undefined,
  years: number | null | undefined,
  divYieldPct: number | null | undefined = 0,
): number | null {
  if (!finite(price) || !finite(futurePrice) || !finite(years)) return null;
  if (price <= 0 || futurePrice <= 0 || years <= 0) return null;
  const priceRor = (Math.pow(futurePrice / price, 1 / years) - 1) * 100;
  const div = finite(divYieldPct) ? divYieldPct : 0;
  return priceRor + div;
}

export type HorizonEstimate = {
  year: number;
  date: string | null;
  eps: number;
};

export function pickHorizonEstimate(
  estimates: Array<{ year: number; date?: string | null; eps?: number | null; metric?: number | null }>,
): HorizonEstimate | null {
  const last = [...estimates]
    .filter((e) => {
      const eps = finite(e.eps) ? e.eps : e.metric;
      return finite(eps) && (eps as number) > 0 && Number.isFinite(e.year);
    })
    .sort((a, b) => a.year - b.year)
    .pop();
  if (!last) return null;
  const eps = (finite(last.eps) ? last.eps : last.metric) as number;
  return {
    year: last.year,
    date: last.date ? String(last.date).slice(0, 10) : null,
    eps,
  };
}

export type ForecastScenarios = {
  horizonYears: number | null;
  horizonDate: string | null;
  horizonEps: number | null;
  futurePricePeg: number | null;
  futurePriceNormal: number | null;
  futurePriceCustom: number | null;
  rorPegPct: number | null;
  rorNormalPct: number | null;
  rorCustomPct: number | null;
  marginOfSafetyPct: number | null;
};

export function buildForecastScenarios(opts: {
  price: number | null | undefined;
  fairValue: number | null | undefined;
  fairValueRatio: number | null | undefined;
  normalMultiple: number | null | undefined;
  customMultiple?: number | null;
  dividendYieldPct?: number | null;
  estimates: Array<{ year: number; date?: string | null; eps?: number | null; metric?: number | null }>;
  asOfIso?: string;
}): ForecastScenarios {
  const asOf = opts.asOfIso?.slice(0, 10) ?? new Date().toISOString().slice(0, 10);
  const horizon = pickHorizonEstimate(opts.estimates);
  const horizonDate = horizon?.date ?? (horizon ? `${horizon.year}-12-31` : null);
  const years = forecastHorizonYears(asOf, horizonDate);
  const div = opts.dividendYieldPct ?? 0;
  const futurePeg = futurePriceAt(horizon?.eps ?? null, opts.fairValueRatio ?? null);
  const futureNormal = futurePriceAt(horizon?.eps ?? null, opts.normalMultiple ?? null);
  const futureCustom = futurePriceAt(horizon?.eps ?? null, opts.customMultiple ?? null);
  return {
    horizonYears: years,
    horizonDate,
    horizonEps: horizon?.eps ?? null,
    futurePricePeg: futurePeg,
    futurePriceNormal: futureNormal,
    futurePriceCustom: futureCustom,
    rorPegPct: annualizedRorPct(opts.price, futurePeg, years, div),
    rorNormalPct: annualizedRorPct(opts.price, futureNormal, years, div),
    rorCustomPct: annualizedRorPct(opts.price, futureCustom, years, div),
    marginOfSafetyPct: marginOfSafetyPct(opts.price, opts.fairValue),
  };
}

export type EarningsPrint = {
  date: string | null;
  epsActual: number | null;
  epsEstimated: number | null;
};

export type BeatMissBucket = {
  beat: number;
  meet: number;
  miss: number;
  total: number;
  beatPct: number | null;
  missPct: number | null;
};

const MEET_EPS = 0.005;

export function emptyBeatMiss(): BeatMissBucket {
  return { beat: 0, meet: 0, miss: 0, total: 0, beatPct: null, missPct: null };
}

export function scoreAnalystBeats(
  rows: EarningsPrint[],
  lookbackYears: number,
  asOfIso = new Date().toISOString().slice(0, 10),
): BeatMissBucket {
  const asOf = asOfIso.slice(0, 10);
  const cutoffMs = Date.parse(`${asOf}T00:00:00Z`) - lookbackYears * 365.25 * 86_400_000;
  const bucket = emptyBeatMiss();
  for (const r of rows) {
    const d = r.date?.slice(0, 10) ?? '';
    if (d.length !== 10 || d > asOf) continue;
    const t = Date.parse(`${d}T00:00:00Z`);
    if (!Number.isFinite(t) || t < cutoffMs) continue;
    if (!finite(r.epsActual) || !finite(r.epsEstimated)) continue;
    bucket.total += 1;
    const diff = r.epsActual - r.epsEstimated;
    if (Math.abs(diff) <= MEET_EPS) bucket.meet += 1;
    else if (diff > 0) bucket.beat += 1;
    else bucket.miss += 1;
  }
  if (bucket.total > 0) {
    bucket.beatPct = (bucket.beat / bucket.total) * 100;
    bucket.missPct = (bucket.miss / bucket.total) * 100;
  }
  return bucket;
}

export type DividendCoverStatus = 'covered' | 'thin' | 'uncovered' | 'none';

export type DividendCoverage = {
  ocfCover: number | null;
  fcfCover: number | null;
  status: DividendCoverStatus;
};

function coverRatio(cash: number | null | undefined, dps: number | null | undefined, shares: number | null | undefined): number | null {
  if (!finite(cash) || !finite(dps) || !finite(shares) || dps <= 0 || shares <= 0) return null;
  const paid = dps * shares;
  if (paid <= 0) return null;
  return cash / paid;
}

export function dividendCoverage(point: {
  dividend?: number | null;
  dilutedShares?: number | null;
  operatingCashFlow?: number | null;
  freeCashFlow?: number | null;
}): DividendCoverage {
  const dps = point.dividend;
  if (!finite(dps) || dps <= 0) {
    return { ocfCover: null, fcfCover: null, status: 'none' };
  }
  const ocfCover = coverRatio(point.operatingCashFlow, dps, point.dilutedShares);
  const fcfCover = coverRatio(point.freeCashFlow, dps, point.dilutedShares);
  let status: DividendCoverStatus = 'none';
  const best = [ocfCover, fcfCover].filter((n): n is number => n != null);
  if (best.length) {
    const max = Math.max(...best);
    if (max >= 1.5) status = 'covered';
    else if (max >= 1) status = 'thin';
    else status = 'uncovered';
  }
  return { ocfCover, fcfCover, status };
}
