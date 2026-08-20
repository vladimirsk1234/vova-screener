/** Apply listing-currency + ADR scaling to FMP rows before valuation. */
import {
  buildFundamentalsScale,
  fallbackForeignPerUsd,
  fxToListingMultiplier,
  normalizeCurrency,
  pickScaledEps,
  scaleCompany,
  scaleDividend,
  scalePerShare,
  type FundamentalsScale,
} from '@vova/engine';
import type { AnnualFundamentalPoint } from '@vova/engine';
import { fmpNum, fmpStr, type FmpClient } from '../market/fmp.client';

export type IncomeScaleAnchor = {
  year: number;
  netIncome: number | null;
  fmpEps: number | null;
  dilutedShares: number | null;
  reportedCurrency: string | null;
};

export function incomeAnchor(rows: Record<string, unknown>[]): IncomeScaleAnchor | null {
  const sorted = [...rows].sort((a, b) => {
    const da = fmpStr(a.date) ?? '';
    const db = fmpStr(b.date) ?? '';
    return db.localeCompare(da);
  });
  for (const row of sorted) {
    const date = fmpStr(row.date) ?? fmpStr(row.calendarYear);
    const year = date ? Number(date.slice(0, 4)) : Number(fmpStr(row.calendarYear));
    const fmpEps =
      fmpNum(row.epsdiluted) ?? fmpNum(row.epsDiluted) ?? fmpNum(row.eps);
    const netIncome = fmpNum(row.netIncome);
    const dilutedShares =
      fmpNum(row.weightedAverageShsOutDil) ?? fmpNum(row.weightedAverageShsOut);
    if (!Number.isFinite(year)) continue;
    if (fmpEps == null && netIncome == null) continue;
    return {
      year,
      netIncome,
      fmpEps,
      dilutedShares,
      reportedCurrency: normalizeCurrency(fmpStr(row.reportedCurrency)),
    };
  }
  return null;
}

export function mostCommonReportedCurrency(rows: Record<string, unknown>[]): string | null {
  const counts = new Map<string, number>();
  for (const row of rows) {
    const c = normalizeCurrency(fmpStr(row.reportedCurrency));
    if (!c) continue;
    counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  let best: string | null = null;
  let n = 0;
  for (const [c, k] of counts) {
    if (k > n) {
      best = c;
      n = k;
    }
  }
  return best;
}

export async function fxToListingByYear(
  fmp: FmpClient,
  reported: string | null,
  listing: string | null,
  years: number[],
): Promise<Map<number, number>> {
  const out = new Map<number, number>();
  const from = normalizeCurrency(reported);
  const to = normalizeCurrency(listing) ?? 'USD';
  if (!years.length || !from || from === to) {
    for (const y of years) out.set(y, 1);
    return out;
  }

  const [fromRates, toRates] = await Promise.all([
    fmp.forexForeignPerUsd(from, years),
    to === 'USD' ? Promise.resolve(new Map<number, number>()) : fmp.forexForeignPerUsd(to, years),
  ]);

  const lookup = (cur: string, y: number, fetched: Map<number, number>): number => {
    const hit = fetched.get(y);
    if (hit != null && hit > 0) return hit;
    const latest = [...fetched.entries()].sort((a, b) => b[0] - a[0])[0];
    if (latest && latest[1] > 0) return latest[1];
    return fallbackForeignPerUsd(cur);
  };

  for (const y of years) {
    const m = fxToListingMultiplier(from, to, (c) => {
      if (c === from) return lookup(from, y, fromRates);
      if (c === to) return lookup(to, y, toRates);
      return fallbackForeignPerUsd(c);
    });
    out.set(y, m);
  }
  return out;
}

export function buildScaleForTicker(opts: {
  ticker: string;
  listingCurrency: string | null;
  reportedCurrency: string | null;
  fxToListing: number;
  netIncome: number | null;
  fmpEps: number | null;
  dilutedShares: number | null;
  price: number | null;
  peTtm?: number | null;
}): FundamentalsScale {
  return buildFundamentalsScale({
    ticker: opts.ticker,
    reportedCurrency: opts.reportedCurrency,
    listingCurrency: opts.listingCurrency,
    netIncome: opts.netIncome,
    fmpEps: opts.fmpEps,
    dilutedShares: opts.dilutedShares,
    price: opts.price,
    fxToListing: opts.fxToListing,
    peTtm: opts.peTtm,
  });
}

export function scaleAnnualPoint(
  point: AnnualFundamentalPoint,
  scale: FundamentalsScale,
  yearFx?: number,
  peTtm?: number | null,
): AnnualFundamentalPoint {
  const yearScale =
    yearFx != null && yearFx > 0 && yearFx !== scale.fxToListing
      ? { ...scale, fxToListing: yearFx }
      : scale;
  const eps = pickScaledEps({
    fmpEps: point.eps,
    netIncome: point.netIncome,
    dilutedShares: point.dilutedShares ?? null,
    scale: yearScale,
    price: point.price,
    peTtm,
  });
  const price = point.price;
  const pe =
    price != null && eps != null && eps > 0 && price > 0 ? price / eps : point.pe;
  return {
    ...point,
    eps,
    revenuePerShare: scalePerShare(point.revenuePerShare, yearScale),
    fcfPerShare: scalePerShare(point.fcfPerShare, yearScale),
    ownerEarningsPerShare: scalePerShare(point.ownerEarningsPerShare, yearScale),
    pe,
    revenue: scaleCompany(point.revenue, yearScale),
    netIncome: scaleCompany(point.netIncome, yearScale),
    operatingCashFlow: scaleCompany(point.operatingCashFlow, yearScale),
    freeCashFlow: scaleCompany(point.freeCashFlow, yearScale),
    dividend: scaleDividend(point.dividend, price, yearScale),
  };
}

export function hasCurrentScale(payload: { scale?: { version?: number } | null } | null | undefined, version: number): boolean {
  return payload?.scale?.version === version;
}
