/**
 * Card fair-value premium as of a trade open date.
 * Same 5Y Op. EPS trailing metric Results / Value cards persist as `premiumPct`.
 */
import {
  buildCardValuation,
  closeOnOrBefore,
  ttmFromQuarterly,
  type AnnualFundamentalPoint,
} from './fundamentalsValuation.ts';

export type EntryPremiumFields = {
  premiumPctAtEntry: number | null;
  undervaluedAtEntry: boolean | null;
  premiumPctAtEntryAsOf: string | null;
};

/** Stored FMP payload fields the as-of resolver reads. Extra keys are ignored. */
export type EntryPremiumPayload = {
  annual?: Array<Partial<AnnualFundamentalPoint> & { date?: string; year?: number }>;
  quarters?: Array<{
    date?: string;
    operatingEps?: number | null;
    eps?: number | null;
  }>;
  scale?: { reliable?: boolean } | null;
};

function finite(n: unknown): number | null {
  return typeof n === 'number' && Number.isFinite(n) ? n : null;
}

function isoDate(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const d = value.slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(d) ? d : null;
}

function annualFromPartial(
  p: Partial<AnnualFundamentalPoint> & { date?: string; year?: number },
): AnnualFundamentalPoint | null {
  const date = isoDate(p.date);
  const year = typeof p.year === 'number' && Number.isFinite(p.year) ? p.year : null;
  if (!date || year == null) return null;
  return {
    date,
    year,
    price: finite(p.price),
    eps: finite(p.eps),
    operatingEps: finite(p.operatingEps) ?? undefined,
    gaapEps: finite(p.gaapEps) ?? undefined,
    operatingIncome: finite(p.operatingIncome) ?? undefined,
    incomeBeforeTax: finite(p.incomeBeforeTax) ?? undefined,
    incomeTaxExpense: finite(p.incomeTaxExpense) ?? undefined,
    revenuePerShare: finite(p.revenuePerShare),
    fcfPerShare: finite(p.fcfPerShare),
    ownerEarningsPerShare: finite(p.ownerEarningsPerShare),
    pe: finite(p.pe),
    revenue: finite(p.revenue),
    netIncome: finite(p.netIncome),
    operatingCashFlow: finite(p.operatingCashFlow),
    freeCashFlow: finite(p.freeCashFlow),
    dividend: finite(p.dividend) ?? undefined,
    dilutedShares: finite(p.dilutedShares) ?? undefined,
  };
}

/**
 * Card premium as of `asOf`. Uses only annual / quarter prints on or before that
 * date plus `entryPrice` (the trade's open). No payload / unreliable scale / no
 * price / no as-of annuals → `{ premiumPct: null }`.
 */
export function resolvePremiumPctAtEntry(opts: {
  asOf: string;
  entryPrice?: number | null;
  payload?: EntryPremiumPayload | null;
}): { premiumPct: number | null; asOf: string | null } {
  const asOf = isoDate(opts.asOf);
  if (!asOf || !opts.payload) return { premiumPct: null, asOf: null };
  if (opts.payload.scale?.reliable === false) return { premiumPct: null, asOf: null };

  const annual = (opts.payload.annual ?? [])
    .map(annualFromPartial)
    .filter((p): p is AnnualFundamentalPoint => p != null && p.date <= asOf)
    .sort((a, b) => a.year - b.year || a.date.localeCompare(b.date));
  if (!annual.length) return { premiumPct: null, asOf: null };

  const quarters = (opts.payload.quarters ?? [])
    .map((q) => {
      const date = isoDate(q.date);
      if (!date || date > asOf) return null;
      return { date, eps: finite(q.operatingEps) ?? finite(q.eps) };
    })
    .filter((q): q is { date: string; eps: number | null } => q != null);

  const ttm = ttmFromQuarterly(quarters, asOf).ttm;
  const price =
    finite(opts.entryPrice) ??
    closeOnOrBefore(
      annual
        .filter((p) => p.price != null)
        .map((p) => ({ date: p.date, close: p.price as number })),
      asOf,
    );
  if (price == null) return { premiumPct: null, asOf: null };

  const valuation = buildCardValuation(annual, {
    currentPrice: price,
    ttmOperatingEps: ttm != null && ttm > 0 ? ttm : null,
  });
  const premiumPct = finite(valuation.summary.premiumPct);
  return { premiumPct, asOf };
}

export function stampFromResolved(resolved: {
  premiumPct: number | null;
  asOf: string | null;
}): EntryPremiumFields {
  return {
    premiumPctAtEntry: resolved.premiumPct,
    undervaluedAtEntry: resolved.premiumPct == null ? null : resolved.premiumPct < 0,
    premiumPctAtEntryAsOf: resolved.asOf,
  };
}

/**
 * Stamp when a payload exists (even if the number is null). No payload → null so
 * the caller leaves the fields unset (NEW/VALID can still fall back to live).
 */
export function stampIfResolvable(
  asOf: string,
  entryPrice: number | null | undefined,
  payload: EntryPremiumPayload | null | undefined,
): EntryPremiumFields | null {
  if (!payload || !Array.isArray(payload.annual) || !payload.annual.length) return null;
  return stampFromResolved(resolvePremiumPctAtEntry({ asOf, entryPrice, payload }));
}
