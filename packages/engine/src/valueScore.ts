/**
 * Value-screener stars: how many of EPS / FCF / DCF say the name is undervalued.
 * One star is 1/3 (any single metric), not “star 1 = EPS”.
 */

export const VALUE_STAR_TOTAL = 3;

export type ValuePremia = {
  epsPremiumPct: number | null;
  fcfPremiumPct: number | null;
  dcfPremiumPct: number | null;
};

export type ValueScore = {
  epsUndervalued: boolean;
  fcfUndervalued: boolean;
  dcfUndervalued: boolean;
  stars: 0 | 1 | 2 | 3;
};

export type ValueStarsFilter = 'undervalued' | '1' | '2' | '3' | 'all';
export type ValueScreenerSort = 'stars' | 'eps' | 'fcf' | 'dcf' | 'symbol';
export type ValueSortDir = 'asc' | 'desc';

/** Price below fair value. Missing numbers do not count. */
export function isUndervaluedPremium(premiumPct: number | null | undefined): boolean {
  return premiumPct != null && Number.isFinite(premiumPct) && premiumPct < 0;
}

export function scoreValueStars(premia: ValuePremia): ValueScore {
  const epsUndervalued = isUndervaluedPremium(premia.epsPremiumPct);
  const fcfUndervalued = isUndervaluedPremium(premia.fcfPremiumPct);
  const dcfUndervalued = isUndervaluedPremium(premia.dcfPremiumPct);
  const stars = (Number(epsUndervalued) +
    Number(fcfUndervalued) +
    Number(dcfUndervalued)) as 0 | 1 | 2 | 3;
  return { epsUndervalued, fcfUndervalued, dcfUndervalued, stars };
}

/** Most negative finite premium — used as the tie-break after star count. */
export function bestValuePremium(premia: ValuePremia): number | null {
  const vals = [premia.epsPremiumPct, premia.fcfPremiumPct, premia.dcfPremiumPct].filter(
    (n): n is number => n != null && Number.isFinite(n),
  );
  if (!vals.length) return null;
  return Math.min(...vals);
}

export function rowMatchesStarsFilter(stars: number, filter: ValueStarsFilter): boolean {
  if (filter === 'all') return true;
  if (filter === 'undervalued') return stars >= 1;
  return stars === Number(filter);
}

function finiteOrNull(n: number | null): number | null {
  return n != null && Number.isFinite(n) ? n : null;
}

/** Nulls sort last. `sign` is +1 for asc. */
function compareNullable(av: number | null, bv: number | null, sign: number): number {
  if (av == null && bv == null) return 0;
  if (av == null) return 1;
  if (bv == null) return -1;
  if (av === bv) return 0;
  return av < bv ? -sign : sign;
}

export function compareValueRows<
  T extends {
    stars: number;
    symbol: string;
    epsPremiumPct: number | null;
    fcfPremiumPct: number | null;
    dcfPremiumPct: number | null;
    bestPremiumPct: number | null;
  },
>(a: T, b: T, sort: ValueScreenerSort, dir: ValueSortDir): number {
  const sign = dir === 'asc' ? 1 : -1;
  let cmp = 0;
  if (sort === 'stars') {
    if (a.stars !== b.stars) return (a.stars - b.stars) * sign;
    // Same star count: more undervalued (more negative) first, independent of dir.
    cmp = compareNullable(finiteOrNull(a.bestPremiumPct), finiteOrNull(b.bestPremiumPct), 1);
  } else if (sort === 'eps') {
    cmp = compareNullable(finiteOrNull(a.epsPremiumPct), finiteOrNull(b.epsPremiumPct), sign);
  } else if (sort === 'fcf') {
    cmp = compareNullable(finiteOrNull(a.fcfPremiumPct), finiteOrNull(b.fcfPremiumPct), sign);
  } else if (sort === 'dcf') {
    cmp = compareNullable(finiteOrNull(a.dcfPremiumPct), finiteOrNull(b.dcfPremiumPct), sign);
  } else {
    cmp = a.symbol.localeCompare(b.symbol) * sign;
  }
  if (cmp !== 0) return cmp;
  return a.symbol.localeCompare(b.symbol);
}
