/**
 * Value-screener stars: how many of EPS / FCF / DCF / LT D/C say the name is a buy.
 * One star is 1/4 (any single metric), not “star 1 = EPS”.
 */

export const VALUE_STAR_TOTAL = 4;
export const LT_DEBT_CAPITAL_MAX_PCT = 50;

export type ValuePremia = {
  epsPremiumPct: number | null;
  fcfPremiumPct: number | null;
  dcfPremiumPct: number | null;
  ltDebtToCapitalTTM?: number | null;
};

export type ValueScore = {
  epsUndervalued: boolean;
  fcfUndervalued: boolean;
  dcfUndervalued: boolean;
  ltDebtLow: boolean;
  stars: 0 | 1 | 2 | 3 | 4;
};

export type ValueStarsFilter = 'undervalued' | '0' | '1' | '2' | '3' | '4' | 'all' | 'garp';

/** Colton Growth-style GARP: 10Y EPS CAGR ≥ 15%, forward ≥ 10%, EPS below fair value. */
export const GARP_TRAILING_MIN_PCT = 15;
export const GARP_FORWARD_MIN_PCT = 10;
export type ValueScreenerSort = 'stars' | 'eps' | 'fcf' | 'dcf' | 'symbol' | 'interest';
export type ValueSortDir = 'asc' | 'desc';
export type ValueInterest = 'interested' | 'not_interested';

/** Same ranks Results uses on tracked signals: interested 2, unmarked 1, not interested 0. */
export const VALUE_INTEREST_RANK: Record<ValueInterest | 'none', number> = {
  interested: 2,
  none: 1,
  not_interested: 0,
};

export function interestRankOf(interest: ValueInterest | null | undefined): number {
  return VALUE_INTEREST_RANK[interest ?? 'none'];
}

/** Price below fair value. Missing numbers do not count. */
export function isUndervaluedPremium(premiumPct: number | null | undefined): boolean {
  return premiumPct != null && Number.isFinite(premiumPct) && premiumPct < 0;
}

/** FMP mixes decimals (0.18) and whole percents (18); normalize to percent points. */
export function normalizePctPoints(n: number | null | undefined): number | null {
  if (n == null || !Number.isFinite(n)) return null;
  return Math.abs(n) <= 1.5 ? n * 100 : n;
}

/** Long-term debt / capital below 50%. Missing numbers do not count. */
export function isLowLtDebt(ltDebtToCapitalTTM: number | null | undefined): boolean {
  const pct = normalizePctPoints(ltDebtToCapitalTTM);
  return pct != null && pct < LT_DEBT_CAPITAL_MAX_PCT;
}

export function scoreValueStars(premia: ValuePremia): ValueScore {
  const epsUndervalued = isUndervaluedPremium(premia.epsPremiumPct);
  const fcfUndervalued = isUndervaluedPremium(premia.fcfPremiumPct);
  const dcfUndervalued = isUndervaluedPremium(premia.dcfPremiumPct);
  const ltDebtLow = isLowLtDebt(premia.ltDebtToCapitalTTM);
  const stars = (Number(epsUndervalued) +
    Number(fcfUndervalued) +
    Number(dcfUndervalued) +
    Number(ltDebtLow)) as 0 | 1 | 2 | 3 | 4;
  return { epsUndervalued, fcfUndervalued, dcfUndervalued, ltDebtLow, stars };
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
  if (filter === 'all' || filter === 'garp') return true;
  if (filter === 'undervalued') return stars >= 1;
  return stars === Number(filter);
}

export function isGarpCandidate(opts: {
  growth10yPct: number | null | undefined;
  forwardGrowthPct: number | null | undefined;
  epsPremiumPct: number | null | undefined;
}): boolean {
  return (
    opts.growth10yPct != null &&
    Number.isFinite(opts.growth10yPct) &&
    opts.growth10yPct >= GARP_TRAILING_MIN_PCT &&
    opts.forwardGrowthPct != null &&
    Number.isFinite(opts.forwardGrowthPct) &&
    opts.forwardGrowthPct >= GARP_FORWARD_MIN_PCT &&
    isUndervaluedPremium(opts.epsPremiumPct)
  );
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
    interestRank?: number | null;
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
  } else if (sort === 'interest') {
    const ar = a.interestRank ?? VALUE_INTEREST_RANK.none;
    const br = b.interestRank ?? VALUE_INTEREST_RANK.none;
    if (ar !== br) return (ar - br) * sign;
    if (a.stars !== b.stars) return b.stars - a.stars;
    cmp = compareNullable(finiteOrNull(a.bestPremiumPct), finiteOrNull(b.bestPremiumPct), 1);
  } else {
    cmp = a.symbol.localeCompare(b.symbol) * sign;
  }
  if (cmp !== 0) return cmp;
  return a.symbol.localeCompare(b.symbol);
}
