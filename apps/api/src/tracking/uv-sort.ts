/**
 * Results / History UV sort. Prefer the trade's `premiumPctAtEntry` so the
 * order matches the Settings UV/OV filter; live card premia fill gaps.
 */

export type UvPremia = {
  bestPremiumPct?: number | null;
  epsPremiumPct?: number | null;
};

export type UvRow = {
  yahooTicker: string;
  symbol: string;
  premiumPctAtEntry?: number | null;
};

function finite(n: number | null | undefined): number | null {
  return n != null && Number.isFinite(n) ? n : null;
}

function lookup<T>(ticker: string, cards: Record<string, T | undefined>): T | undefined {
  return cards[ticker] ?? cards[ticker.toUpperCase()];
}

/** Sort key: most undervalued = lowest premium. Entry snapshot wins over live cards. */
export function uvSortKey(
  card: UvPremia | undefined,
  entryPremium?: number | null,
): number | null {
  const entry = finite(entryPremium);
  if (entry != null) return entry;
  const best = finite(card?.bestPremiumPct);
  if (best != null) return best;
  return finite(card?.epsPremiumPct);
}

function compareNullable(av: number | null, bv: number | null, sign: number): number {
  if (av == null && bv == null) return 0;
  if (av == null) return 1;
  if (bv == null) return -1;
  if (av === bv) return 0;
  return av < bv ? -sign : sign;
}

export function compareUvRows<T extends UvRow>(
  a: T,
  b: T,
  cards: Record<string, UvPremia | undefined>,
  dir: 'asc' | 'desc',
): number {
  const sign = dir === 'asc' ? 1 : -1;
  const aCard = lookup(a.yahooTicker, cards);
  const bCard = lookup(b.yahooTicker, cards);
  const byBest = compareNullable(
    uvSortKey(aCard, a.premiumPctAtEntry),
    uvSortKey(bCard, b.premiumPctAtEntry),
    sign,
  );
  if (byBest !== 0) return byBest;
  const byEps = compareNullable(finite(aCard?.epsPremiumPct), finite(bCard?.epsPremiumPct), sign);
  if (byEps !== 0) return byEps;
  return a.symbol.localeCompare(b.symbol);
}

export function sortByUndervaluation<T extends UvRow>(
  rows: T[],
  cards: Record<string, UvPremia | undefined>,
  dir: 'asc' | 'desc',
): T[] {
  return [...rows].sort((a, b) => compareUvRows(a, b, cards, dir));
}
